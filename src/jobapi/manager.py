import asyncio
import logging
import shutil
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from logging import Logger, getLogger
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Iterator, Optional
from uuid import UUID, uuid4

from edps import Service
from edps.compression.zip import ZipAlgorithm
from edps.file import build_real_sub_path, sanitize_path
from edps.service import get_report_path
from edps.taskcontextimpl import TaskContextImpl
from jobapi.config import AppConfig
from jobapi.exception import ApiClientException
from jobapi.repo import Job, JobRepository
from jobapi.types import JobData, JobState, JobView


class AnalysisJobManager:
    def __init__(self, app_config: AppConfig, job_repo: JobRepository):
        self._app_config = app_config
        self._job_repo = job_repo
        self._logger = getLogger(__name__)
        self._service = Service()

    async def create_job(self, job_data: JobData) -> UUID:
        """Create a job based on the user provided EDP data.
        The job gets an ID and a job working directory and is initially in state 'WAITING_FOR_DATA'.
        Returns Job-ID.
        """

        async with self._job_repo.new_session() as session:
            job_id = uuid4()

            # Create job dir
            job_base_dir = self._app_config.working_dir / str(job_id)
            job_base_dir.mkdir(parents=True)

            await session.create_job(
                job_id=job_id,
                job_data=job_data,
                job_base_dir=job_base_dir,
            )

            self._logger.info("Job created: %s", job_id)
            return job_id

    async def get_job_view(self, job_id: UUID) -> JobView:
        """Get a JobView by ID."""

        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            return job.to_job_view()

    async def get_zipped_result(self, job_id: UUID) -> Path:
        """If an analysis job has reached state COMPLETED, this call returns the path to the zipped EDP including images."""

        async with self._job_repo.new_session() as session:
            job: Job = await session.get_job(job_id)
            if job.state != JobState.COMPLETED:
                raise ApiClientException(f"There is no result because job is in state {job.state.value}.")
            return job.zip_archive

    async def get_log_file(self, job_id: UUID) -> Path:
        """This call returns the path to the job log file, no matter if and how far the job has been processed."""

        async with self._job_repo.new_session() as session:
            job: Job = await session.get_job(job_id)
            return job.log_file

    async def get_report_file(self, job_id: UUID) -> Path:
        """If an analysis job has reached state COMPLETED, this call returns the path to the report file."""

        async with self._job_repo.new_session() as session:
            job: Job = await session.get_job(job_id)
            if job.state != JobState.COMPLETED:
                raise ApiClientException(f"There is no report because job is in state {job.state.value}.")
            return job.report_file

    async def process_job(self, job_id: UUID):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """

        # First check the job and put it in PROCESSING state.
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            if job.state != JobState.QUEUED:
                raise ApiClientException(f"Job can't be processed because it's in state {job.state.value}.")
            job.update_state(JobState.PROCESSING)
            job.started = datetime.now(tz=UTC)

        # In a new session do the actual processing.
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            self._logger.info("Starting job %s...", job_id)
            with init_file_logger(job.log_file) as job_logger:
                process_job_task = asyncio.create_task(self._process_job_worker(job, job_logger))
                cancellation_listener_task = asyncio.create_task(self._cancellation_listener(job_id))

                try:
                    # Wait until the job has completed normally or it has been canceled.
                    done, _ = await asyncio.wait(
                        [process_job_task, cancellation_listener_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    # This re-raises any exceptions from the completed tasks.
                    for task in done:
                        task.result()

                    job_logger.info("EDP created successfully.")
                    self._logger.info("Job %s completed.", job.job_id)
                    job.update_state(JobState.COMPLETED)

                except asyncio.CancelledError:
                    job_logger.info("Job was cancelled.")
                    self._logger.info("Job %s cancelled.", job.job_id)
                    job.update_state(JobState.CANCELLED, "Analysis was cancelled.")

                except Exception as exception:
                    job_logger.info("Job has failed.")
                    self._logger.error("Job %s has failed.", job.job_id, exc_info=exception)
                    job.update_state(JobState.FAILED, f"Processing failed: {exception}")

                finally:
                    job.finished = datetime.now(tz=UTC)
                    process_job_task.cancel()
                    cancellation_listener_task.cancel()

    async def _process_job_worker(self, job: Job, job_logger: Logger):
        self._logger.debug("Job data directory: %s", job.job_base_dir)
        with TemporaryDirectory() as temp_working_dir:
            self._logger.debug("Temporary working directory: %s", temp_working_dir)
            ctx = TaskContextImpl(job.configuration, job_logger, Path(temp_working_dir))
            shutil.copytree(job.input_data_dir, ctx.input_path, dirs_exist_ok=True)
            main_ref = job.user_provided_edp_data.assetRefs[0]
            job_logger.info("Analysing asset '%s' version '%s'...", main_ref.assetId, main_ref.assetVersion)
            await self._service.analyse_asset(ctx, job.user_provided_edp_data)
            await ZipAlgorithm().compress(ctx.output_path, job.zip_archive)
            if get_report_path(ctx).exists():
                shutil.copy(get_report_path(ctx), job.report_file)

    async def _cancellation_listener(self, job_id: UUID):
        while True:
            async with self._job_repo.new_session() as session:
                job = await session.get_job(job_id)
                if job.state == JobState.CANCELLATION_REQUESTED:
                    raise asyncio.CancelledError()
                await asyncio.sleep(1)

    async def cancel_job(self, job_id: UUID):
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            if job.state not in [JobState.PROCESSING, JobState.QUEUED]:
                raise ApiClientException(f"Job cannot be cancelled because it's in state {job.state.value}.")

            # Update state to CANCELLATION_REQUEST which is handled by _cancellation_listener
            self._logger.info("Job %s marked for cancellation.", job.job_id)
            job.update_state(JobState.CANCELLATION_REQUESTED, "Cancelling job")

    async def store_input_file(self, job_id: UUID, filename: Optional[str], file):
        """Store uploaded job data which will be analyzed later.
        The content of the given file (either a TemporaryFile or a FastAPI UploadFile) is copied to a file named 'filename' in the job working dir.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. If an error occurs this needs to be repeated.
        After successul upload the state changes to 'QUEUED'.
        """

        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)

            if job.state != JobState.WAITING_FOR_DATA:
                raise ApiClientException(
                    f"Job doesn't accept any file uploads because it's in state {job.state.value}."
                )

            filename = sanitize_path(filename or "")
            if not filename:
                raise ApiClientException("Filename is missing!")

            job.input_data_dir.mkdir(parents=True, exist_ok=True)
            data_file_path = build_real_sub_path(job.input_data_dir, filename)

            try:
                with data_file_path.open("wb") as dest:
                    copyfileobj(file, dest)
                if data_file_path.stat().st_size == 0:
                    raise ApiClientException("Upload was empty!")
            except:
                # If there is an error delete the file.
                data_file_path.unlink()
                raise

            job.update_state(JobState.QUEUED)

            self._logger.info(
                "File upload for job %s is complete: %s (%s bytes)",
                job.job_id,
                data_file_path,
                data_file_path.stat().st_size,
            )


@contextmanager
def init_file_logger(log_path: Path) -> Iterator[Logger]:
    """Create a new logger that logs to the given file. File is closed on contextmanager exit."""
    logger = getLogger("edps.jobapi")
    logger.setLevel(logging.DEBUG)
    with closing(logging.FileHandler(log_path)) as file_handler:
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s - %(message)s"))
        logger.addHandler(file_handler)
        yield logger
