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
from edps.taskcontextimpl import TaskContextImpl
from edps.types import Config, UserProvidedEdpData
from jobapi.config import AppConfig
from jobapi.repo import Job, JobRepository
from jobapi.types import JobState, JobView


class AnalysisJobManager:
    def __init__(self, app_config: AppConfig, job_repo: JobRepository):
        self._app_config = app_config
        self._job_repo = job_repo
        self._logger = getLogger(__name__)
        self._service = Service()

    async def create_job(self, user_data: UserProvidedEdpData) -> UUID:
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
                user_data=user_data,
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
                raise RuntimeError(f"There is no result for job {job.job_id}")
            return job.zip_archive

    async def get_log_file(self, job_id: UUID) -> Path:
        """This call returns the path to the job log file, no matter if and how far the job has been processed."""

        async with self._job_repo.new_session() as session:
            job: Job = await session.get_job(job_id)
            return job.log_file

    async def process_job(self, job_id: UUID):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """

        # First check the job and put it in PROCESSING state.
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            if job.state != JobState.QUEUED:
                raise RuntimeError(f"Job can't be processed because it's in state {job.state}.")
            job.update_state(JobState.PROCESSING)
            job.started = datetime.now(tz=UTC)

        # In a new session do the actual processing.
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)
            self._logger.info("Starting job %s...", job_id)
            self._logger.debug("Job data directory: %s", job.job_base_dir)
            try:
                with (
                    TemporaryDirectory() as temp_working_dir,
                    init_file_logger(job.log_file) as job_logger,
                ):
                    self._logger.debug("Temporary working directory: %s", temp_working_dir)
                    user_data = job.user_data
                    config = Config(userProvidedEdpData=user_data)
                    ctx = TaskContextImpl(config, job_logger, Path(temp_working_dir))
                    shutil.copytree(job.input_data_dir, ctx.input_path, dirs_exist_ok=True)
                    job_logger.info("Analysing asset '%s' version '%s'...", user_data.assetId, user_data.version)
                    await self._service.analyse_asset(ctx)
                    await ZipAlgorithm().compress(ctx.output_path, job.zip_archive)
                    job_logger.info("EDP created successfully")
                    job.update_state(JobState.COMPLETED)
                    job.finished = datetime.now(tz=UTC)
                    self._logger.info("Job %s completed.", job.job_id)

            except Exception as exception:
                job.update_state(JobState.FAILED, f"Processing failed: {exception}")
                job.finished = datetime.now(tz=UTC)
                self._logger.error("Job %s has failed", job.job_id, exc_info=exception)

    async def store_input_file(self, job_id: UUID, filename: Optional[str], file):
        """Store uploaded job data which will be analyzed later.
        The content of the given file (either a TemporaryFile or a FastAPI UploadFile) is copied to a file named 'filename' in the job working dir.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. If an error occurs this needs to be repeated.
        After successul upload the state changes to 'QUEUED'.
        """

        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)

            if job.state != JobState.WAITING_FOR_DATA:
                raise RuntimeError(f"Job doesn't accept any file uploads because it's in state {job.state}.")
            if not filename:
                raise RuntimeError("Filename is missing!")

            job.input_data_dir.mkdir(parents=True, exist_ok=True)
            data_file_path = job.input_data_dir / filename

            try:
                with data_file_path.open("wb") as dest:
                    copyfileobj(file, dest)
                if data_file_path.stat().st_size == 0:
                    raise RuntimeError("Upload was empty!")
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
