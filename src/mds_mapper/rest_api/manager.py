import asyncio
import logging
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from logging import Logger, getLogger
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Iterator, Optional
from uuid import UUID, uuid4

from mds_mapper import convert_edp_to_mds
from mds_mapper.compression.zip import ZipAlgorithm
from mds_mapper.file import build_real_sub_path, sanitize_path
from mds_mapper.rest_api.repo import Job, JobRepository
from mds_mapper.rest_api.repo.base import JobRepositoryFactory
from mds_mapper.rest_api.repo.inmemory import InMemoryJobRepositoryFactory
from mds_mapper.rest_api.repo.persistent import DbJobRepositoryFactory

from .config import AppConfig
from .exception import ApiClientException
from .types import END_STATES, Config, JobState, JobView

logger = getLogger(__name__)


class AnalysisJobManager:
    def __init__(self, app_config: AppConfig):
        self._app_config = app_config
        self._executor = create_executor(app_config)
        self._job_repo_factory = create_job_repository_factory(app_config)
        self._job_repo = self._job_repo_factory.create()
        logger.info("Started MDS Mapper")

    async def create_job(self, config: Config) -> UUID:
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
                config=config,
                job_base_dir=job_base_dir,
            )

            logger.info("Job created: %s", job_id)
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

    async def process_job(self, job_id: UUID):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """

        processor = AnalysisJobProcessor(self._job_repo_factory, job_id)
        if self._executor is None:
            await processor.run_async()
        else:
            logger.info("Launching job %s with executor.", job_id)
            self._executor.submit(processor.run)

    async def cancel_job(self, job_id: UUID):
        async with self._job_repo.new_session() as session:
            job = await session.get_job(job_id)

            if job.state in END_STATES:
                raise ApiClientException(f"Job cannot be cancelled because it's in state {job.state.value}.")

            # Update state to CANCELLATION_REQUEST which is handled by _cancellation_listener
            logger.info("Job %s marked for cancellation.", job.job_id)
            job.set_state(JobState.CANCELLATION_REQUESTED, "Cancelling job")

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

            job.set_state(JobState.QUEUED)

            logger.info(
                "File upload for job %s is complete: %s (%s bytes)",
                job.job_id,
                data_file_path,
                data_file_path.stat().st_size,
            )


@contextmanager
def init_file_logger(job_id: UUID, log_path: Path) -> Iterator[Logger]:
    """Create a new logger that logs to the given file. File is closed on contextmanager exit."""
    logger = getLogger(f"mds_mapper.jobapi.{job_id}")
    logger.setLevel(logging.DEBUG)
    with closing(logging.FileHandler(log_path)) as file_handler:
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s - %(message)s"))
        logger.addHandler(file_handler)
        yield logger


def create_executor(app_config: AppConfig) -> Optional[Executor]:
    workers = app_config.workers
    db_url = app_config.db_url
    if workers == 0:
        logger.info("No executor pool is used for job processing.")
        return None
    elif db_url is None:
        logger.info("ThreadPoolExecutor(%d) is used for job processing.", workers)
        return ThreadPoolExecutor(workers)
    else:
        logger.info("ProcessPoolExecutor(%d) is used for job processing.", workers)
        return ProcessPoolExecutor(workers)


def create_job_repository_factory(app_config: AppConfig) -> JobRepositoryFactory:
    db_url = app_config.db_url
    if db_url is None:
        logger.info("Using InMemoryJobRepositoryFactory")
        return InMemoryJobRepositoryFactory()
    else:
        logger.info("Using DbJobRepositoryFactory")
        return DbJobRepositoryFactory(str(db_url))


class AnalysisJobProcessor:
    def __init__(self, job_repo_factory: JobRepositoryFactory, job_id: UUID):
        self._job_repo_factory = job_repo_factory
        self._job_id = job_id

    def run(self):
        asyncio.run(self.run_async())

    async def run_async(self):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """

        job_repo = self._job_repo_factory.create()
        job_id = self._job_id

        # First check the job and put it in PROCESSING state.
        async with job_repo.new_session() as session:
            job = await session.get_job(job_id)
            if job.state != JobState.QUEUED:
                raise ApiClientException(f"Job can't be processed because it's in state {job.state.value}.")
            job.set_state(JobState.PROCESSING)
            job.started = datetime.now(tz=UTC)

        # In a new session do the actual processing.
        async with job_repo.new_session() as session:
            job = await session.get_job(job_id)
            logger.info("Starting job %s...", job_id)
            with init_file_logger(job.job_id, job.log_file) as job_logger:
                process_job_task = asyncio.create_task(self._process_job_worker(job, job_logger))
                cancellation_listener_task = asyncio.create_task(self._cancellation_listener(job_id, job_repo))

                try:
                    # Wait until the job has completed normally or it has been canceled.
                    done, _ = await asyncio.wait(
                        [process_job_task, cancellation_listener_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    # This re-raises any exceptions from the completed tasks.
                    for task in done:
                        task.result()

                    job_logger.info("EDP created successfully.")
                    logger.info("Job %s completed.", job.job_id)
                    job.set_state(JobState.COMPLETED)

                except asyncio.CancelledError:
                    job_logger.info("Job was cancelled.")
                    logger.info("Job %s cancelled.", job.job_id)
                    job.set_state(JobState.CANCELLED, "Analysis was cancelled.")

                except Exception as exception:
                    job_logger.info("Job has failed.")
                    logger.error("Job %s has failed.", job.job_id, exc_info=exception)
                    job.set_state(JobState.FAILED, f"Processing failed: {exception}")

                finally:
                    job.finished = datetime.now(tz=UTC)
                    process_job_task.cancel()
                    cancellation_listener_task.cancel()

    @staticmethod
    async def _process_job_worker(job: Job, job_logger: Logger):
        logger.debug("Job data directory: %s", job.job_base_dir)
        with TemporaryDirectory() as temp_working_dir_str:
            temp_working_dir = Path(temp_working_dir_str)
            logger.debug("Temporary working directory: %s", temp_working_dir)
            jsons = list(_iter_json_in_dir(job.input_data_dir))
            if len(jsons) == 0:
                raise RuntimeError("Expected at least one EDP json file, found no json files.")
            for file in jsons:
                filename = file.with_suffix("").name
                target_file_name = f"{filename}_mds.json"
                await convert_edp_to_mds(file, temp_working_dir / target_file_name)
            await ZipAlgorithm().compress(temp_working_dir, job.zip_archive)

    @staticmethod
    async def _cancellation_listener(job_id: UUID, job_repo: JobRepository):
        while True:
            async with job_repo.new_session() as session:
                job = await session.get_job(job_id)
                if job.state == JobState.CANCELLATION_REQUESTED:
                    raise asyncio.CancelledError()
                await asyncio.sleep(1)


def _iter_json_in_dir(dir: Path) -> Iterator[Path]:
    for file in dir.iterdir():
        if file.is_file() and file.match("*json"):
            yield file
