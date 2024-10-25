from abc import ABC, abstractmethod
from logging import getLogger
from shutil import copyfileobj
from uuid import uuid4

from fastapi import Request, UploadFile

from edp.config import app_config
from edp.context import OutputLocalFilesContext
from edp.service import Service
from edp.types import Job, JobState, UserProvidedEdpData
from edp.zip import zip_directory


class AnalysisJobManager(ABC):
    @abstractmethod
    async def create_job(self, userdata: UserProvidedEdpData) -> Job:
        """Create a job based on the user provided EDP data.
        The job gets an ID and a job working directory and is initially in state 'WAITING_FOR_DATA'.
        """
        ...

    @abstractmethod
    async def get_job(self, job_id: str) -> Job:
        """Get a job by ID."""
        ...

    @abstractmethod
    async def upload_file(self, job: Job, filename: str, request: Request):
        """Upload job data which will be analyzed later.
        The raw data is extracted from the 'request' and saved to a file named 'filename' in the job working dir.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. This needs to be repeated if an error occurs.
        After successul upload the state changes to 'QUEUED'.
        """
        ...

    @abstractmethod
    async def upload_file_multipart(self, job: Job, upload_file: UploadFile):
        """Upload job data which will be analyzed later.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. This needs to be repeated if an error occurs.
        After successul upload the state changes to 'QUEUED'.
        """
        ...

    @abstractmethod
    async def process_jobs(self):
        """Look for jobs in state 'QUEUED' and call #process_job() for each."""
        ...

    @abstractmethod
    async def process_job(self, job: Job):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """
        ...


class InMemoryJobManager(AnalysisJobManager):
    def __init__(self):
        self._jobs = dict[str, Job]()
        self._logger = getLogger(__name__)

    async def create_job(self, userdata: UserProvidedEdpData) -> Job:
        job_id = str(uuid4())

        # Create job dir
        job_dir = app_config.working_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            state=JobState.WAITING_FOR_DATA,
            user_data=userdata,
            input_data_dir=job_dir / "input",
            result_dir=job_dir / "result",
            zip_path=job_dir / "result.zip",
        )
        self._jobs[job_id] = job

        self._logger.info(f"Job created: {job.job_id}")
        return job

    async def get_job(self, job_id: str):
        if job_id not in self._jobs:
            raise RuntimeError(f"Job {job_id} doesn't exist.")
        return self._jobs[job_id]

    async def upload_file(self, job: Job, filename: str, request: Request):
        if job.state != JobState.WAITING_FOR_DATA:
            raise RuntimeError(f"Job doesn't accept any file uploads because it's in state {job.state}.")

        self._logger.info(f"File upload for job {job.job_id} started.")
        job.input_data_dir.mkdir(parents=True, exist_ok=True)
        data_path = job.input_data_dir / filename
        # Security: filename is not allowed to escape input_data_dir!
        if job.input_data_dir not in data_path.parents:
            raise RuntimeError(f"Illegal filename: {filename}")

        try:
            with open(data_path, mode="wb") as writer:
                # Stream the request into the file chunk by chunk.
                async for chunk in request.stream():
                    writer.write(chunk)
            if data_path.stat().st_size == 0:
                raise RuntimeError("Upload was empty!")
        except:
            # If there is an error delete the file.
            data_path.unlink()
            raise

        job.state = JobState.QUEUED

        self._logger.info(
            f"File upload for job {job.job_id} is complete: {data_path} ({data_path.stat().st_size} bytes)"
        )

    async def upload_file_multipart(self, job: Job, upload_file: UploadFile):
        if job.state != JobState.WAITING_FOR_DATA:
            raise RuntimeError(f"Job doesn't accept any file uploads because it's in state {job.state}.")
        if not upload_file.filename:
            raise RuntimeError("Filename is missing")

        job.input_data_dir.mkdir(parents=True, exist_ok=True)
        data_file_path = job.input_data_dir / upload_file.filename

        try:
            with data_file_path.open("wb") as dest:
                copyfileobj(upload_file.file, dest)
        except:
            # If there is an error delete the file.
            data_file_path.unlink()
            raise
        finally:
            await upload_file.close()

        job.state = JobState.QUEUED

        self._logger.info(
            f"File upload for job {job.job_id} is complete: {data_file_path} ({data_file_path.stat().st_size} bytes)"
        )

    async def process_jobs(self):
        self._logger.debug(f"Checking for queued jobs")
        for job in self._jobs.values():
            try:
                if job.state == JobState.QUEUED:
                    await self.process_job(job)
            except Exception as e:
                self._logger.error(f"Job {job.job_id} has failed: {e}")

    async def process_job(self, job: Job):
        if job.state != JobState.QUEUED:
            raise RuntimeError(f"Job can't be processed because it's in state {job.state}.")

        self._logger.info(f"Starting job {job.job_id}...")
        job.state = JobState.PROCESSING

        try:
            service = Service()
            output_context = OutputLocalFilesContext(job.result_dir)
            await service.analyse_asset(job.input_data_dir, job.user_data, output_context)
            zip_directory(job.result_dir, job.zip_path)
            job.state = JobState.COMPLETED
            self._logger.info(f"Job {job.job_id} completed.")

        except Exception as e:
            job.state = JobState.FAILED
            job.state_detail = f"Processing failed: {e}"
            raise


job_manager: AnalysisJobManager = InMemoryJobManager()
