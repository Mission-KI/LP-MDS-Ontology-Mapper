from logging import getLogger
from shutil import copyfileobj
from uuid import uuid4

from fastapi import Request, UploadFile

from edp.compression.zip import ZipAlgorithm
from edp.config import AppConfig
from edp.context import OutputLocalFilesContext
from edp.job.repo import JobRepository
from edp.job.types import Job, JobState, UserProvidedEdpData
from edp.service import Service
from edp.types import Config


class AnalysisJobManager:
    def __init__(self, app_config: AppConfig, job_repo: JobRepository):
        self._app_config = app_config
        self._job_repo = job_repo
        self._logger = getLogger(__name__)
        self._service = Service()

    async def create_job(self, userdata: UserProvidedEdpData) -> Job:
        """Create a job based on the user provided EDP data.
        The job gets an ID and a job working directory and is initially in state 'WAITING_FOR_DATA'.
        """

        job_id = str(uuid4())

        # Create job dir
        job_base_dir = self._app_config.working_dir / job_id
        job_base_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            user_data=userdata,
            job_base_dir=job_base_dir,
        )
        await self._job_repo.create(job)

        self._logger.info("Job created: %s", job.job_id)
        return job

    async def get_job(self, job_id: str) -> Job:
        """Get a job by ID."""

        return await self._job_repo.get(job_id)

    async def process_job(self, job: Job):
        """If the job is in state 'QUEUED' process the job.
        During processing it changes to state 'PROCESSING'. When finished it changes to 'COMPLETED' or 'FAILED'.
        Processing involves analyzing the asset and zipping the result.
        """

        if job.state != JobState.QUEUED:
            raise RuntimeError(f"Job can't be processed because it's in state {job.state}.")

        self._logger.info("Starting job %s...", job.job_id)
        job.state = JobState.PROCESSING
        await self._job_repo.update(job)

        try:
            output_context = OutputLocalFilesContext(job.result_dir)
            await self._service.analyse_asset(
                job.input_data_dir, Config(userProvidedEdpData=job.user_data), output_context
            )
            await ZipAlgorithm().compress(job.result_dir, job.zip_archive)
            job.state = JobState.COMPLETED
            await self._job_repo.update(job)
            self._logger.info("Job %s completed.", job.job_id)

        except Exception as exception:
            job.state = JobState.FAILED
            job.state_detail = f"Processing failed: {exception}"
            await self._job_repo.update(job)
            self._logger.error("Job %s has failed: %s", job.job_id, exception)

    async def upload_file(self, job: Job, filename: str, request: Request):
        """Upload job data which will be analyzed later.
        The raw data is extracted from the 'request' and saved to a file named 'filename' in the job working dir.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. This needs to be repeated if an error occurs.
        After successul upload the state changes to 'QUEUED'.
        """

        if job.state != JobState.WAITING_FOR_DATA:
            raise RuntimeError(f"Job doesn't accept any file uploads because it's in state {job.state}.")

        self._logger.info("File upload for job %s started.", job.job_id)
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
        await self._job_repo.update(job)

        self._logger.info(
            "File upload for job %s is complete: %s (%s bytes)", job.job_id, data_path, data_path.stat().st_size
        )

    async def upload_file_multipart(self, job: Job, upload_file: UploadFile):
        """Upload job data which will be analyzed later.
        This must be called exactly once when in state 'WAITING_FOR_DATA'. This needs to be repeated if an error occurs.
        After successul upload the state changes to 'QUEUED'.
        """

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
        await self._job_repo.update(job)

        self._logger.info(
            "File upload for job %s is complete: %s (%s bytes)",
            job.job_id,
            data_file_path,
            data_file_path.stat().st_size,
        )
