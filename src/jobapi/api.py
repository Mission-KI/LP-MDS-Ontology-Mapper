from enum import Enum
from tempfile import TemporaryFile
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlmodel import create_engine

from edps.types import UserProvidedEdpData
from jobapi.config import AppConfig
from jobapi.manager import AnalysisJobManager
from jobapi.repo.base import JobRepository
from jobapi.repo.inmemory import InMemoryJobRepository
from jobapi.repo.persistent import DbJobRepository
from jobapi.types import JobView


class Tags(str, Enum):
    AnalysisJob = "Analysis job for dataspace"


def get_job_api_router(app_config: AppConfig):
    job_repo = create_job_repository(app_config)
    job_manager = AnalysisJobManager(app_config, job_repo)
    router = APIRouter()

    @router.post("/analysisjob", tags=[Tags.AnalysisJob], summary="Create analysis job")
    async def create_analysis_job(user_data: UserProvidedEdpData) -> JobView:
        """Create an analysis job based on the user provided EDP data.
        This must be followed by uploading the actual data.

        Returns infos about the job including an ID.
        """

        job_id = await job_manager.create_job(user_data)
        return await job_manager.get_job_view(job_id)

    # TODO mka: Is the filename needed for the analysis? Or the type (File.type) derived only by the content? -> Filename is needed for identifying the type!
    # TODO mka: We can use an explicit parameter (filename:str or type:str), the mime-type or use multipart/form-data. Or put the type in the userdata.
    @router.post(
        "/analysisjob/{job_id}/data/{filename}",
        summary="Upload data for new analysis job",
        tags=[Tags.AnalysisJob],
        openapi_extra={
            "requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}
        },
    )
    async def upload_analysis_data(
        job_id: UUID, request: Request, filename: str, background_tasks: BackgroundTasks
    ) -> JobView:
        """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

        Returns infos about the job.
        """

        if request.headers.get("Content-Type") != "application/octet-stream":
            raise HTTPException(status_code=415, detail="Unsupported Media Type. Expected 'application/octet-stream'.")

        with TemporaryFile(mode="w+b") as temp_file:
            # Stream the request into the temp file chunk by chunk.
            async for chunk in request.stream():
                temp_file.write(chunk)
            # Seek back to start (needs 'w+b' mode)!
            temp_file.seek(0)
            await job_manager.store_input_file(job_id, filename, temp_file)

        background_tasks.add_task(job_manager.process_job, job_id)
        return await job_manager.get_job_view(job_id)

    @router.post(
        "/analysisjob/{job_id}/data",
        summary="Upload data for new analysis job as multipart form data",
        tags=[Tags.AnalysisJob],
    )
    async def upload_analysis_data_multipart(
        job_id: UUID, upload_file: UploadFile, background_tasks: BackgroundTasks
    ) -> JobView:
        """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

        Returns infos about the job.
        """

        try:
            await job_manager.store_input_file(job_id, upload_file.filename, upload_file.file)
        finally:
            await upload_file.close()

        background_tasks.add_task(job_manager.process_job, job_id)
        return await job_manager.get_job_view(job_id)

    @router.get("/analysisjob/{job_id}/status", tags=[Tags.AnalysisJob], summary="Get analysis job status")
    async def get_status(job_id: UUID) -> JobView:
        """Returns infos about the job."""

        return await job_manager.get_job_view(job_id)

    @router.get(
        "/analysisjob/{job_id}/result",
        tags=[Tags.AnalysisJob],
        summary="Return zipped EDP after completed analysis",
        response_class=Response,
        responses={
            200: {
                "description": "Successful Response",
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def get_result(job_id: UUID):
        """If an analysis job has reached state COMPLETED, this call returns the zipped EDP including images."""

        zip_archive = await job_manager.get_zipped_result(job_id)
        return FileResponse(zip_archive, media_type="application/zip", filename=zip_archive.name)

    @router.get(
        "/analysisjob/{job_id}/report",
        tags=[Tags.AnalysisJob],
        summary="Return PDF report after completed analysis",
        response_class=Response,
        responses={
            200: {
                "description": "Successful Response",
                "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def get_report(job_id: UUID):
        """If an analysis job has reached state COMPLETED, this call returns the PDF report."""
        raise NotImplementedError()

    return router


def create_job_repository(app_config: AppConfig) -> JobRepository:
    db_url = app_config.db_url
    if db_url is None:
        return InMemoryJobRepository()
    else:
        db_engine = create_engine(str(db_url))
        return DbJobRepository(db_engine)
