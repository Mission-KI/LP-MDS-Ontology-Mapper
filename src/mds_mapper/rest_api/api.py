from enum import Enum
from tempfile import TemporaryFile
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, PlainTextResponse

from .config import AppConfig
from .manager import AnalysisJobManager
from .types import Config, JobView


class Tags(str, Enum):
    AnalysisJob = "Analysis job for dataspace"


def get_job_api_router(app_config: AppConfig):
    job_manager = AnalysisJobManager(app_config)
    router = APIRouter()

    @router.post("/analysisjob", tags=[Tags.AnalysisJob], summary="Create analysis job")
    async def create_analysis_job(config: Config) -> JobView:
        """Create an analysis job based on the user provided EDP data.
        This must be followed by uploading the actual data.

        Returns infos about the job including an ID.
        """

        job_id = await job_manager.create_job(config)
        return await job_manager.get_job_view(job_id)

    @router.post(
        "/analysisjob/{job_id}/cancel",
        tags=[Tags.AnalysisJob],
        summary="Cancel analysis job",
        response_class=PlainTextResponse,
        responses={
            204: {
                "description": "Job cancellation request accepted",
            },
        },
    )
    async def cancel_analysis_job(job_id: UUID) -> PlainTextResponse:
        """Cancel an analysis job based on the job ID."""

        await job_manager.cancel_job(job_id)
        return PlainTextResponse(status_code=204)

    @router.post(
        "/analysisjob/{job_id}/data/{filename}",
        summary="Upload data for new analysis job",
        tags=[Tags.AnalysisJob],
        openapi_extra={"requestBody": {"content": {"*/*": {"schema": {"type": "string", "format": "binary"}}}}},
    )
    async def upload_analysis_data(
        job_id: UUID, request: Request, filename: str, background_tasks: BackgroundTasks
    ) -> JobView:
        """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

        Returns infos about the job.
        """

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
                "content": {"application/zip": {}},
            },
        },
    )
    async def get_result(job_id: UUID):
        """If an analysis job has reached state COMPLETED, this call returns the zipped EDP including images."""

        zip_archive = await job_manager.get_zipped_result(job_id)
        return FileResponse(zip_archive, media_type="application/zip", filename=zip_archive.name)

    @router.get(
        "/analysisjob/{job_id}/log",
        tags=[Tags.AnalysisJob],
        summary="Return job log",
        response_class=PlainTextResponse,
        responses={
            204: {
                "description": "No log data",
            },
        },
    )
    async def get_log(job_id: UUID):
        """This call returns the job log up to now."""

        log_file = await job_manager.get_log_file(job_id)
        if not log_file.exists():
            return PlainTextResponse(status_code=204)
        return FileResponse(log_file, media_type="text/plain")

    return router
