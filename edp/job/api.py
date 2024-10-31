from enum import Enum

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from edp.config import AppConfig
from edp.job.manager import AnalysisJobManager, InMemoryJobManager
from edp.types import JobState, JobView, UserProvidedEdpData


class Tags(str, Enum):
    AnalysisJob = "Analysis job for dataspace"


def get_job_api_router(app_config: AppConfig):
    job_manager: AnalysisJobManager = InMemoryJobManager(app_config)
    router = APIRouter()

    @router.post("/analysisjob", tags=[Tags.AnalysisJob], summary="Create analysis job")
    async def create_analysis_job(userdata: UserProvidedEdpData) -> JobView:
        """Create an analysis job based on the user provided EDP data.
        This must be followed by uploading the actual data.

        Returns infos about the job including an ID.
        """

        return await job_manager.create_job(userdata)

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
    async def upload_analysis_data(job_id: str, request: Request, filename: str) -> JobView:
        """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

        Returns infos about the job.
        """

        if request.headers.get("Content-Type") != "application/octet-stream":
            raise HTTPException(status_code=415, detail="Unsupported Media Type. Expected 'application/octet-stream'.")

        job = await job_manager.get_job(job_id)
        await job_manager.upload_file(job, filename, request)
        return job

    @router.post(
        "/analysisjob/{job_id}/data",
        summary="Upload data for new analysis job as multipart form data",
        tags=[Tags.AnalysisJob],
    )
    async def upload_analysis_data_multipart(job_id: str, upload_file: UploadFile) -> JobView:
        """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

        Returns infos about the job.
        """

        job = await job_manager.get_job(job_id)
        await job_manager.upload_file_multipart(job, upload_file)
        return job

    @router.get("/analysisjob/{job_id}/status", tags=[Tags.AnalysisJob], summary="Get analysis job status")
    async def get_status(job_id: str) -> JobView:
        """Returns infos about the job."""

        return await job_manager.get_job(job_id)

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
    async def get_result(job_id: str):
        """If an analysis job has reached state COMPLETED, this call returns the zipped EDP including images."""

        job = await job_manager.get_job(job_id)
        if job.state != JobState.COMPLETED:
            raise RuntimeError(f"There is no result for job {job.job_id}")
        return FileResponse(job.zip_path, media_type="application/zip", filename=job.zip_path.name)

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
    async def get_report(job_id: str):
        """If an analysis job has reached state COMPLETED, this call returns the PDF report."""
        raise NotImplementedError()

    return router
