from enum import Enum
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from edp.types import UserProvidedEdpData


class Tags(Enum):
    analysisjob: str = "Analysis job for dataspace"


class JobState(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobInfo(BaseModel):
    job_id: str = Field(description="Job ID")
    state: JobState = Field(
        description="""Job state:\n
- NOT_FOUND: job with given ID doesn't exist\n
- WAITING_FOR_DATA: job is waiting for data upload\n
- QUEUED: job is queued for analyzation\n
- PROCESSING: job is processing data\n
- COMPLETED: job has completed successfully\n
- FAILED: job has failed
"""
    )
    detail_message: str | None = Field(description="Detailed message", default=None)


router = APIRouter()


@router.post("/analysisjob", tags=[Tags.analysisjob], summary="Create analysis job")
async def create_analysis_job(userdata: UserProvidedEdpData) -> JobInfo:
    """Create an analysis job based on the user provided EDP data.
    This must be followed by uploading the actual data.

    Returns infos about the job including an ID.
    """
    return JobInfo(job_id=str(uuid4()), state=JobState.WAITING_FOR_DATA, detail_message="Waiting for data upload..")


@router.post(
    "/analysisjob/{job_id}/data",
    summary="Upload data for new analysis job",
    tags=[Tags.analysisjob],
    openapi_extra={
        "requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}
    },
)
async def upload_analysis_data(job_id: str, request: Request) -> JobInfo:
    """Upload a file to be analyzed for a previously created job. This starts (or enqueues) the analysis job.

    Returns infos about the job.
    """
    if request.headers.get("Content-Type") != "application/octet-stream":
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Expected 'application/octet-stream'.")

    file_size = 0
    file_name = f"output/{job_id}.bin"
    with open(file_name, mode="wb") as file:
        async for chunk in request.stream():
            file_size += len(chunk)
            file.write(chunk)
            print(f"Received chunk of size: {len(chunk)} bytes")
    print(f"Writing file {file_name} completed (size: {file_size}).")

    return JobInfo(job_id=job_id, state=JobState.QUEUED, detail_message="Queued")


@router.get("/analysisjob/{job_id}/status", tags=[Tags.analysisjob], summary="Get analysis job status")
async def get_status(job_id: str) -> JobInfo:
    """Returns infos about the job."""
    return JobInfo(job_id=job_id, state=JobState.PROCESSING, detail_message="Processing")


@router.get(
    "/analysisjob/{job_id}/result",
    tags=[Tags.analysisjob],
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
    # raise NotImplementedError()
    file_name = f"output/{job_id}.zip"
    return FileResponse(file_name, media_type="application/zip", filename="downloaded_file.zip")


@router.get(
    "/analysisjob/{job_id}/report",
    tags=[Tags.analysisjob],
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
