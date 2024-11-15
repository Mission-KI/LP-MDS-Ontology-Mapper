from enum import Enum

from pydantic import BaseModel, Field


class JobState(str, Enum):
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobView(BaseModel):
    job_id: str = Field(description="Job ID", frozen=True)
    state: JobState = Field(
        description="""Job state:\n
- WAITING_FOR_DATA: job is waiting for data upload\n
- QUEUED: job is queued for analyzation\n
- PROCESSING: job is processing data\n
- COMPLETED: job has completed successfully\n
- FAILED: job has failed
""",
        default=JobState.WAITING_FOR_DATA,
    )
    state_detail: str | None = Field(description="State details", default=None)
