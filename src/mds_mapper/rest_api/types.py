from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from mds_mapper.types import Config as Config


class JobState(str, Enum):
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    CANCELLED = "CANCELLED"


END_STATES = set([JobState.FAILED, JobState.CANCELLED, JobState.COMPLETED])


class JobView(BaseModel):
    job_id: UUID = Field(description="Job ID", frozen=True)
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
