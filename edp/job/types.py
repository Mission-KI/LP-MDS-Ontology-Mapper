from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from edp.types import UserProvidedEdpData


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


class Job(JobView):
    user_data: UserProvidedEdpData = Field(frozen=True)
    job_base_dir: Path = Field(frozen=True)

    @property
    def input_data_dir(self) -> Path:
        return self.job_base_dir / "input"

    @property
    def result_dir(self) -> Path:
        return self.job_base_dir / "result"

    @property
    def zip_archive(self) -> Path:
        return self.job_base_dir / "result.zip"
