from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from edp.job.types import JobState, JobView
from edp.types import UserProvidedEdpData


class Job(ABC):
    """Internal Job class. There are different implementations depending on persistence."""

    @property
    @abstractmethod
    def job_id(self) -> str: ...

    @property
    @abstractmethod
    def state(self) -> JobState: ...

    @property
    @abstractmethod
    def state_detail(self) -> Optional[str]: ...

    @abstractmethod
    def update_state(self, state: JobState, detail: Optional[str] = None) -> None: ...

    @property
    @abstractmethod
    def user_data(self) -> UserProvidedEdpData: ...

    @property
    @abstractmethod
    def job_base_dir(self) -> Path: ...

    def to_job_view(self) -> JobView:
        return JobView(job_id=self.job_id, state=self.state, state_detail=self.state_detail)

    @property
    def input_data_dir(self) -> Path:
        return self.job_base_dir / "input"

    @property
    def result_dir(self) -> Path:
        return self.job_base_dir / "result"

    @property
    def zip_archive(self) -> Path:
        return self.job_base_dir / "result.zip"


class JobSession(ABC):
    """Abstract JobSession for creating and finding jobs. It is instantiated by the JobRepository."""

    @abstractmethod
    async def create_job(self, job_id: str, user_data: UserProvidedEdpData, job_base_dir: Path) -> Job:
        """Create a new job."""
        ...

    @abstractmethod
    async def get_job(self, job_id: str) -> Job:
        """Find a job by ID."""
        ...


class JobRepository(ABC):
    """Abstract JobRepository for creating JobSessions."""

    @abstractmethod
    @asynccontextmanager
    def new_session(self) -> AsyncIterator[JobSession]:
        """Start a new session in a context. When the context is closed, it gets automatically committed."""
        ...
