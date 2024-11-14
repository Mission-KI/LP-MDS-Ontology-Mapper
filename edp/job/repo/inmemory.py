from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from edp.job.repo import Job, JobRepository, JobSession
from edp.job.types import JobState
from edp.types import UserProvidedEdpData


class InMemoryJob(Job):
    def __init__(self, job_id: str, user_data: UserProvidedEdpData, job_base_dir: Path):
        self._job_id = job_id
        self._user_data = user_data
        self._job_base_dir = job_base_dir
        self._state = JobState.WAITING_FOR_DATA
        self._state_detail: Optional[str] = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def state_detail(self) -> Optional[str]:
        return self._state_detail

    def update_state(self, state: JobState, detail: Optional[str] = None) -> None:
        self._state = state
        self._state_detail = detail

    @property
    def user_data(self) -> UserProvidedEdpData:
        return self._user_data

    @property
    def job_base_dir(self) -> Path:
        return self._job_base_dir


class InMemoryJobSession(JobSession):
    def __init__(self, jobs: dict[str, InMemoryJob]):
        self._jobs = jobs

    async def create_job(self, job_id: str, user_data: UserProvidedEdpData, job_base_dir: Path):
        job = InMemoryJob(
            job_id=job_id,
            user_data=user_data,
            job_base_dir=job_base_dir,
        )
        self._jobs[job_id] = job
        return job

    async def get_job(self, job_id: str):
        if job_id not in self._jobs:
            raise RuntimeError(f"Job {job_id} doesn't exist.")
        return self._jobs[job_id]


class InMemoryJobRepository(JobRepository):
    def __init__(self):
        self._jobs = dict[str, InMemoryJob]()

    @asynccontextmanager
    async def new_session(self) -> AsyncIterator[JobSession]:
        yield InMemoryJobSession(self._jobs)
