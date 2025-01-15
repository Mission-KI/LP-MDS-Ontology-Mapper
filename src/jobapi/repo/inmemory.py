from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import UUID

from edps.types import UserProvidedEdpData
from jobapi.repo import Job, JobRepository, JobSession
from jobapi.types import JobState


class InMemoryJob(Job):
    def __init__(self, job_id: UUID, user_data: UserProvidedEdpData, job_base_dir: Path):
        self._job_id = job_id
        self._user_data = user_data
        self._job_base_dir = job_base_dir
        self._state = JobState.WAITING_FOR_DATA
        self._state_detail: Optional[str] = None
        self._asset_id = user_data.assetId
        self._asset_version = user_data.version
        self._started: Optional[datetime]
        self._finished: Optional[datetime]

    @property
    def job_id(self) -> UUID:
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
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def asset_version(self) -> Optional[str]:
        return self._asset_version

    @property
    def started(self) -> Optional[datetime]:
        return self._started

    @started.setter
    def started(self, started: Optional[datetime]):
        self._started = started

    @property
    def finished(self) -> Optional[datetime]:
        return self._finished

    @finished.setter
    def finished(self, finished: Optional[datetime]):
        self._finished = finished

    @property
    def user_data(self) -> UserProvidedEdpData:
        return self._user_data

    @property
    def job_base_dir(self) -> Path:
        return self._job_base_dir


class InMemoryJobSession(JobSession):
    def __init__(self, jobs: dict[UUID, InMemoryJob]):
        self._jobs = jobs

    async def create_job(self, job_id: UUID, user_data: UserProvidedEdpData, job_base_dir: Path):
        job = InMemoryJob(
            job_id=job_id,
            user_data=user_data,
            job_base_dir=job_base_dir,
        )
        self._jobs[job_id] = job
        return job

    async def get_job(self, job_id: UUID):
        if job_id not in self._jobs:
            raise RuntimeError(f"Job {job_id} doesn't exist.")
        return self._jobs[job_id]


class InMemoryJobRepository(JobRepository):
    def __init__(self):
        self._jobs = dict[UUID, InMemoryJob]()

    @asynccontextmanager
    async def new_session(self) -> AsyncIterator[JobSession]:
        yield InMemoryJobSession(self._jobs)
