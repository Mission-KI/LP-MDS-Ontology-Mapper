from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import UUID

from jobapi.exception import ApiClientException
from jobapi.repo import Job, JobRepository, JobSession
from jobapi.repo.base import JobRepositoryFactory
from jobapi.types import JobData, JobState


class InMemoryJob(Job):
    def __init__(self, job_id: UUID, job_data: JobData, job_base_dir: Path):
        self._job_id = job_id
        self._job_data = job_data
        self._job_base_dir = job_base_dir
        self._state = JobState.WAITING_FOR_DATA
        self._state_detail: Optional[str] = None

        main_ref = job_data.user_provided_edp_data.assetRefs[0]
        self._asset_id = main_ref.assetId
        self._asset_version = main_ref.assetVersion
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
    def job_data(self) -> JobData:
        return self._job_data

    @property
    def job_base_dir(self) -> Path:
        return self._job_base_dir


class InMemoryJobSession(JobSession):
    def __init__(self, jobs: dict[UUID, InMemoryJob]):
        self._jobs = jobs

    async def create_job(self, job_id: UUID, job_data: JobData, job_base_dir: Path):
        job = InMemoryJob(
            job_id=job_id,
            job_data=job_data,
            job_base_dir=job_base_dir,
        )
        self._jobs[job_id] = job
        return job

    async def get_job(self, job_id: UUID):
        if job_id not in self._jobs:
            raise ApiClientException(f"Job '{job_id}' doesn't exist.")
        return self._jobs[job_id]


class InMemoryJobRepository(JobRepository):
    def __init__(self, job_dictionary: dict[UUID, InMemoryJob]):
        self._jobs = job_dictionary

    @asynccontextmanager
    async def new_session(self) -> AsyncIterator[JobSession]:
        yield InMemoryJobSession(self._jobs)


class InMemoryJobRepositoryFactory(JobRepositoryFactory):
    def __init__(self):
        self._jobs = dict[UUID, InMemoryJob]()

    def create(self) -> InMemoryJobRepository:
        return InMemoryJobRepository(self._jobs)
