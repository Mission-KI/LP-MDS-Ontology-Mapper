from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import UUID

from ..exception import ApiClientException
from ..repo import Job, JobRepository, JobSession
from ..repo.base import JobRepositoryFactory
from ..types import Config, JobState


class InMemoryJob(Job):
    def __init__(self, job_id: UUID, configuration: Config, job_base_dir: Path):
        self._job_id = job_id
        self._configuration = configuration
        self._job_base_dir = job_base_dir
        self._state = JobState.WAITING_FOR_DATA
        self._state_detail: Optional[str] = None

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

    def _set_state_impl(self, state: JobState, detail: Optional[str] = None) -> None:
        self._state = state
        self._state_detail = detail

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
    def configuration(self) -> Config:
        return self._configuration

    @property
    def job_base_dir(self) -> Path:
        return self._job_base_dir


class InMemoryJobSession(JobSession):
    def __init__(self, jobs: dict[UUID, InMemoryJob]):
        self._jobs = jobs

    async def create_job(self, job_id: UUID, config: Config, job_base_dir: Path):
        job = InMemoryJob(
            job_id=job_id,
            configuration=config,
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
