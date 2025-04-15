from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Optional
from uuid import UUID

import sqlmodel
from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, select

from ..exception import ApiClientException
from ..types import Config, JobState
from .base import Job, JobRepository, JobRepositoryFactory, JobSession


class DbJob(SQLModel, Job, table=True):
    __tablename__ = "job"

    id: UUID = Field(primary_key=True)
    f_job_base_dir: str = Field(sa_column_kwargs={"name": "base_dir"})
    f_state: JobState = Field(sa_column_kwargs={"name": "state"})
    f_state_detail: Optional[str] = Field(sa_column_kwargs={"name": "state_detail"})
    f_started: Optional[datetime] = Field(sa_column_kwargs={"name": "started"})
    f_finished: Optional[datetime] = Field(sa_column_kwargs={"name": "finished"})
    # tasks: list["DbTask"] = Relationship(back_populates="job")

    def __init__(self, job_id: UUID, config: Config, job_base_dir: Path):
        self.id = job_id
        self.f_job_base_dir = str(PurePosixPath(job_base_dir))
        self.f_state = JobState.WAITING_FOR_DATA
        self.f_state_detail: Optional[str] = None

        self.config_data_path.write_text(config.model_dump_json(by_alias=True))

    @property
    def job_id(self) -> UUID:
        return self.id

    @property
    def state(self) -> JobState:
        return self.f_state

    @property
    def state_detail(self) -> Optional[str]:
        return self.f_state_detail

    def _set_state_impl(self, state: JobState, detail: Optional[str] = None) -> None:
        self.f_state = state
        self.f_state_detail = detail

    @property
    def started(self) -> Optional[datetime]:
        return self.f_started

    @started.setter
    def started(self, started: Optional[datetime]):
        self.f_started = started

    @property
    def finished(self) -> Optional[datetime]:
        return self.f_finished

    @finished.setter
    def finished(self, finished: Optional[datetime]):
        self.f_finished = finished

    @property
    def configuration(self) -> Config:
        return Config.model_validate_json(self.config_data_path.read_text())

    @property
    def job_base_dir(self) -> Path:
        return Path(self.f_job_base_dir)

    @property
    def config_data_path(self) -> Path:
        return self.job_base_dir / "job_data.json"


# class DbTask(SQLModel, table=True):
#     __tablename__ = "task"

#     id: UUID = Field(primary_key=True)
#     name: str
#     started: Optional[datetime]
#     finished: Optional[datetime]
#     job_id: UUID = Field(foreign_key="job.id")
#     job: DbJob = Relationship(back_populates="tasks")


class DbJobSession(JobSession):
    def __init__(self, session: Session):
        self._session = session

    async def create_job(self, job_id: UUID, config: Config, job_base_dir: Path):
        job = DbJob(
            job_id=job_id,
            config=config,
            job_base_dir=job_base_dir,
        )
        self._session.add(job)
        return job

    async def get_job(self, job_id: UUID):
        statement = select(DbJob).where(DbJob.id == job_id)
        results = self._session.exec(statement)
        job = results.first()
        if job is None:
            raise ApiClientException(f"Job '{job_id}' doesn't exist.")
        return job


class DbJobRepository(JobRepository):
    def __init__(self, engine: Engine):
        self._engine = engine
        SQLModel.metadata.create_all(engine)

    @asynccontextmanager
    async def new_session(self) -> AsyncIterator[JobSession]:
        with Session(self._engine) as session:
            yield DbJobSession(session)
            session.commit()


class DbJobRepositoryFactory(JobRepositoryFactory):
    def __init__(self, db_url: str):
        self._db_url = db_url

    def create(self) -> DbJobRepository:
        db_engine = sqlmodel.create_engine(str(self._db_url))
        return DbJobRepository(db_engine)
