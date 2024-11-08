from abc import ABC, abstractmethod

from edp.job.types import Job


class JobRepository(ABC):
    @abstractmethod
    async def create(self, job: Job) -> None:
        """Save the new job."""
        ...

    async def update(self, job: Job) -> None:
        """Save the updated job."""
        ...

    @abstractmethod
    async def get(self, job_id: str) -> Job:
        """Find a job by ID."""
        ...


class InMemoryJobRepository(JobRepository):
    def __init__(self):
        self._jobs = dict[str, Job]()

    async def create(self, job: Job):
        if job.job_id in self._jobs:
            raise RuntimeError(f"Job {job.job_id} already exists.")
        self._jobs[job.job_id] = job

    async def update(self, job: Job):
        if job.job_id not in self._jobs:
            raise RuntimeError(f"Job {job.job_id} doesn't exist.")
        self._jobs[job.job_id] = job

    async def get(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            raise RuntimeError(f"Job {job_id} doesn't exist.")
        return self._jobs[job_id]
