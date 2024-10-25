from asyncio import create_task, sleep
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI
from uvicorn import run as server_run

from edp.config import app_config
from edp.job.api import router as job_api_router
from edp.job.manager import job_manager

_logger = getLogger(__name__)


async def process_jobs():
    """Loop to process jobs."""
    while True:
        await job_manager.process_jobs()
        # Sleep for 10 secs
        await sleep(10)


@asynccontextmanager
async def process_jobs_in_background(app: FastAPI):
    _logger.info("Starting background task.")
    task = create_task(process_jobs())
    try:
        yield
    finally:
        _logger.info("Stopping background task.")
        task.cancel()


# We want to load the config right away!
_logger.info(f"App configuration:\n{app_config.model_dump()}")

# Config FastAPI.
app = FastAPI(lifespan=process_jobs_in_background)
app.include_router(job_api_router, prefix="/v1/dataspace")


def main():
    """As an alternative to the FastAPI CLI we can start and configure the server programmatically."""

    _logger.info("Starting server..")
    # To bind to all network interfaces add host="0.0.0.0".
    server_run(app, port=8000, host="0.0.0.0")
    _logger.info("Shutting down.")


if __name__ == "__main__":
    main()
