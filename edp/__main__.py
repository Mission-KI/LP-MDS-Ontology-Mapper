import logging
from logging import getLogger

from fastapi import FastAPI
from uvicorn import run as server_run

from edp.config import get_app_config
from edp.job import get_job_api_router


def config_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


def main():
    """We start and configure the server programmatically as an alternative to the FastAPI CLI."""

    config_logging()

    app_config = get_app_config()
    job_api_router = get_job_api_router(app_config)

    app = FastAPI()
    app.include_router(job_api_router, prefix="/v1/dataspace")

    logger = getLogger(__name__)
    logger.info("Starting server..")
    # To bind to all network interfaces add host="0.0.0.0".
    server_run(app, port=8000, host="0.0.0.0")
    logger.info("Shutting down.")


if __name__ == "__main__":
    main()
