import json
import logging
from argparse import ArgumentParser
from importlib.metadata import version as get_version
from logging import getLogger
from pathlib import Path

from fastapi import FastAPI
from uvicorn import run as server_run

from edp.config import AppConfig, get_app_config
from edp.job import get_job_api_router


def config_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


def main():
    """We start and configure the server programmatically as an alternative to the FastAPI CLI."""

    config_logging()
    app_config = get_app_config()
    app = init_fastapi(app_config)

    logger = getLogger(__name__)
    logger.info("Starting server..")
    # To bind to all network interfaces add host="0.0.0.0".
    server_run(app, port=8000, host=app_config.host)
    logger.info("Shutting down.")


def init_fastapi(app_config: AppConfig):
    job_api_router = get_job_api_router(app_config)
    app = FastAPI(title="EDPS Rest API", version=get_version("edp"))
    app.include_router(job_api_router, prefix="/v1/dataspace")
    return app


def cmdline_export_openapi_schema():
    def _get_args():
        parser = ArgumentParser()
        parser.add_argument(
            "-o", "--output", type=Path, required=True, help="PurePosixPath to output the OpenAPI schema to"
        )
        return parser.parse_args()

    args = _get_args()
    app_config = get_app_config()
    app = init_fastapi(app_config)
    schema = app.openapi()

    with open(args.output, "wt") as file:
        json.dump(schema, file, indent=4)


if __name__ == "__main__":
    main()
