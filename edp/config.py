from logging import INFO as LOGGING_INFO
from logging import basicConfig as loggingBasicConfig
from logging import getLogger
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Set the global (root) log level
loggingBasicConfig(level=LOGGING_INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


# App config is read from .env file and validate on app startup.
# If you have secret settings that shouldn't be logged set exclude=True.
# Settings may be required without a default value. If they are not specified this leads to an error on startup.
class AppConfig(BaseSettings):
    working_dir: Path = Field(description="Working directory used for job data", default="work")
    # api_key: str = Field(description="API key", exclude=True)     # example for required secret setting

    class Config:
        env_file = ".env"


# Prevent MyPy check to allow required settings without default.
app_config = AppConfig()  # type: ignore

_logger = getLogger(__name__)
_logger.info(f"App configuration:\n{app_config.model_dump()}")
