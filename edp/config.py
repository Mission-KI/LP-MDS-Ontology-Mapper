from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# App config is read from .env file and validate on app startup.
# If you have secret settings that shouldn't be logged set exclude=True.
# Settings may be required without a default value. If they are not specified this leads to an error on startup.
class AppConfig(BaseSettings):
    working_dir: Path = Field(description="Working directory used for job data", default=Path("work"))
    host: str = Field(description="Bind FastAPI to host", default="127.0.0.1")
    # api_key: str = Field(description="API key", exclude=True)     # example for required secret setting

    model_config = SettingsConfigDict(env_file="edp/.env")


def get_app_config():
    # Prevent MyPy check to allow required settings without default.
    return AppConfig()  # type: ignore
