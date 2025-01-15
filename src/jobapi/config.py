from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# App config is read from .env file and validate on app startup.
# If you have secret settings that shouldn't be logged set exclude=True.
# Settings may be required without a default value. If they are not specified this leads to an error on startup.
class AppConfig(BaseSettings):
    working_dir: Path = Field(description="Working directory used for job data", default=Path.cwd())
    host: str = Field(description="Bind FastAPI to host", default="127.0.0.1")
    db_url: Optional[AnyUrl] = Field(
        description="DB connection URL for persisting jobs; None uses in-memory repository",
        exclude=True,
        examples=["postgresql://user:pass@localhost:5432/edps"],
        default=None,
    )

    model_config = SettingsConfigDict(env_file="edp/.env")


def get_app_config():
    # Prevent MyPy check to allow required settings without default.
    return AppConfig()  # type: ignore
