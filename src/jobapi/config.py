from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ENV file should be in the same directory (only relevant for development; for production we use ENV variables).
ENV_FILE = Path(__file__).parent / ".env"


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
    workers: int = Field(
        default=8,
        description="""
            Number of workers for job processing in Process/Thread-Pool.
            Depending on 'db_url' and 'workers' either use no pool or ProcessPoolExecutor or ThreadPoolExecutor:
            - workers==0: no pool
            - workers and !db_url: use ThreadPoolExecutor
            - workers and db_url: use ProcessPoolExecutor
        """,
    )

    model_config = SettingsConfigDict(env_file=ENV_FILE)


def get_app_config():
    # Prevent MyPy check to allow required settings without default.
    config = AppConfig()  # type: ignore
    # We can't log the DB credentials, so use model_dump() to respect the exclude attribute.
    getLogger(__name__).info("App config: %s", config.model_dump())
    return config
