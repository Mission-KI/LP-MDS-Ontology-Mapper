from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from typing import AsyncIterator, Tuple

from matplotlib.axes import Axes
from pytest import fixture

from edp.context import OutputContext, OutputDaseenContext, OutputLocalFilesContext
from edp.task import SimpleTaskContext
from edp.types import ExtendedDatasetProfile, FileReference

DIR = Path(__file__).parent.absolute()

_OUTPUT_PATH = DIR.parent / "output"


@fixture
def output_directory():
    if not _OUTPUT_PATH.exists():
        _OUTPUT_PATH.mkdir()
    yield _OUTPUT_PATH


@fixture
def output_context(output_directory):
    return OutputLocalFilesContext(output_directory)


@fixture
def ctx(output_context):
    return SimpleTaskContext(getLogger("TEST"), output_context)


@fixture
def daseen_output_context(monkeypatch, output_directory):
    with monkeypatch.context() as monkey:
        monkey.setattr(OutputDaseenContext, "_upload_to_elastic", lambda *args: None)
        monkey.setattr(OutputDaseenContext, "_upload_to_s3", lambda *args: None)
        yield OutputDaseenContext(
            local_path=output_directory,
            s3_access_key_id="ABC",
            s3_secret_access_key="PW",
            s3_bucket_name="dummybucket",
            elastic_url="http://elastic",
            elastic_apikey="APIKEY",
        )


class DummyOutputContext(OutputContext):
    async def write_edp(self, name: str, edp: ExtendedDatasetProfile) -> FileReference:
        raise NotImplementedError()

    @asynccontextmanager
    def get_plot(self, name: str) -> AsyncIterator[Tuple[Axes, FileReference]]:
        raise NotImplementedError()
