from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from typing import AsyncIterator, Tuple

from matplotlib.axes import Axes
from pytest import fixture

from edps.context import OutputContext, OutputDaseenContext, OutputLocalFilesContext
from edps.task import SimpleTaskContext
from edps.types import ExtendedDatasetProfile, FileReference

TESTS_ROOT_PATH = Path(__file__).parent.absolute()


@fixture
def path_output():
    path = TESTS_ROOT_PATH.parent / "output"
    if not path.exists():
        path.mkdir()
    yield path


@fixture
def path_data_test_csv():
    return TESTS_ROOT_PATH / "data/test.csv"


@fixture
def path_data_test_headerless_csv():
    return TESTS_ROOT_PATH / "data/test_headerless.csv"


@fixture
def path_data_test_pickle():
    return TESTS_ROOT_PATH / "data/test.pickle"


@fixture
def path_data_test_xls():
    return TESTS_ROOT_PATH / "data/test.xls"


@fixture
def path_data_test_xlsx():
    return TESTS_ROOT_PATH / "data/test.xlsx"


@fixture
def path_data_test_zip():
    return TESTS_ROOT_PATH / "data/test.zip"


@fixture
def path_data_pontusx_algocustomdata():
    return TESTS_ROOT_PATH / "data/pontusx/algoCustomData.json"


@fixture
def path_data_pontusx_ddo():
    return TESTS_ROOT_PATH / "data/pontusx/ddo.json"


@fixture
def output_context(path_output):
    return OutputLocalFilesContext(path_output)


@fixture
def ctx(output_context):
    return SimpleTaskContext(getLogger("TEST"), output_context)


@fixture
def daseen_output_context(monkeypatch, path_output):
    with monkeypatch.context() as monkey:
        monkey.setattr(OutputDaseenContext, "_upload_to_elastic", lambda *args: None)
        monkey.setattr(OutputDaseenContext, "_upload_to_s3", lambda *args: None)
        yield OutputDaseenContext(
            local_path=path_output,
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
