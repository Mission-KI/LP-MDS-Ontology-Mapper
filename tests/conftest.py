from logging import getLogger
from pathlib import Path

from pytest import fixture

from edps.context import OutputDaseenContext, OutputLocalFilesContext
from edps.task import SimpleTaskContext

TESTS_ROOT_PATH = Path(__file__).parent.absolute()


@fixture
def path_work(tmp_path):
    """This is the path to the working directory. Change this to the following code, to review the results in a directory:

    Example:
        path = TESTS_ROOT_PATH / "work"
        if not path.exists():
            path.mkdir()
        yield path
    """
    return tmp_path


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
def path_data_german_decimal_comma_csv():
    return TESTS_ROOT_PATH / "data/german_decimal_comma.csv"


@fixture
def path_data_hamburg_csv():
    return TESTS_ROOT_PATH / "data/hamburg.csv"


@fixture
def path_data_test_xlsx():
    return TESTS_ROOT_PATH / "data/test.xlsx"


@fixture
def path_data_test_zip():
    return TESTS_ROOT_PATH / "data/test.zip"


@fixture
def path_data_test_multiassets_zip():
    return TESTS_ROOT_PATH / "data/test_multiassets.zip"


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
def ctx(path_work, output_context):
    return SimpleTaskContext(getLogger("TEST"), path_work, output_context)


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
