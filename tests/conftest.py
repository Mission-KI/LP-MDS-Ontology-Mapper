from pathlib import Path

from pytest import fixture

from edp.context import OutputDaseenContext, OutputLocalFilesContext

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
def daseen_output_context(output_directory):
    return OutputDaseenContext(
        local_path=output_directory,
        s3_access_key_id="ABC",
        s3_secret_access_key="PW",
        s3_bucket_name="dummybucket",
        elastic_url="http://elastic",
        elastic_apikey="APIKEY",
        skip_upload=True,
    )
