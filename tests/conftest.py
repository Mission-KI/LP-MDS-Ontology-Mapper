from pathlib import Path

from pytest import fixture

from edp.context import OutputLocalFilesContext

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
