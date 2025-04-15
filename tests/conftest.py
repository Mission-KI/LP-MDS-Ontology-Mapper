from pathlib import Path

from pytest import fixture

TESTS_ROOT_PATH = Path(__file__).parent


@fixture
def path_work(tmp_path):
    """This is the path to the working directory. Change this to the following code, to review the results in a directory:

    Example:
        path = TESTS_ROOT_PATH / "work"
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        path.mkdir()
        return path
    """
    return tmp_path


@fixture
def path_edp_csv():
    return TESTS_ROOT_PATH / "data/edp_csv.json"


@fixture
def path_edp_pdf():
    return TESTS_ROOT_PATH / "data/edp_pdf.json"
