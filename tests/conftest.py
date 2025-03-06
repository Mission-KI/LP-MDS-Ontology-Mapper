import shutil
from logging import getLogger
from pathlib import Path

from easyocr import easyocr
from pytest import fixture

from edps.filewriter import setup_matplotlib
from edps.taskcontext import TaskContext
from edps.taskcontextimpl import TaskContextImpl

TESTS_ROOT_PATH = Path(__file__).parent.absolute()


@fixture(autouse=True, scope="session")
def setup():
    setup_matplotlib()


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
def path_data_test_pdf():
    return TESTS_ROOT_PATH / "data/test.pdf"


@fixture
def path_data_test_docx():
    return TESTS_ROOT_PATH / "data/test.docx"


@fixture
def path_data_test_multiassets_zip():
    return TESTS_ROOT_PATH / "data/test_multiassets.zip"


@fixture
def path_data_test_png():
    return TESTS_ROOT_PATH / "data/test.png"


@fixture
def path_data_test_jpg():
    return TESTS_ROOT_PATH / "data/test.jpg"


@fixture
def path_data_test_jpeg():
    return TESTS_ROOT_PATH / "data/test.jpeg"


@fixture
def path_data_test_gif():
    return TESTS_ROOT_PATH / "data/test.gif"


@fixture
def path_data_test_bmp():
    return TESTS_ROOT_PATH / "data/test.bmp"


@fixture
def path_data_test_tiff():
    return TESTS_ROOT_PATH / "data/test.tiff"


@fixture
def path_data_test_tif():
    return TESTS_ROOT_PATH / "data/test.tif"


@fixture
def path_data_test_webp():
    return TESTS_ROOT_PATH / "data/test.webp"


@fixture
def path_data_test_mp4():
    return TESTS_ROOT_PATH / "data/test.mp4"


@fixture
def path_data_test_avi():
    return TESTS_ROOT_PATH / "data/test.avi"


@fixture
def path_data_test_mkv():
    return TESTS_ROOT_PATH / "data/test.mkv"


@fixture
def path_data_test_mov():
    return TESTS_ROOT_PATH / "data/test.mov"


@fixture
def path_data_test_flv():
    return TESTS_ROOT_PATH / "data/test.flv"


@fixture
def path_data_test_wmv():
    return TESTS_ROOT_PATH / "data/test.wmv"


@fixture
def path_data_test_with_text():
    return TESTS_ROOT_PATH / "data/test_with_text.png"


@fixture
def path_data_pontusx_algocustomdata():
    return TESTS_ROOT_PATH / "data/pontusx/algoCustomData.json"


@fixture
def path_data_pontusx_ddo():
    return TESTS_ROOT_PATH / "data/pontusx/ddo.json"


@fixture
def path_unstructured_text_only_txt():
    return TESTS_ROOT_PATH / "data/unstructured_text_only.txt"


@fixture
def path_unstructured_text_with_table():
    return TESTS_ROOT_PATH / "data/unstructured_text_with_table.txt"


@fixture
def path_language_deu_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_wiki_llm.txt"


@fixture
def path_language_deu_eng_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_eng_wiki_llm.txt"


@fixture
def ctx(path_work) -> TaskContext:
    return TaskContextImpl(getLogger("edps.test"), path_work)


@fixture
def download_ocr_models(ctx):
    ctx.logger.info("Downloading OCR models.")
    easyocr.Reader(lang_list=["en", "de"], gpu=False, download_enabled=True)


def copy_asset_to_ctx_input_dir(asset_path: Path, ctx: TaskContext):
    shutil.rmtree(ctx.input_path, ignore_errors=True)
    ctx.input_path.mkdir()
    dest_path = ctx.input_path / asset_path.name
    shutil.copy(asset_path, dest_path)
    return dest_path
