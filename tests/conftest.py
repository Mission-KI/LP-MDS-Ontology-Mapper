import pickle
import shutil
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path

from easyocr import easyocr
from extended_dataset_profile.models.v0.edp import (
    AssetReference,
    DataSpace,
    License,
    Publisher,
)
from pydantic import HttpUrl
from pytest import fixture

from edps.filewriter import setup_matplotlib
from edps.importers.structured import csv_import_dataframe
from edps.taskcontext import TaskContext
from edps.taskcontextimpl import TaskContextImpl
from edps.types import Config, UserProvidedEdpData

TESTS_ROOT_PATH = Path(__file__).parent.absolute()


@fixture(autouse=True, scope="session")
def setup():
    setup_matplotlib()


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
def path_data_test_csv():
    return TESTS_ROOT_PATH / "data/test.csv"


@fixture(scope="session")
def path_data_test_counts_csv():
    return TESTS_ROOT_PATH / "data/test_counts.csv"


@fixture
def path_data_bast_csv():
    return TESTS_ROOT_PATH / "data/bast.csv"


@fixture
def path_data_test_extra_headers_csv():
    return TESTS_ROOT_PATH / "data/test_extra_headers.csv"


@fixture
def path_data_test_missing_headers_csv():
    return TESTS_ROOT_PATH / "data/test_missing_headers.csv"


@fixture
def path_data_test_headerless_csv():
    return TESTS_ROOT_PATH / "data/test_headerless.csv"


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
def path_data_test_json():
    return TESTS_ROOT_PATH / "data/test.json"


@fixture
def path_data_test_without_structured_data():
    return TESTS_ROOT_PATH / "data/test_without_structured_data.json"


@fixture
def path_data_test_with_normalization():
    return TESTS_ROOT_PATH / "data/test_with_normalization.json"


@fixture
def path_language_deu_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_wiki_llm.txt"


@fixture
def path_language_deu_eng_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_eng_wiki_llm.txt"


@fixture(scope="session")
def user_provided_data():
    return UserProvidedEdpData(
        assetRefs=[
            AssetReference(
                assetId="my-dataset-id",
                dataSpace=DataSpace(name="TestDataSpace", url="https://beebucket.ai/en/"),
                assetUrl=HttpUrl("https://beebucket.ai/en/"),
                assetVersion="2.3.1",
                publisher=Publisher(name="beebucket"),
                publishDate=datetime(year=1995, month=10, day=10, hour=10, tzinfo=timezone.utc),
                license=License(url="https://opensource.org/license/mit"),
            )
        ],
        name="dataset-dummy-name",
        dataCategory="TestDataCategory",
        description="Our very first test edp",
        tags=["test", "csv"],
        freely_available=True,
    )


@fixture(scope="session")
def config_data(user_provided_data):
    return Config(userProvidedEdpData=user_provided_data, augmentedColumns=[])


@fixture(scope="session")
def logger():
    return getLogger("edps.test")


@fixture
def ctx(path_work, config_data, logger) -> TaskContext:
    return TaskContextImpl(config_data, logger, path_work)


@fixture(scope="session")
def download_ocr_models(logger):
    logger.info("Downloading OCR models.")
    easyocr.Reader(lang_list=["en", "de"], gpu=False, download_enabled=True)


def copy_asset_to_ctx_input_dir(asset_path: Path, ctx: TaskContext):
    shutil.rmtree(ctx.input_path, ignore_errors=True)
    ctx.input_path.mkdir()
    dest_path = ctx.input_path / asset_path.name
    shutil.copy(asset_path, dest_path)
    return dest_path


@fixture
async def path_data_test_pickle(ctx, path_data_test_csv, tmp_path):
    # Freshly pickle the dataframe from the CSV
    dataframe = await csv_import_dataframe(ctx, path_data_test_csv)
    pickle_path = tmp_path / "test.pickle"
    with open(pickle_path, "wb") as file:  # Use "wb" mode to write in binary
        pickle.dump(dataframe, file)
    return pickle_path
