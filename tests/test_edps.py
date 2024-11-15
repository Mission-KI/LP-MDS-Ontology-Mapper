from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from pytest import fixture, mark, raises

from edp import Service
from edp.types import (
    Augmentation,
    AugmentedColumn,
    Config,
    DataSetType,
    DataSpace,
    ExtendedDatasetProfile,
    License,
    Publisher,
    UserProvidedEdpData,
)

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH = DIR / "data/test.csv"
CSV_HEADERLESS_PATH = DIR / "data/test_headerless.csv"
PICKLE_PATH = DIR / "data/test.pickle"
ZIP_PATH = DIR / "data/test.zip"


@fixture
def user_provided_data():
    return UserProvidedEdpData(
        assetId="my-dataset-id",
        name="dataset-dummy-name",
        url="https://beebucket.ai/en/",
        dataCategory="TestDataCategory",
        dataSpace=DataSpace(dataSpaceId=1, name="TestDataSpace", url="https://beebucket.ai/en/"),
        publisher=Publisher(id="0815-1234", name="beebucket"),
        license=License(name="my-very-own-license-id"),
        description="Our very first test edp",
        publishDate=datetime(year=1995, month=10, day=10, hour=10, tzinfo=timezone.utc),
        version="2.3.1",
        tags=["test", "csv"],
    )


@fixture
def config_data(user_provided_data):
    augmented_columns = [
        AugmentedColumn(name="aufenthalt", augmentation=Augmentation(sourceColumns=["einfahrt", "ausfahrt"]))
    ]
    return Config(userProvidedEdpData=user_provided_data, augmentedColumns=augmented_columns)


@mark.asyncio
async def test_load_unknown_dir(output_context, config_data):
    service = Service()
    with raises(FileNotFoundError):
        await service._compute_asset(Path("/does/not/exist/"), config_data, output_context)


async def test_analyse_pickle(output_context, config_data):
    service = Service()
    computed_data = await service._compute_asset(PICKLE_PATH, config_data, output_context)
    assert len(computed_data.structuredDatasets) == 1
    assert computed_data.periodicity == "h"

    dataset = computed_data.structuredDatasets[0]
    assert len(dataset.datetimeColumns) == 2
    assert len(dataset.numericColumns) == 2
    assert len(dataset.stringColumns) == 1

    aufenthalt = next(item for item in dataset.numericColumns if item.name == "aufenthalt")
    assert aufenthalt.dataType == "uint32"
    assert aufenthalt.augmentation == config_data.augmentedColumns[0].augmentation

    parkhaus = next(item for item in dataset.numericColumns if item.name == "parkhaus")
    assert parkhaus.dataType == "uint8"

    einfahrt = next(item for item in dataset.datetimeColumns if item.name == "einfahrt")
    assert einfahrt.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert einfahrt.temporalCover.latest == datetime.fromisoformat("2016-01-01 11:50:45")
    assert einfahrt.periodicity == "h"

    assert len(computed_data.dataTypes) == 1
    assert DataSetType.structured in computed_data.dataTypes

    assert computed_data.temporalCover is not None
    assert computed_data.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert computed_data.temporalCover.latest == datetime.fromisoformat("2016-01-01 21:13:08")


@mark.asyncio
async def test_analyse_csv(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(CSV_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_analyse_csv_no_headers(output_context, user_provided_data):
    config_data = Config(userProvidedEdpData=user_provided_data)
    service = Service()
    json_file = await service.analyse_asset(CSV_HEADERLESS_PATH, config_data, output_context)
    edp: ExtendedDatasetProfile = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
    assert edp.compression is None
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_analyse_zip(output_context, config_data):
    service = Service()
    json_file = await service.analyse_asset(ZIP_PATH, config_data, output_context)
    edp = read_edp(output_context.build_full_path(json_file))
    assert edp.assetId == config_data.userProvidedEdpData.assetId
    assert "zip" in edp.compression.algorithms
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_raise_on_only_unknown_datasets(tmp_path, output_context, config_data):
    service = Service()
    file_path = tmp_path / "unsupported.txt"
    with open(file_path, "wt", encoding="utf-8") as file:
        file.write("This type is not supported")
    with raises((RuntimeWarning, RuntimeError)):
        await service.analyse_asset(tmp_path, config_data, output_context)


def read_edp(json_file: PurePosixPath):
    with open(json_file, "r") as file:
        json_data = file.read()
    return ExtendedDatasetProfile.model_validate_json(json_data)
