from datetime import datetime, timezone
from pathlib import Path

from pydantic import HttpUrl
from pytest import fixture, mark, raises

from edp import Service
from edp.types import (
    Augmentation,
    AugmentedColumn,
    Config,
    DataSetType,
    DataSpace,
    License,
    Publisher,
    UserProvidedEdpData,
)

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH = DIR / "data/test.csv"
PICKLE_PATH = DIR / "data/test.pickle"


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

    dataset = computed_data.structuredDatasets[0]
    assert len(dataset.datetimeColumns) == 2
    assert len(dataset.numericColumns) == 2
    assert len(dataset.stringColumns) == 1

    aufenthalt = next(item for item in dataset.numericColumns if item.name == "aufenthalt")
    assert aufenthalt.dataType == "uint32"
    assert aufenthalt.augmentation == config_data.augmentedColumns[0].augmentation

    parkhaus = next(item for item in dataset.numericColumns if item.name == "parkhaus")
    assert parkhaus.dataType == "uint8"

    assert len(computed_data.dataTypes) == 1
    assert DataSetType.structured in computed_data.dataTypes


@mark.asyncio
async def test_analyse_csv(output_context, config_data):
    service = Service()
    await service.analyse_asset(CSV_PATH, config_data, output_context)


@mark.asyncio
async def test_raise_on_only_unknown_datasets(tmp_path, output_context, config_data):
    service = Service()
    file_path = tmp_path / "unsupported.txt"
    with open(file_path, "wt", encoding="utf-8") as file:
        file.write("This type is not supported")
    with raises((RuntimeWarning, RuntimeError)):
        await service.analyse_asset(tmp_path, config_data, output_context)
