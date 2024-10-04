from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from pytest import fixture, mark, raises

from edp import Service
from edp.context import OutputLocalFilesContext
from edp.types import (
    Asset,
    DataSetType,
    DataSpace,
    DateTimeColumn,
    NumericColumn,
    Publisher,
    StringColumn,
    UserProvidedAssetData,
)

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH = DIR / "data/test.csv"
PICKLE_PATH = DIR / "data/test.pickle"


@fixture
def output_context(output_directory):
    return OutputLocalFilesContext(output_directory)


@fixture
def user_data():
    return UserProvidedAssetData(
        id="my-dataset-id",
        name="dataset-dummy-name",
        url="https://beebucket.ai/en/",
        dataCategory="TestDataCategory",
        dataSpace=DataSpace(dataSpaceId=1, name="TestDataSpace", url="https://beebucket.ai/en/"),
        publisher=Publisher(id="0815-1234", name="beebucket"),
        licenseId=0,
        description="Our very first test edp",
        publishDate=datetime(year=1995, month=10, day=10, hour=10, tzinfo=timezone.utc),
        version="2.3.1",
        tags=["test", "csv"],
    )


@mark.asyncio
async def test_load_unknown_dir(output_context):
    service = Service()
    with raises(FileNotFoundError):
        await service.analyse_asset(Path("/does/not/exist/"), output_context)


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}


@mark.asyncio
async def test_analyse_csv(output_directory, output_context, user_data):
    service = Service()
    result = await service.analyse_asset(CSV_PATH, output_context)
    assert len(result.datasets) == 1
    assert len(result.datasets["test.csv"].columns) == 5
    dataset = result.datasets["test.csv"]
    assert isinstance(dataset.columns["uuid"], StringColumn)
    assert isinstance(dataset.columns["einfahrt"], DateTimeColumn)
    assert isinstance(dataset.columns["ausfahrt"], DateTimeColumn)
    aufenthalt = dataset.columns["aufenthalt"]
    assert isinstance(aufenthalt, NumericColumn)
    assert aufenthalt.dataType == "uint32"
    parkhaus = dataset.columns["parkhaus"]
    assert isinstance(parkhaus, NumericColumn)
    assert parkhaus.dataType == "uint8"

    asset = Asset(**_as_dict(result), **_as_dict(user_data))
    with open(output_directory / "csv_edp.json", "wt", encoding=ENCODING) as file:
        file.write(asset.model_dump_json())
    assert len(result.dataTypes) == 1
    assert DataSetType.structured in result.dataTypes


async def test_analyse_pickle(output_context):
    service = Service()
    await service.analyse_asset(PICKLE_PATH, output_context)
