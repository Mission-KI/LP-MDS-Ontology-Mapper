from pathlib import Path

import pytest
from pydantic import BaseModel
from pytest import mark

from edp import Service
from edp.types import Asset, DataSetType, DataSpace, Publisher, UserProvidedAssetData

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH = DIR / "data/test.csv"


@mark.asyncio
async def test_load_unknown_dir():
    service = Service()
    with pytest.raises(FileNotFoundError):
        await service.analyse_asset(Path("/does/not/exist/"))


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}


@mark.asyncio
async def test_load_pickle_dir(output_directory):
    service = Service()
    result = await service.analyse_asset(CSV_PATH)
    user_data = UserProvidedAssetData(
        id=1234,
        name="BeebucketCsv",
        url="https://beebucket.ai/en/",
        dataCategory="TestData",
        dataSpace=DataSpace(dataSpaceId=1, name="TestDataSpace", url="https://beebucket.ai/en/"),
        publisher=Publisher(id="0815-1234", name="beebucket"),
        licenseId=0,
        description="Our very first test edp",
        tags=["test", "csv"],
    )
    asset = Asset(**_as_dict(result), **_as_dict(user_data))
    with open(output_directory / "csv_edp.json", "wt", encoding=ENCODING) as file:
        file.write(asset.model_dump_json())
    assert len(result.dataTypes) == 1
    assert DataSetType.structured in result.dataTypes
