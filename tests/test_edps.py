from pathlib import Path

import pytest
from pytest import mark

from edp import Service
from edp.types import DataSetType

DIR = Path(__file__).parent
ENCODING = "utf-8"
CSV_PATH = DIR / "data/test.csv"


@mark.asyncio
async def test_load_unknown_dir():
    service = Service()
    with pytest.raises(FileNotFoundError):
        await service.analyse_asset(Path("/does/not/exist/"))


@mark.asyncio
async def test_load_pickle_dir():
    service = Service()
    result = await service.analyse_asset(CSV_PATH)
    with open(DIR.parent / "output/csv_edp.json", "wt", encoding=ENCODING) as file:
        file.write(result.model_dump_json())
    assert len(result.dataTypes) == 1
    assert DataSetType.structured in result.dataTypes
