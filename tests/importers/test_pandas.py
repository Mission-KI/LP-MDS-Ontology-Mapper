from pathlib import Path

from edp.file import File
from edp.importers.pandas import csv as csv_importer

DIR = Path(__file__).parent.parent
CSV_PATH = DIR / "data/test.csv"
CSV_HEADERLESS_PATH = DIR / "data/test_headerless.csv"


async def test_import_csv():
    file = get_file(CSV_PATH)
    analyzer = await csv_importer(file)
    data = analyzer._data
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_csv_no_headers():
    file = get_file(CSV_HEADERLESS_PATH)
    analyzer = await csv_importer(file)
    data = analyzer._data
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["col000", "col001", "col002", "col003", "col004"]


def get_file(path: Path):
    return File(path.parent, path)
