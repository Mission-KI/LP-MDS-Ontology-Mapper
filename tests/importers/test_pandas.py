from pathlib import Path

from edp.file import File
from edp.importers.pandas import csv_importer, xls_importer, xlsx_importer

DIR = Path(__file__).parent.parent
CSV_PATH = DIR / "data/test.csv"
CSV_HEADERLESS_PATH = DIR / "data/test_headerless.csv"
XLSX_PATH: Path = DIR / "data/test.xlsx"
XLS_PATH: Path = DIR / "data/test.xls"


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


async def test_import_csv_with_clevercsv(tmp_path: Path):
    """Test the advanced detection of clevercsv.

    The standard csv would fail with 'Could not determine delimiter'
    """

    csv_file = tmp_path / "clever.csv"
    csv_file.write_text(
        """Idx;1 col;;3 col;;5 col;
2022;28 ;16.8 ;49 ;5.7 ;78 ;4.5
2023;42 ;17.0 ;28 ;5.3 ;61 ;4.3
"""
    )
    file = get_file(csv_file)
    analyzer = await csv_importer(file)
    data = analyzer._data
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert headers == ["Idx", "1 col", "Unnamed: 2", "3 col", "Unnamed: 4", "5 col", "Unnamed: 6"]
    assert col_count == 7
    assert row_count == 2
    first_row = data.iloc[0]
    assert first_row.to_list() == [2022.0, 28.0, 16.8, 49.0, 5.7, 78.0, 4.5]


async def test_import_csv_with_quoted_strings(tmp_path: Path):
    csv_file = tmp_path / "clever.csv"
    csv_file.write_text(
        """id;value;num
1;"hello";2023
2;"world";2024
"""
    )
    file = get_file(csv_file)
    analyzer = await csv_importer(file)
    data = analyzer._data
    assert data.to_dict() == {
        "id": {0: 1, 1: 2},  # TODO(KB) index needs to be fixed
        "value": {0: "hello", 1: "world"},
        "num": {0: 2023, 1: 2024},
    }


async def test_import_xlsx():
    file = get_file(XLSX_PATH)
    analyzer = await xlsx_importer(file)
    data = analyzer._data
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_xls():
    file = get_file(XLS_PATH)
    analyzer = await xls_importer(file)
    data = analyzer._data
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


def get_file(path: Path):
    return File(path.parent, path)
