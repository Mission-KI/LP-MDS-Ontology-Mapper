from pathlib import Path

from edps.file import File
from edps.importers.structured import csv_import_dataframe, excel_import_dataframes


async def test_import_csv(path_data_test_csv, ctx):
    file = get_file(path_data_test_csv)
    data = await csv_import_dataframe(ctx, file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_csv_no_headers(path_data_test_headerless_csv, ctx):
    file = get_file(path_data_test_headerless_csv)
    data = await csv_import_dataframe(ctx, file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["col000", "col001", "col002", "col003", "col004"]


async def test_import_csv_with_clevercsv(ctx, tmp_path: Path):
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
    data = await csv_import_dataframe(ctx, file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert headers == ["Idx", "1 col", "Unnamed: 2", "3 col", "Unnamed: 4", "5 col", "Unnamed: 6"]
    assert col_count == 7
    assert row_count == 2
    first_row = data.iloc[0]
    assert first_row.to_list() == [2022.0, 28.0, 16.8, 49.0, 5.7, 78.0, 4.5]


async def test_import_csv_with_quoted_strings(ctx, tmp_path: Path):
    csv_file = tmp_path / "clever.csv"
    csv_file.write_text(
        """id;value;num
1;"hello";2023
2;"world";2024
"""
    )
    file = get_file(csv_file)
    data = await csv_import_dataframe(ctx, file)
    assert data.to_dict() == {
        "id": {0: 1, 1: 2},  # TODO(KB) index needs to be fixed
        "value": {0: "hello", 1: "world"},
        "num": {0: 2023, 1: 2024},
    }


async def test_import_xlsx(path_data_test_xlsx, ctx):
    file = get_file(path_data_test_xlsx)
    dataframes_map = await excel_import_dataframes(ctx, file, "openpyxl")
    data = list(dataframes_map.values())[0]
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_xls(path_data_test_xls, ctx):
    file = get_file(path_data_test_xls)
    dataframes_map = await excel_import_dataframes(ctx, file, "xlrd")
    data = list(dataframes_map.values())[0]
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_detect_german_decimal_comma(path_data_german_decimal_comma_csv, ctx):
    file = get_file(path_data_german_decimal_comma_csv)
    data = await csv_import_dataframe(ctx, file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 15
    assert row_count == 6
    assert headers == [
        "Zeit",
        "0-17 Jahre",
        "Unnamed: 2",
        "18-24 Jahre",
        "Unnamed: 4",
        "25-29 Jahre",
        "Unnamed: 6",
        "30-49 Jahre",
        "Unnamed: 8",
        "50-64 Jahre",
        "Unnamed: 10",
        "65 Jahre und aelter",
        "Unnamed: 12",
        "Insgesamt",
        "Unnamed: 14",
    ]


async def test_hamburg(path_data_hamburg_csv, ctx):
    file = get_file(path_data_hamburg_csv)
    data = await csv_import_dataframe(ctx, file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 7
    assert row_count == 55
    assert headers == [
        "Unnamed: 0",
        "Unnamed: 1",
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4",
        "Unnamed: 5",
        "in Tsd. Euro",
    ]


def get_file(path: Path):
    return File(path.parent, path)
