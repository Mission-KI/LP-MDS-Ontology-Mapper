from pathlib import Path

from pandas import DataFrame, Timestamp

from edp.analyzers.pandas import Pandas, _ColumnType
from edp.file import File

DUMMY_FILE = Path(__file__).parent.parent / "data/test.csv"


async def test_determine_datetime_iso_Ymd_HMS():
    analyzer = buildPandasAnalyzer(
        [
            "2016-03-01 00:03:14",
            "1997-04-02 00:26:57",
            "2045-06-30 13:29:53",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_iso_Ymd_HMSm():
    analyzer = buildPandasAnalyzer(
        [
            "2016-03-01 00:03:14.123",
            "1997-04-02 00:26:57.999",
            "2045-06-30 13:29:53.000",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(
        year=2016, month=3, day=1, hour=0, minute=3, second=14, microsecond=123000
    )
    assert analyzer._data["col"][1] == Timestamp(
        year=1997, month=4, day=2, hour=0, minute=26, second=57, microsecond=999000
    )
    assert analyzer._data["col"][2] == Timestamp(
        year=2045, month=6, day=30, hour=13, minute=29, second=53, microsecond=0
    )


async def test_determine_datetime_iso_Ymd_HM():
    analyzer = buildPandasAnalyzer(
        [
            "2016-03-01 00:03",
            "1997-04-02 00:26",
            "2045-06-30 13:29",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


async def test_determine_datetime_iso_Ymd():
    analyzer = buildPandasAnalyzer(
        [
            "2016-03-01",
            "1997-04-02",
            "2045-06-30",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_iso_Ymd_short():
    analyzer = buildPandasAnalyzer(
        [
            "20160301",
            "19970402",
            "20450630",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_iso_YmdTHMS():
    analyzer = buildPandasAnalyzer(
        [
            "20160301T00:03:14",
            "19970402T00:26:57",
            "20450630T13:29:53",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_iso_YmdTHMS_UTC():
    analyzer = buildPandasAnalyzer(
        [
            "20160301T00:03:14Z",
            "19970402T00:26:57Z",
            "20450630T13:29:53Z",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC")
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC")
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC")


async def test_determine_datetime_iso_YmdTHMS_TZ():
    analyzer = buildPandasAnalyzer(
        [
            "20160301T00:03:14+0100",
            "19970402T00:26:57+0100",
            "20450630T13:29:53+0100",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC+0100")
    assert analyzer._data["col"][2] == Timestamp(
        year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0100"
    )


async def test_determine_datetime_de_dmY_HMS():
    analyzer = buildPandasAnalyzer(
        [
            "01.03.2016 00:03:14",
            "02.04.1997 00:26:57",
            "30.06.2045 13:29:53",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_de_dmY_HM():
    analyzer = buildPandasAnalyzer(
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26",
            "30.06.2045 13:29",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


async def test_determine_datetime_de_dmY():
    analyzer = buildPandasAnalyzer(
        [
            "01.03.2016",
            "02.04.1997",
            "30.06.2045",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=2016, month=3, day=1)
    assert analyzer._data["col"][1] == Timestamp(year=1997, month=4, day=2)
    assert analyzer._data["col"][2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_de_HMS():
    analyzer = buildPandasAnalyzer(
        [
            "00:03:14",
            "00:26:57",
            "13:29:53",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=1900, month=1, day=1, hour=0, minute=3, second=14)
    assert analyzer._data["col"][1] == Timestamp(year=1900, month=1, day=1, hour=0, minute=26, second=57)
    assert analyzer._data["col"][2] == Timestamp(year=1900, month=1, day=1, hour=13, minute=29, second=53)


async def test_determine_datetime_de_HM():
    analyzer = buildPandasAnalyzer(
        [
            "00:03",
            "00:26",
            "13:29",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.DateTime] == ["col"]
    assert analyzer._data["col"].dtype.kind == "M"
    assert analyzer._data["col"][0] == Timestamp(year=1900, month=1, day=1, hour=0, minute=3)
    assert analyzer._data["col"][1] == Timestamp(year=1900, month=1, day=1, hour=0, minute=26)
    assert analyzer._data["col"][2] == Timestamp(year=1900, month=1, day=1, hour=13, minute=29)


async def test_determine_datetime_mixed():
    analyzer = buildPandasAnalyzer(
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26:57",
            "30.06.2045",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.String] == ["col"]
    assert analyzer._data["col"].dtype.kind == "O"
    assert analyzer._data["col"][0] == "01.03.2016 00:03"


async def test_determine_unknown():
    analyzer = buildPandasAnalyzer(
        [
            "01-03-2016",
            "02-04-1997",
            "30-06-2045",
        ]
    )
    types = await analyzer._determine_types()
    assert types[_ColumnType.String] == ["col"]
    assert analyzer._data["col"].dtype.kind == "O"
    assert analyzer._data["col"][0] == "01-03-2016"


def buildPandasAnalyzer(col: list[str]):
    df = DataFrame(
        {
            "col": col,
        }
    )
    f = File(DUMMY_FILE.parent, DUMMY_FILE)
    return Pandas(df, f)
