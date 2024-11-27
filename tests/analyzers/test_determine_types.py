from datetime import timedelta, timezone
from pathlib import Path

from pandas import DataFrame, Timestamp

from edp.analyzers.pandas.type_parser import (
    DatetimeColumnInfo,
    StringColumnInfo,
    parse_types,
)

DUMMY_FILE = Path(__file__).parent.parent / "data/test.csv"


async def test_determine_datetime_iso_Ymd_HMS():
    col, info = await parse_datetime_col(
        [
            "2016-03-01 00:03:14",
            "1997-04-02 00:26:57",
            "2045-06-30 13:29:53",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_iso_Ymd_HMSm():
    col, info = await parse_datetime_col(
        [
            "2016-03-01 00:03:14.123",
            "1997-04-02 00:26:57.999",
            "2045-06-30 13:29:53.000",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, microsecond=123000)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, microsecond=999000)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, microsecond=0)


async def test_determine_datetime_iso_Ymd_HM():
    col, info = await parse_datetime_col(
        [
            "2016-03-01 00:03",
            "1997-04-02 00:26",
            "2045-06-30 13:29",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


async def test_determine_datetime_iso_Ymd():
    col, info = await parse_datetime_col(
        [
            "2016-03-01",
            "1997-04-02",
            "2045-06-30",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_iso_Ymd_short():
    col, info = await parse_datetime_col(
        [
            "20160301",
            "19970402",
            "20450630",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_iso_YmdTHMS():
    col, info = await parse_datetime_col(
        [
            "20160301T00:03:14",
            "19970402T00:26:57",
            "20450630T13:29:53",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_iso_YmdTHMS_UTC():
    col, info = await parse_datetime_col(
        [
            "20160301T00:03:14Z",
            "19970402T00:26:57Z",
            "20450630T13:29:53Z",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz == timezone.utc for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC")


async def test_determine_datetime_iso_YmdTHMS_TZ():
    col, info = await parse_datetime_col(
        [
            "20160301T00:03:14+0100",
            "19970402T00:26:57+0100",
            "20450630T13:29:53+0100",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz == timezone(timedelta(hours=1)) for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC+0100")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0100")


async def test_determine_datetime_iso_YmdTHMS_TZ_different():
    # For different time zones we need to_datetime(utc=True). This normalizes tz to UTC!
    col, info = await parse_datetime_col(
        [
            "20160301T00:03:14+0100",
            "19970402T00:26:57+0200",
            "20450630T13:29:53+0300",
            "20450607T03:30:00+0230",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz == timezone.utc for i in range(4))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC+0200")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0300")
    assert col[3] == Timestamp(year=2045, month=6, day=7, hour=1, minute=0, second=0, tz="UTC")


async def test_determine_datetime_iso_YmdTHMS_TZ_partially():
    # For mixed datetimes with and without tz we need to_datetime(utc=True). This normalizes tz to UTC!
    # For datetimes without tz the parser assumes the same tz as the previous value.
    col, info = await parse_datetime_col(
        [
            "20160301T00:03:14+0100",
            "20450630T13:29:53+0300",
            "20450607T03:30:00",
            "20450607T04:30:00",
        ]
    )
    assert info.format == "ISO8601"
    assert all(col[i].tz == timezone.utc for i in range(4))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0300")
    assert col[2] == Timestamp(year=2045, month=6, day=7, hour=0, minute=30, second=0, tz="UTC")
    assert col[3] == Timestamp(year=2045, month=6, day=7, hour=1, minute=30, second=0, tz="UTC")


async def test_determine_datetime_de_dmY_HMS():
    col, info = await parse_datetime_col(
        [
            "01.03.2016 00:03:14",
            "02.04.1997 00:26:57",
            "30.06.2045 13:29:53",
        ]
    )
    assert info.format == "%d.%m.%Y %H:%M:%S"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


async def test_determine_datetime_de_dmY_HM():
    col, info = await parse_datetime_col(
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26",
            "30.06.2045 13:29",
        ]
    )
    assert info.format == "%d.%m.%Y %H:%M"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


async def test_determine_datetime_de_dmY():
    col, info = await parse_datetime_col(
        [
            "01.03.2016",
            "02.04.1997",
            "30.06.2045",
        ]
    )
    assert info.format == "%d.%m.%Y"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


async def test_determine_datetime_de_HMS():
    col, info = await parse_col(
        [
            "00:03:14",
            "00:26:57",
            "13:29:53",
        ]
    )
    # For now a pure time column is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert col.dtype.kind == "O"


async def test_determine_datetime_de_HM():
    col, info = await parse_col(
        [
            "00:03",
            "00:26",
            "13:29",
        ]
    )
    # For now a pure time column is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert col.dtype.kind == "O"


async def test_determine_datetime_mixed():
    col, info = await parse_col(
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26:57",
            "30.06.2045",
        ]
    )
    # A column with mixed datetime formats is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert col.dtype.kind == "O"
    assert col[0] == "01.03.2016 00:03"


async def test_determine_unknown():
    col, info = await parse_col(
        [
            "01-03-2016",
            "02-04-1997",
            "30-06-2045",
        ]
    )
    # A column with a unknown format is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert col.dtype.kind == "O"
    assert col[0] == "01-03-2016"


async def parse_datetime_col(col: list[str]):
    parsed_col, col_info = await parse_col(col)
    assert isinstance(col_info, DatetimeColumnInfo)
    return parsed_col, col_info


async def parse_col(col: list[str]):
    df = DataFrame(
        {
            "col": col,
        }
    )
    result = await parse_types(df)
    cols = result.all_cols
    return cols.get_col("col"), cols.get_info("col")
