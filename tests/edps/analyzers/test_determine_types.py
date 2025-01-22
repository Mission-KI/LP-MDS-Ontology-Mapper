from datetime import datetime, timedelta, timezone

import numpy.dtypes as numpytype
import pandas as pd
from pandas import DataFrame, Series, Timestamp
from pytest import fixture, mark

from edps.analyzers.pandas.type_parser import (
    DatetimeColumnInfo,
    DatetimeIsoColumnInfo,
    DatetimeKind,
    DatetimePatternColumnInfo,
    NumericColumnInfo,
    StringColumnInfo,
    parse_types,
)
from edps.task import TaskContext


# Test type parsing
def test_determine_datetime_iso_Ymd_HMS(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "2016-03-01 00:03:14",
            "1997-04-02 00:26:57",
            "2045-06-30 13:29:53",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert info.get_kind() == DatetimeKind.UNKNOWN  # for ISO we can't determine if it's a DATE or DATETIME
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


def test_determine_datetime_iso_Ymd_HMSm(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "2016-03-01 00:03:14.123",
            "1997-04-02 00:26:57.999",
            "2045-06-30 13:29:53.000",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, microsecond=123000)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, microsecond=999000)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, microsecond=0)


def test_determine_datetime_iso_Ymd_HM(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "2016-03-01 00:03",
            "1997-04-02 00:26",
            "2045-06-30 13:29",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


def test_determine_datetime_iso_Ymd(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "2016-03-01",
            "1997-04-02",
            "2045-06-30",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


def test_determine_datetime_iso_Ymd_short(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "20160301",
            "19970402",
            "20450630",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


def test_determine_datetime_iso_YmdTHMS(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "20160301T00:03:14",
            "19970402T00:26:57",
            "20450630T13:29:53",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


def test_determine_datetime_iso_YmdTHMS_UTC(
    ctx,
):
    col, info = expect_datetime_column(
        ctx,
        [
            "20160301T00:03:14Z",
            "19970402T00:26:57Z",
            "20450630T13:29:53Z",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns, UTC]"
    assert all(col[i].tz == timezone.utc for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC")


def test_determine_datetime_iso_YmdTHMS_TZ(
    ctx,
):
    col, info = expect_datetime_column(
        ctx,
        [
            "20160301T00:03:14+0100",
            "19970402T00:26:57+0100",
            "20450630T13:29:53+0100",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns, UTC]"
    assert all(col[i].tz == timezone.utc for i in range(3))  # original TZ is not preserved
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC+0100")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0100")


def test_determine_datetime_iso_YmdTHMS_TZ_different(
    ctx,
):
    # For different time zones we need to_datetime(utc=True). This normalizes tz to UTC!
    col, info = expect_datetime_column(
        ctx,
        [
            "20160301T00:03:14+0100",
            "19970402T00:26:57+0200",
            "20450630T13:29:53+0300",
            "20450607T03:30:00+0230",
        ],
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns, UTC]"
    assert all(col[i].tz == timezone.utc for i in range(4))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57, tz="UTC+0200")
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0300")
    assert col[3] == Timestamp(year=2045, month=6, day=7, hour=1, minute=0, second=0, tz="UTC")


def test_determine_datetime_iso_YmdTHMS_TZ_partially(
    ctx,
):
    # For mixed datetimes with and without tz we need conversion to UTC!
    col, info = expect_datetime_column(
        ctx, ["20160301T00:03:14+0100", "20450630T13:29:53+0300", "20450607T03:30:00", "20450607T04:30:00", "2024"]
    )
    assert info.get_format() == "ISO8601"
    assert str(col.dtype) == "datetime64[ns, UTC]"
    assert all(col[i].tz == timezone.utc for i in range(4))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14, tz="UTC+0100")
    assert col[1] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53, tz="UTC+0300")
    assert col[2] == Timestamp(year=2045, month=6, day=7, hour=3, minute=30, second=0, tz="UTC")
    assert col[3] == Timestamp(year=2045, month=6, day=7, hour=4, minute=30, second=0, tz="UTC")
    assert pd.isna(col[4])


def test_determine_datetime_de_dmY_HMS(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "01.03.2016 00:03:14",
            "02.04.1997 00:26:57",
            "30.06.2045 13:29:53",
        ],
    )
    assert info.get_format() == "%d.%m.%Y %H:%M:%S"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


def test_determine_datetime_de_dmY_HM(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26",
            "30.06.2045 13:29",
        ],
    )
    assert info.get_format() == "%d.%m.%Y %H:%M"
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29)


def test_determine_datetime_de_dmY(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "01.03.2016",
            "02.04.1997",
            "30.06.2045",
        ],
    )
    assert info.get_format() == "%d.%m.%Y"
    assert info.get_kind() == DatetimeKind.DATE
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


def test_determine_datetime_de_HMS(ctx):
    col, info = parse_column(
        ctx,
        [
            "00:03:14",
            "00:26:57",
            "13:29:53",
        ],
    )
    # A pure time column is handled as a string column
    assert info.get_format() == "%H:%M:%S"
    assert info.get_kind() == DatetimeKind.TIME
    assert isinstance(col.dtype, pd.StringDtype)
    assert col[0] == "00:03:14"


def test_determine_datetime_de_HM(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "00:03",
            "00:26",
            "13:29",
        ],
    )
    # A pure time column is handled as a string column
    assert info.get_format() == "%H:%M"
    assert info.get_kind() == DatetimeKind.TIME
    assert isinstance(col.dtype, pd.StringDtype)
    assert col[0] == "00:03"


def test_determine_datetime_us_dmY_HMS(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "03/01/2016 00:03:14",
            "04/02/1997 00:26:57",
            "06/30/2045 13:29:53",
        ],
    )
    assert info.get_format() == "%m/%d/%Y %H:%M:%S"
    assert info.get_kind() == DatetimeKind.DATETIME
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1, hour=0, minute=3, second=14)
    assert col[1] == Timestamp(year=1997, month=4, day=2, hour=0, minute=26, second=57)
    assert col[2] == Timestamp(year=2045, month=6, day=30, hour=13, minute=29, second=53)


def test_determine_datetime_us_dmY(ctx):
    col, info = expect_datetime_column(
        ctx,
        [
            "03-01-2016",
            "04-02-1997",
            "06-30-2045",
        ],
    )
    assert info.get_format() == "%m-%d-%Y"
    assert info.get_kind() == DatetimeKind.DATE
    assert str(col.dtype) == "datetime64[ns]"
    assert all(col[i].tz is None for i in range(3))
    assert col[0] == Timestamp(year=2016, month=3, day=1)
    assert col[1] == Timestamp(year=1997, month=4, day=2)
    assert col[2] == Timestamp(year=2045, month=6, day=30)


def test_determine_datetime_mixed(ctx):
    col, info = parse_column(
        ctx,
        [
            "01.03.2016 00:03",
            "02.04.1997 00:26:57",
            "30.06.2045",
        ],
    )
    # A column with mixed datetime formats is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert isinstance(col.dtype, pd.StringDtype)
    assert col[0] == "01.03.2016 00:03"


def test_determine_unknown(ctx):
    col, info = parse_column(
        ctx,
        [
            "01_03_2016",
            "02_04_1997",
            "30_06_2045",
        ],
    )
    # A column with a unknown format is categorized as string column
    assert isinstance(info, StringColumnInfo)
    assert isinstance(col.dtype, pd.StringDtype)
    assert col[0] == "01_03_2016"


def test_determine_mixed_number_string(ctx):
    data = ["A"] + [str(i) for i in range(20)] + ["Z"]
    col, info = expect_numeric_column(ctx, data)
    assert col.notna().sum() == 20


def test_determine_number_downcast(ctx):
    data = ["2024-03-01", "01.06.2023 12:30", "-", "34", 42, "-324", "444", 343, 112, -1, 0]
    col, info = expect_numeric_column(ctx, data)
    assert isinstance(col.dtype, pd.Int16Dtype)
    assert col.notna().sum() == 8


def test_determine_mixed_datetimes1(ctx):
    # one datetime type and some invalid values
    data = ["A"] + ["01.03.1979"] * 20 + ["Z"] + [None] * 5
    col, info = expect_datetime_column(ctx, data)
    assert info.get_format() == "%d.%m.%Y"
    assert col.notna().sum() == 20


def test_determine_mixed_datetimes2(ctx):
    # clear majority for one datetime type
    data = ["A"] + ["01.03.1979 03:14", "2016-03-01 00:03:14", "01.03.1979 01:13"] * 25 + [None] * 3 + ["Z"]
    col, info = expect_datetime_column(ctx, data)
    assert info.get_format() == "%d.%m.%Y %H:%M"
    assert col.notna().sum() == 50
    assert col.isna().sum() == 30
    assert len(col.index) == 80
    assert (col.index == range(80)).all()


def test_determine_mixed_datetimes3(ctx):
    # no clear majority
    data = ["A"] + ["01.03.1979 03:14", "01.03.1979 03:14:17", "2016-03-01 00:03:14"] * 20 + ["Z"] + [None] * 5
    col, info = parse_column(ctx, data)
    assert isinstance(info, StringColumnInfo)
    assert col.notna().sum() == 62


# 1 million rows
BIGDATA_COUNT = 1000000


@mark.slow
def test_determine_uint32_bigdata(ctx, benchmark):
    data = Series(list(str(i % 1000000) for i in range(BIGDATA_COUNT)), dtype=str)
    benchmark.pedantic(expect_numeric_column, (ctx, data, pd.UInt32Dtype))


@mark.slow
def test_determine_uint8_bigdata(ctx, benchmark):
    data = Series(list(str(i % 256) for i in range(BIGDATA_COUNT)), dtype=str)
    benchmark.pedantic(expect_numeric_column, (ctx, data, pd.UInt8Dtype))


@mark.slow
def test_determine_int8_bigdata(ctx, benchmark):
    data = Series(list(str(i % 256 - 128) for i in range(BIGDATA_COUNT)), dtype=str)
    benchmark.pedantic(expect_numeric_column, (ctx, data, pd.Int8Dtype))


@mark.slow
def test_determine_float_bigdata(ctx, benchmark):
    data = Series(list(str(i * 10000 / BIGDATA_COUNT) for i in range(BIGDATA_COUNT)), dtype=str)
    benchmark.pedantic(expect_numeric_column, (ctx, data, pd.Float32Dtype))


START_OF_EPOCH = datetime(1970, 1, 1)


@fixture
def datetimes_bigdata():
    # Dates between 1970 and 2084
    return [START_OF_EPOCH + timedelta(hours=i % 1000000) for i in range(BIGDATA_COUNT)]


@mark.slow
def test_determine_datetime_iso_bigdata(ctx, datetimes_bigdata, benchmark):
    data = Series([i.isoformat() for i in datetimes_bigdata], dtype=str)
    col, info = benchmark.pedantic(expect_datetime_column, (ctx, data, None))
    assert info == DatetimeIsoColumnInfo()
    assert col[0] == START_OF_EPOCH


@mark.slow
def test_determine_datetime_de_bigdata(ctx, datetimes_bigdata, benchmark):
    data = Series([i.strftime("%d.%m.%Y %H:%M") for i in datetimes_bigdata], dtype=str)
    col, info = benchmark.pedantic(expect_datetime_column, (ctx, data, None))
    assert info == DatetimePatternColumnInfo(format="%d.%m.%Y %H:%M", kind=DatetimeKind.DATETIME)
    assert col[0] == START_OF_EPOCH


@mark.slow
def test_determine_datetime_us_bigdata(ctx, datetimes_bigdata, benchmark):
    data = Series([i.strftime("%m/%d/%Y %H:%M:%S") for i in datetimes_bigdata], dtype=str)
    col, info = benchmark.pedantic(expect_datetime_column, (ctx, data, None))
    assert info == DatetimePatternColumnInfo(format="%m/%d/%Y %H:%M:%S", kind=DatetimeKind.DATETIME)
    assert col[0] == START_OF_EPOCH


@mark.slow
def test_determine_datetime_unknown_bigdata(ctx, datetimes_bigdata, benchmark):
    data = Series([i.strftime("%Y:%m:%d %H:%M") for i in datetimes_bigdata], dtype=str)
    # col, info = parse_column(data)
    col, info = benchmark.pedantic(parse_column, (ctx, data, None))
    assert info == StringColumnInfo()
    assert col[0] == "1970:01:01 00:00"


# Test downcasting
@fixture
def uint8_string_series():
    return Series(["0", "127", "255"], dtype=str)


@fixture
def int8_string_series():
    return Series(["0", "127", "-128"], dtype=str)


@fixture
def float32_string_series():
    return Series(["0.0", "1.4e-8", "-1.0", "123.2e8"], dtype=str)


@fixture
def float32_german_string_series():
    return Series(["1,0", "-127,0", "128,3"], dtype=str)


@fixture
def complex_series():
    return Series([0.0, 1.4e-8, -1 + 2j, 123.2e8, None])


@fixture
def bool_series():
    return Series([False, True, float("nan"), None], dtype=object)


@fixture
def datetime_string_series():
    now = datetime.now()
    return Series(
        [
            str(now),
            str(now + timedelta(seconds=1)),
            str(now + timedelta(seconds=2)),
            str(now + timedelta(seconds=3)),
            str(now + timedelta(seconds=4)),
        ]
    )


def test_get_smallest_down_cast_able_type_uint8(ctx, uint8_string_series):
    expect_numeric_column(ctx, uint8_string_series, pd.UInt8Dtype)


def test_get_smallest_down_cast_able_type_int8(ctx, int8_string_series):
    expect_numeric_column(ctx, int8_string_series, pd.Int8Dtype)


def test_get_smallest_down_cast_able_type_bool(ctx, bool_series):
    # Boolean are converted to 0 (False) / 1 (True)
    col, col_info = expect_numeric_column(ctx, bool_series, pd.UInt8Dtype)
    assert col[0] == 0
    assert col[1] == 1
    assert pd.isna(col[2])
    assert pd.isna(col[3])


def test_get_smallest_down_cast_able_type_float32(ctx, float32_string_series):
    expect_numeric_column(ctx, float32_string_series, pd.Float32Dtype)


def test_get_smallest_down_cast_able_type_float32_german(ctx, float32_german_string_series):
    expect_numeric_column(ctx, float32_german_string_series, pd.Float32Dtype)


def test_get_smallest_down_cast_able_type_complex(ctx, complex_series):
    expect_numeric_column(ctx, complex_series, numpytype.Complex128DType)


def test_get_smallest_down_cast_able_type_datetime(ctx, datetime_string_series):
    expect_datetime_column(ctx, datetime_string_series, numpytype.DateTime64DType)


# Utils
def expect_numeric_column(ctx: TaskContext, col: list | Series, expected_dtype=None):
    parsed_col, col_info = parse_column(ctx, col, expected_dtype)
    assert isinstance(col_info, NumericColumnInfo)
    return parsed_col, col_info


def expect_datetime_column(ctx: TaskContext, col: list | Series, expected_dtype=None):
    parsed_col, col_info = parse_column(ctx, col, expected_dtype)
    assert isinstance(col_info, DatetimeColumnInfo)
    return parsed_col, col_info


def parse_column(ctx: TaskContext, col: list | Series, expected_dtype=None):
    COL_ID = "col"
    df = DataFrame(col, columns=[COL_ID])
    result = ctx.exec(parse_types, df)
    cols = result.all_cols
    assert len(cols.ids) == 1
    converted_col = cols.get_col(COL_ID)
    if expected_dtype is not None:
        assert isinstance(converted_col.dtype, expected_dtype)
    assert len(converted_col) == len(col)
    return converted_col, cols.get_info(COL_ID)
