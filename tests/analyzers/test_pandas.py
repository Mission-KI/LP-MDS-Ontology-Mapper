from datetime import datetime, timedelta
from logging import getLogger
from math import isnan

from pandas import DataFrame, Series
from pytest import fixture, mark

from edp.analyzers.pandas import _COMMON_UNIQUE, TypeParser, _get_correlation_graph


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
    return Series(["1,0", "-127,0", "128,0"], dtype=str)


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


async def test_get_smallest_down_cast_able_type_uint8(uint8_string_series):
    await expect_numeric_column(uint8_string_series, "uint8")


async def test_get_smallest_down_cast_able_type_int8(int8_string_series):
    await expect_numeric_column(int8_string_series, "int8")


async def test_get_smallest_down_cast_able_type_bool(bool_series):
    # Boolean are converted to 0.0 (False) / 1.0 (True)
    col = await expect_numeric_column(bool_series, "float32")
    assert col[0] == 0.0
    assert col[1] == 1.0
    assert isnan(col[2])
    assert isnan(col[3])


async def test_get_smallest_down_cast_able_type_float32(float32_string_series):
    await expect_numeric_column(float32_string_series, "float32")


async def test_get_smallest_down_cast_able_type_float32_german(
    float32_german_string_series,
):
    await expect_numeric_column(float32_german_string_series, "float32")


async def test_get_smallest_down_cast_able_type_datetime(datetime_string_series):
    await expect_datetime_column(datetime_string_series, "datetime64[ns]")


@mark.asyncio
async def test_correlation(output_context):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({_COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    await _get_correlation_graph(getLogger("Test"), "test_correlation_graph", columns, fields, output_context)


async def expect_numeric_column(series: Series, dtype_str: str):
    parser = TypeParser(DataFrame(series))
    await parser.process()
    cols = parser.numeric_cols
    col_infos = list(cols.infos.values())
    assert len(col_infos) == 1
    assert col_infos[0].dtype_str == dtype_str
    return cols.data[0]


async def expect_datetime_column(series: Series, dtype_str: str):
    parser = TypeParser(DataFrame(series))
    await parser.process()
    cols = parser.datetime_cols
    col_infos = list(cols.infos.values())
    assert len(col_infos) == 1
    assert col_infos[0].dtype_str == dtype_str
    return cols.data[0]
