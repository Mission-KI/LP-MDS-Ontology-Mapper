from datetime import datetime, timedelta
from logging import getLogger

from pandas import DataFrame, Series
from pytest import fixture, mark

from edp.analyzers.pandas import (
    _COMMON_UNIQUE,
    _get_correlation_graph,
    infer_type_and_convert,
)


@fixture
def uint8_string_series():
    return Series(["0", "127", "255"], dtype=str)


@fixture
def int8_string_series():
    return Series(["0", "127", "-128"], dtype=str)


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


def test_get_smallest_down_cast_able_type_uint8(uint8_string_series):
    assert str(infer_type_and_convert(uint8_string_series).dtype) == "uint8"


def test_get_smallest_down_cast_able_type_int8(int8_string_series):
    assert str(infer_type_and_convert(int8_string_series).dtype) == "int8"


def test_get_smallest_down_cast_able_type_datetime(datetime_string_series):
    assert str(infer_type_and_convert(datetime_string_series).dtype) == "datetime64[ns]"


@mark.asyncio
async def test_correlation(output_context):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({_COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    await _get_correlation_graph(getLogger("Test"), "test_correlation_graph", columns, fields, output_context)
