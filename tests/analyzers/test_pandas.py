from datetime import datetime, timedelta

from pandas import Series
from pytest import fixture

from edp.analyzers.pandas import infer_type_and_convert


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
