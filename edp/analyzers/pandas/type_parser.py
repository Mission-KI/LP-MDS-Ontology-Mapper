import warnings
from logging import getLogger
from typing import Dict, Generic, TypeVar, Union

from numpy import any as numpy_any
from pandas import DataFrame, Index, Series, to_datetime, to_numeric
from pydantic.dataclasses import dataclass

# Try these dateformats one after the other: ISO and a couple of usual German formats.
DATE_TIME_FORMAT_ISO = "ISO8601"
DATE_TIME_OTHER_FORMATS = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y"]


_logger = getLogger(__name__)


@dataclass
class DatetimeColumnInfo:
    format: str


@dataclass
class NumericColumnInfo:
    pass


@dataclass
class StringColumnInfo:
    pass


_ColumnInfo = Union[DatetimeColumnInfo, NumericColumnInfo, StringColumnInfo]


TColumnInfo = TypeVar("TColumnInfo", bound=_ColumnInfo)


class ColumnsWrapper(Generic[TColumnInfo]):
    def __init__(self, all_data: DataFrame, infos: dict[str, TColumnInfo]):
        self._all_data = all_data
        self._infos = infos

    @property
    def ids(self) -> list[str]:
        return list(self._infos.keys())

    def get_info(self, id: str) -> TColumnInfo:
        return self._infos[id]

    @property
    def data(self) -> DataFrame:
        return self._all_data.loc[:, self.ids]

    def get_col(self, id: str) -> Series:
        return self._all_data.loc[:, id]


class Result:
    def __init__(self, data: DataFrame, infos: Dict[str, _ColumnInfo]) -> None:
        self._data = data
        self._infos: Dict[str, _ColumnInfo] = infos

    def set_index(self, index: Index):
        self._data.set_index(index, inplace=True)

    @property
    def all_cols(self) -> ColumnsWrapper[_ColumnInfo]:
        return ColumnsWrapper(self._data, self._infos)

    @property
    def datetime_cols(self) -> ColumnsWrapper[DatetimeColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, DatetimeColumnInfo)}
        return ColumnsWrapper(self._data, infos)

    @property
    def numeric_cols(self) -> ColumnsWrapper[NumericColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, NumericColumnInfo)}
        return ColumnsWrapper(self._data, infos)

    @property
    def string_cols(self) -> ColumnsWrapper[StringColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, StringColumnInfo)}
        return ColumnsWrapper(self._data, infos)


async def parse_types(data: DataFrame) -> Result:
    column_infos = dict[str, _ColumnInfo]()
    results_data = DataFrame(index=data.index)

    for column_name, column in data.items():
        if not isinstance(column_name, str):
            raise RuntimeError(f"Column {column_name} needs a label.")
        inferred_type_data, info = _determine_type(column)
        column_infos[column_name] = info
        results_data[column_name] = inferred_type_data

    result = Result(results_data, column_infos)

    _logger.debug("Found Numeric columns: %s", result.numeric_cols.ids)
    _logger.debug("Found DateTime columns: %s", result.datetime_cols.ids)
    _logger.debug("Found String columns: %s", result.string_cols.ids)

    return result


def _determine_type(column: Series) -> tuple[Series, _ColumnInfo]:
    type_character = column.dtype.kind

    if type_character in "i":
        if numpy_any(column < 0):
            return _numeric_column(to_numeric(column, downcast="signed", errors="raise"))
        else:
            return _numeric_column(to_numeric(column, downcast="unsigned", errors="raise"))

    elif type_character in "u":
        return _numeric_column(to_numeric(column, downcast="unsigned", errors="raise"))

    elif type_character in "f":
        return _numeric_column(to_numeric(column, downcast="float", errors="raise"))

    elif type_character in "c":
        return _numeric_column(column)

    elif type_character in "M":
        return _try_convert_datetime(column)

    elif type_character in "m":
        return _numeric_column(column)

    try:
        return _try_convert_datetime(column)
    except TypeError:
        pass

    try:
        normalized_column = column.apply(lambda val: val.replace(",", ".") if isinstance(val, str) else val)
        numeric_column = to_numeric(normalized_column, errors="raise")
        return _determine_type(numeric_column)
    except (ValueError, TypeError):
        pass

    return _string_column(column)


def _try_convert_datetime(column: Series) -> tuple[Series, DatetimeColumnInfo]:
    # Parse ISO datetimes with either missing or consistent time zone, e.g. "20450630T13:29:53".
    # This needs utc=False otherwise it would just assume UTC for missing time zone!
    try:
        return _try_convert_datetime_format(column, DATE_TIME_FORMAT_ISO, False)
    except TypeError:
        pass

    # Parse ISO datetimes with mixed time zones in one column and convert them to UTC,
    # e.g. "20450630T13:29:53+0100" AND "20450630T13:29:53+0200".
    # This needs utc=True otherwise we get the warning below and resulting column has type "object"!
    # On the downside we lose the info about the original time zone and just get the UTC normalized datetime.
    try:
        return _try_convert_datetime_format(column, DATE_TIME_FORMAT_ISO, True)
    except TypeError:
        pass

    # Try parsing with different local formats (no time zone).
    for format in DATE_TIME_OTHER_FORMATS:
        try:
            return _try_convert_datetime_format(column, format, False)
        except TypeError:
            pass

    raise TypeError


def _try_convert_datetime_format(column: Series, format: str, utc: bool) -> tuple[Series, DatetimeColumnInfo]:
    with warnings.catch_warnings():
        # If we try parsing datetimes with mixed time zones, we get a FutureWarning:
        # FutureWarning: In a future version of pandas, parsing datetimes with mixed time zones will raise an error
        # unless `utc=True`. Please specify `utc=True` to opt in to the new behaviour and silence this warning. To
        # create a `Series` with mixed offsets and `object` dtype, please use `apply` and `datetime.datetime.strptime`
        warnings.simplefilter("ignore", FutureWarning)

        try:
            result_column = to_datetime(column, errors="raise", format=format, utc=utc)
        except (ValueError, TypeError):
            raise TypeError
        if result_column.dtype.kind != "M":
            raise TypeError
        return _datetime_column(result_column, format)


def _numeric_column(column: Series) -> tuple[Series, NumericColumnInfo]:
    return column, NumericColumnInfo()


def _string_column(column: Series) -> tuple[Series, StringColumnInfo]:
    return column, StringColumnInfo()


def _datetime_column(column: Series, format: str) -> tuple[Series, DatetimeColumnInfo]:
    return column, DatetimeColumnInfo(format=format)
