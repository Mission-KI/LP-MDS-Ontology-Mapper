import warnings
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, Generic, Iterator, Optional, TypeVar, Union

from numpy import any as numpy_any
from pandas import DataFrame, Index, Series, StringDtype, to_datetime, to_numeric

_logger = getLogger(__name__)


class _BaseColumnInfo(ABC):
    @abstractmethod
    def cast(self, col: Series) -> Series: ...

    def count_valid_rows(self, col: Series) -> int:
        try:
            converted_col = self.cast(col)
            return converted_col.notna().sum()
        except Exception as e:
            _logger.warning("Type parsing error", exc_info=e)
            return 0


class DatetimeColumnInfo(_BaseColumnInfo):
    def __init__(self, format: str, utc: bool = False):
        self._format = format
        self._utc = utc

    @property
    def format(self):
        return self._format

    @property
    def utc(self):
        return self._utc

    def cast(self, col: Series) -> Series:
        return to_datetime(col, errors="coerce", format=self.format, utc=self.utc)


class NumericColumnInfo(_BaseColumnInfo):
    def __init__(self, decimal_separator: Optional[str] = None):
        self._decimal_separator = decimal_separator

    def cast(self, col: Series) -> Series:
        # Replace decimal separator with "." if needed.
        decimal_separator = self._decimal_separator
        if decimal_separator is not None:
            col = col.apply(lambda val: val.replace(decimal_separator, ".") if isinstance(val, str) else val)
        # Removes rows with incompatible type
        col_numeric = to_numeric(col, errors="coerce")
        # Narrow type to one supporting pd.NA (Int64 or Float64); doesn't work with complex numbers.
        if col_numeric.dtype.kind not in "c":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_numeric = col_numeric.convert_dtypes(
                    infer_objects=False,
                    convert_string=False,
                    convert_integer=True,
                    convert_boolean=False,
                    convert_floating=True,
                )
        # After that dtype should be Int64 or Float64. Try downcasting.
        if col_numeric.dtype.kind in "i" and numpy_any(col_numeric < 0):
            return to_numeric(col_numeric, downcast="signed", errors="coerce")
        elif col_numeric.dtype.kind in ("i", "u"):
            return to_numeric(col_numeric, downcast="unsigned", errors="coerce")
        elif col_numeric.dtype.kind in "f":
            return to_numeric(col_numeric, downcast="float", errors="coerce")
        # E.g. complex numbers
        return col_numeric


class StringColumnInfo(_BaseColumnInfo):
    def cast(self, col: Series) -> Series:
        return col.astype(StringDtype())


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


# First try datetime parsing (because "20240103" is an ISO date and a number):
# 1) Parse ISO datetimes with either missing or consistent time zone, e.g. "20450630T13:29:53".
#    This needs utc=False otherwise it would just assume UTC for missing time zone!
# 2) Parse ISO datetimes with mixed time zones in one column and convert them to UTC,
#    e.g. "20450630T13:29:53+0100" AND "20450630T13:29:53+0200".
#    This needs utc=True otherwise we get the warning below and resulting column has type "object"!
#    On the downside we lose the info about the original time zone and just get the UTC normalized datetime.
# 3) Try parsing with different local formats (no time zone).
# Then try parsing as a number ("." and "," as decimal separator).
ALLOWED_COLUMN_INFOS = (
    DatetimeColumnInfo(format="ISO8601", utc=False),
    DatetimeColumnInfo(format="ISO8601", utc=True),
    DatetimeColumnInfo(format="%d.%m.%Y %H:%M:%S"),
    DatetimeColumnInfo(format="%d.%m.%Y %H:%M"),
    DatetimeColumnInfo(format="%d.%m.%Y"),
    NumericColumnInfo(),
    NumericColumnInfo(","),
)
# String conversion as fallback because that is always possible.
FALLBACK_COLUMN_INFO = StringColumnInfo()
# Required minimum successful conversion ratio considering non-null values.
MIN_CONVERSION_RATIO = 0.5


async def parse_types(data: DataFrame) -> Result:
    column_infos = dict[str, _ColumnInfo]()
    results_data = DataFrame(index=data.index)

    for column_name, column in data.items():
        if not isinstance(column_name, str):
            raise RuntimeError(f"Column {column_name} needs a label.")
        converted_column, col_info = _determine_type(column)
        column_infos[column_name] = col_info
        results_data[column_name] = converted_column

    result = Result(results_data, column_infos)

    _logger.debug("Found Numeric columns: %s", result.numeric_cols.ids)
    _logger.debug("Found DateTime columns: %s", result.datetime_cols.ids)
    _logger.debug("Found String columns: %s", result.string_cols.ids)

    return result


def _determine_type(column: Series) -> tuple[Series, _ColumnInfo]:
    # remove NA values
    column = column[column.notna()]
    # match: converted_series, matched_colinfo, non_null_count
    matches = [match for match in _try_type_conversions(column)]
    if len(matches) > 0:
        # best match has highest non_null_count
        best_match = max(matches, key=lambda match: match[2])
        return best_match[0], best_match[1]
    else:
        return FALLBACK_COLUMN_INFO.cast(column), FALLBACK_COLUMN_INFO


def _try_type_conversions(column: Series) -> Iterator[tuple[Series, _ColumnInfo, int]]:
    count = len(column)
    if count > 0:
        for column_info in ALLOWED_COLUMN_INFOS:
            try:
                converted_column = column_info.cast(column)
                non_null_count = converted_column.notna().sum()
                # succesfully converted values must surpass min ratio
                if non_null_count >= MIN_CONVERSION_RATIO * count:
                    yield converted_column, column_info, non_null_count
            except Exception as e:
                _logger.debug("Can't convert to %s", column_info, exc_info=e)
