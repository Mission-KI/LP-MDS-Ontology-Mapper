import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar, Union

from numpy import any as numpy_any
from pandas import DataFrame, Index, Series, StringDtype, to_datetime, to_numeric
from pydantic.dataclasses import dataclass

_logger = getLogger(__name__)


class _BaseColumnInfo(ABC):
    @abstractmethod
    def cast(self, col: Series) -> Series:
        """This shall convert the column to the destination type and return NA for entries it cannot parse."""

    def count_valid_rows(self, col: Series) -> int:
        """This counts how many entries in the column can be converted to the destination type."""
        converted_col = self.cast(col)
        return converted_col.notna().sum()


@dataclass(frozen=True)
class DatetimeColumnInfo(_BaseColumnInfo):
    @property
    @abstractmethod
    def format(self) -> str: ...

    @abstractmethod
    def _convert_entry(self, element: Any) -> None | datetime: ...

    def cast(self, col: Series) -> Series:
        col_converted = col.apply(self._convert_entry)
        col_valid = col_converted[col_converted.notna()]
        count_with_timezone = col_valid.apply(lambda timestamp: timestamp.tzinfo is not None).sum()
        count_without_timezone = len(col_valid.index) - count_with_timezone

        if count_with_timezone == 0:
            # If there are no entries with timezone we return them as dtype "datetime64[ns]" without TZ info.
            return to_datetime(col_converted, errors="coerce", utc=False)
        else:
            if count_without_timezone > 0:
                _logger.warning(
                    "Found datetime column '%s' which contains entries partially with and without timezone. For entries without timezone UTC is assumed.",
                    col.name,
                )
            # If there is at least one entry with a timezone we convert them to dtype "datetime64[ns, UTC]" with TZ=UTC.
            return to_datetime(col_converted, errors="coerce", utc=True)


@dataclass(frozen=True)
class DatetimeNativeColumnInfo(DatetimeColumnInfo):
    @property
    def format(self) -> str:
        return "NATIVE"

    def _convert_entry(self, element: Any) -> None | datetime:
        if isinstance(element, datetime):
            return element
        else:
            return None


@dataclass(frozen=True)
class DatetimeIsoColumnInfo(DatetimeColumnInfo):
    @property
    def format(self) -> str:
        return "ISO8601"

    def _convert_entry(self, element: Any) -> None | datetime:
        try:
            return datetime.fromisoformat(element)
        except Exception:
            return None


@dataclass(frozen=True)
class DatetimePatternColumnInfo(DatetimeColumnInfo):
    pattern: str

    @property
    def format(self) -> str:
        return self.pattern

    def _convert_entry(self, element: Any) -> None | datetime:
        try:
            return datetime.strptime(element, self.format)
        except Exception:
            return None


@dataclass(frozen=True)
class NumericColumnInfo(_BaseColumnInfo):
    def count_valid_rows(self, col: Series) -> int:
        col_numeric = to_numeric(self._preprocess(col), errors="coerce")
        return col_numeric.notna().sum()

    def cast(self, col: Series) -> Series:
        # Convert to numeric type and set incompatible values to float.nan.
        col_numeric = to_numeric(self._preprocess(col), errors="coerce")
        # Narrow type to one supporting pd.NA (Int64 or Float64).
        col_numeric_nullable = self._cast_to_pandas_type(col_numeric)
        # Downcast to most specific type (e.g. UInt8, Int16, Float32 etc.)
        col_result = self._downcast(col_numeric_nullable)
        _logger.debug("Casting column '%s' to numeric dtype %s", col.name, col_result.dtype)
        return col_result

    def _preprocess(self, col: Series) -> Series:
        # Try replacing decimal separator "," with ".".
        return col.apply(lambda val: val.replace(",", ".") if isinstance(val, str) else val)

    def _cast_to_pandas_type(self, col: Series) -> Series:
        # There is no Pandas type for complex numbers. This is okay as Numpy's complex NA.
        if col.dtype.kind in "c":
            return col
        # Convert to Pandas types because they support NA (e.g. Int8).
        # For Numpy NA is represented as float.nan, so downcasting is not possible!
        with warnings.catch_warnings():
            # convert_dtypes() raises a warning when trying to convert certain float values:
            # "RuntimeWarning: invalid value encountered in cast"
            # at pandas\core\dtypes\cast.py", line 1056, in convert_dtypes
            # This is not a problem because it checks if the converted values are equal to the original ones.
            warnings.simplefilter("ignore", RuntimeWarning)
            return col.convert_dtypes(
                infer_objects=False,
                convert_string=False,
                convert_integer=True,
                convert_boolean=False,
                convert_floating=True,
            )

    def _downcast(self, col: Series) -> Series:
        # After that dtype should be Int64 or Float64. Try downcasting.
        if col.dtype.kind in "f":
            return to_numeric(col, downcast="float", errors="coerce")
        if col.dtype.kind in "i" and numpy_any(col < 0):
            return to_numeric(col, downcast="signed", errors="coerce")
        if col.dtype.kind in ("i", "u"):
            return to_numeric(col, downcast="unsigned", errors="coerce")
        # E.g. complex numbers
        return col


@dataclass(frozen=True)
class StringColumnInfo(_BaseColumnInfo):
    def cast(self, col: Series) -> Series:
        return col.astype(StringDtype())


ColumnInfo = Union[DatetimeColumnInfo, NumericColumnInfo, StringColumnInfo]


TColumnInfo = TypeVar("TColumnInfo", bound=ColumnInfo)


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
    def __init__(self, data: DataFrame, infos: Dict[str, ColumnInfo]) -> None:
        self._data = data
        self._infos: Dict[str, ColumnInfo] = infos

    def set_index(self, index: Index):
        self._data.set_index(index, inplace=True)

    @property
    def all_cols(self) -> ColumnsWrapper[ColumnInfo]:
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
# 1) Parse ISO datetimes with or without timezone, e.g. "20450630T13:29:53" or "20450630T13:29:53+0100".
#    If at least one entry has a TZ all are converted to dtype "datetime64[ns, UTC]" with TZ=UTC. For entries without timezone UTC is assumed.
#    If none has a timezone they are returned as dtype "datetime64[ns]" without timezone info.
# 2) Try parsing with different local formats (no time zone for now).
# Then try parsing as a number ("." and "," as decimal separator).
ALLOWED_COLUMN_INFOS = (
    # Column already has datetime type (e.g. from Excel import)
    DatetimeNativeColumnInfo(),
    # ISO datetime
    DatetimeIsoColumnInfo(),
    # DE datetime
    DatetimePatternColumnInfo(pattern="%d.%m.%Y %H:%M:%S"),
    DatetimePatternColumnInfo(pattern="%d.%m.%Y %H:%M"),
    DatetimePatternColumnInfo(pattern="%d.%m.%Y"),
    # US datetime
    DatetimePatternColumnInfo(pattern="%m/%d/%Y %H:%M:%S"),
    DatetimePatternColumnInfo(pattern="%m-%d-%Y %H:%M:%S"),
    DatetimePatternColumnInfo(pattern="%m/%d/%Y %H:%M"),
    DatetimePatternColumnInfo(pattern="%m-%d-%Y %H:%M"),
    DatetimePatternColumnInfo(pattern="%m/%d/%Y"),
    DatetimePatternColumnInfo(pattern="%m-%d-%Y"),
    # Numeric
    NumericColumnInfo(),
)
# String conversion as fallback because that is always possible.
FALLBACK_COLUMN_INFO = StringColumnInfo()
# Required minimum successful conversion ratio considering non-null values.
MIN_CONVERSION_RATIO = 0.5
# Conversion is first tried with a sample
SAMPLE_SIZE = 1000


def parse_types(data: DataFrame) -> Result:
    column_infos = dict[str, ColumnInfo]()
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


def _determine_type(column: Series) -> tuple[Series, ColumnInfo]:
    column = column[column.notna()]
    column_sample = column.sample(SAMPLE_SIZE) if len(column.index) > SAMPLE_SIZE else column
    # determine best type match based on the sample
    preferred_col_info = _determine_best_type_match(column_sample)
    return _do_type_conversion(column, preferred_col_info)


def _determine_best_type_match(column: Series) -> Optional[ColumnInfo]:
    # match: matched_colinfo, valid_count
    matches = [match for match in _try_type_conversions(column)]
    if len(matches) > 0:
        # best match is the one with highest valid_count
        best_match = max(matches, key=lambda match: match[1])
        return best_match[0]
    else:
        return None


def _try_type_conversions(column: Series) -> Iterator[tuple[ColumnInfo, int]]:
    # passed columns don't contain NA values anymore
    count = len(column.index)
    if count == 0:
        return

    for column_info in ALLOWED_COLUMN_INFOS:
        valid_count = column_info.count_valid_rows(column)
        _logger.debug(
            "Trying to convert column '%s' with %s, found %d valid rows out of %d.",
            column.name,
            column_info,
            valid_count,
            count,
        )
        # succesfully converted values must surpass min ratio
        if _check_conversion_ratio_passed(valid_count, count):
            yield column_info, valid_count


def _do_type_conversion(column: Series, column_info: Optional[ColumnInfo]) -> tuple[Series, ColumnInfo]:
    if column_info is not None:
        count = len(column.index)
        # convert the whole column based on the preferred column info
        converted_col = column_info.cast(column)
        count_na = converted_col.isna().sum()
        if count_na > 0:
            _logger.warning(
                "Couldn't convert some entries of column '%s' using %s (%d invalid of %d non-null entries).",
                column.name,
                column_info,
                count_na,
                count,
            )
        else:
            _logger.debug(
                "Converting all %d non-null entries of column '%s' using %s.", count, column.name, column_info
            )
        return converted_col, column_info

    # string column as fallback
    _logger.debug("Converting column '%s' using FALLBACK %s.", column.name, FALLBACK_COLUMN_INFO)
    return FALLBACK_COLUMN_INFO.cast(column), FALLBACK_COLUMN_INFO


def _check_conversion_ratio_passed(valid_count, count):
    return valid_count >= MIN_CONVERSION_RATIO * count
