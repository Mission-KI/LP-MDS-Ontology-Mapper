import warnings
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Generic, TypeVar

from numpy import any as numpy_any
from pandas import DataFrame, Series, to_datetime, to_numeric
from pydantic import BaseModel

# Try these dateformats one after the other: ISO and a couple of usual German formats.
DATE_TIME_FORMAT_ISO = "ISO8601"
DATE_TIME_OTHER_FORMATS = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y"]


class ColumnInfo(BaseModel):
    dtype_str: str
    dtype_kind: str


class DatetimeColumnInfo(ColumnInfo):
    format: str


class NumericColumnInfo(ColumnInfo):
    pass


class StringColumnInfo(ColumnInfo):
    pass


COL_INFO = TypeVar("COL_INFO", bound=ColumnInfo)


@dataclass(frozen=True)
class ColumnWithInfo(Generic[COL_INFO]):
    id: str
    info: COL_INFO
    data: Series


@dataclass(frozen=True)
class ColumnsWithInfo(Generic[COL_INFO]):
    infos: dict[str, COL_INFO]
    data: DataFrame

    @property
    def ids(self) -> list[str]:
        return list(self.infos.keys())

    def __getitem__(self, id: str):
        return ColumnWithInfo(id, self.infos[id], self.data[id])


class TypeParser:
    def __init__(self, data: DataFrame):
        self._data = data
        self._processed_data = DataFrame(index=data.index)
        self._column_infos: Dict[str, ColumnInfo] = dict()
        self._logger = getLogger(__name__)

    @property
    def all_cols(self) -> ColumnsWithInfo[ColumnInfo]:
        return ColumnsWithInfo(self._column_infos, self._processed_data)

    @property
    def datetime_cols(self) -> ColumnsWithInfo[DatetimeColumnInfo]:
        return self._build_filtered_cols(DatetimeColumnInfo)

    @property
    def numeric_cols(self) -> ColumnsWithInfo[NumericColumnInfo]:
        return self._build_filtered_cols(NumericColumnInfo)

    @property
    def string_cols(self) -> ColumnsWithInfo[StringColumnInfo]:
        return self._build_filtered_cols(StringColumnInfo)

    def _build_filtered_cols(self, info_type: type[COL_INFO]) -> ColumnsWithInfo[COL_INFO]:
        infos = {key: value for key, value in self._column_infos.items() if isinstance(value, info_type)}
        data = self._processed_data.loc[:, list(infos.keys())]
        return ColumnsWithInfo(infos, data)

    async def process(self):
        for column_name in self._data.columns:
            column, column_info = TypeParser._determine_type(self._data[column_name])
            self._column_infos[column_name] = column_info
            self._processed_data[column_name] = column

        self._logger.debug("Found Numeric columns: %s", self.numeric_cols.ids)
        self._logger.debug("Found DateTime columns: %s", self.datetime_cols.ids)
        self._logger.debug("Found String columns: %s", self.string_cols.ids)

    @staticmethod
    def _determine_type(column: Series) -> tuple[Series, ColumnInfo]:
        type_character = column.dtype.kind

        if type_character in "i":
            if numpy_any(column < 0):
                return TypeParser._numeric_column(to_numeric(column, downcast="signed", errors="raise"))
            else:
                return TypeParser._numeric_column(to_numeric(column, downcast="unsigned", errors="raise"))

        elif type_character in "u":
            return TypeParser._numeric_column(to_numeric(column, downcast="unsigned", errors="raise"))

        elif type_character in "f":
            return TypeParser._numeric_column(to_numeric(column, downcast="float", errors="raise"))

        elif type_character in "c":
            return TypeParser._numeric_column(column)

        elif type_character in "M":
            return TypeParser._try_convert_datetime(column)

        elif type_character in "m":
            return TypeParser._numeric_column(column)

        try:
            return TypeParser._try_convert_datetime(column)
        except TypeError:
            pass

        try:
            normalized_column = column.apply(lambda val: val.replace(",", ".") if isinstance(val, str) else val)
            numeric_column = to_numeric(normalized_column, errors="raise")
            return TypeParser._determine_type(numeric_column)
        except (ValueError, TypeError):
            pass

        return TypeParser._string_column(column)

    @staticmethod
    def _try_convert_datetime(column: Series) -> tuple[Series, DatetimeColumnInfo]:
        # Parse ISO datetimes with either missing or consistent time zone, e.g. "20450630T13:29:53".
        # This needs utc=False otherwise it would just assume UTC for missing time zone!
        try:
            return TypeParser._try_convert_datetime_format(column, DATE_TIME_FORMAT_ISO, False)
        except TypeError:
            pass

        # Parse ISO datetimes with mixed time zones in one column and convert them to UTC,
        # e.g. "20450630T13:29:53+0100" AND "20450630T13:29:53+0200".
        # This needs utc=True otherwise we get the warning below and resulting column has type "object"!
        # On the downside we lose the info about the original time zone and just get the UTC normalized datetime.
        try:
            return TypeParser._try_convert_datetime_format(column, DATE_TIME_FORMAT_ISO, True)
        except TypeError:
            pass

        # Try parsing with different local formats (no time zone).
        for format in DATE_TIME_OTHER_FORMATS:
            try:
                return TypeParser._try_convert_datetime_format(column, format, False)
            except TypeError:
                pass

        raise TypeError

    @staticmethod
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
            return TypeParser._datetime_column(result_column, format)

    @staticmethod
    def _numeric_column(column: Series) -> tuple[Series, NumericColumnInfo]:
        return column, NumericColumnInfo(dtype_str=str(column.dtype), dtype_kind=column.dtype.kind)

    @staticmethod
    def _string_column(column: Series) -> tuple[Series, StringColumnInfo]:
        return column, StringColumnInfo(dtype_str=str(column.dtype), dtype_kind=column.dtype.kind)

    @staticmethod
    def _datetime_column(column: Series, format: str) -> tuple[Series, DatetimeColumnInfo]:
        return column, DatetimeColumnInfo(dtype_str=str(column.dtype), dtype_kind=column.dtype.kind, format=format)
