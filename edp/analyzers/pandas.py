from datetime import timedelta
from logging import getLogger
from typing import AsyncIterator, Tuple

from numpy import any, count_nonzero
from numpy import max as numpy_max
from numpy import mean, median
from numpy import min as numpy_min
from numpy import std, unique
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    to_datetime,
    to_numeric,
    to_timedelta,
)

from edp.analyzers.base import Analyzer
from edp.types import (
    Column,
    DataSetType,
    DateTimeColumn,
    NumericColumn,
    StringColumn,
    StructuredEDPDataSet,
    TemporalConsistency,
)

DATE_TIME_FORMAT = "ISO8601"


class Pandas(Analyzer):
    def __init__(self, data: DataFrame):
        self._logger = getLogger(__name__)
        self._data = data

    @property
    def data_set_type(self):
        return DataSetType.structured

    async def analyze(self) -> StructuredEDPDataSet:
        columns = {name: column async for name, column in self._analyze_columns()}
        return StructuredEDPDataSet(rowCount=len(self._data.index), columns=columns)

    async def _analyze_columns(self) -> AsyncIterator[Tuple[str, Column]]:
        for column_name in self._data.columns:
            yield column_name, await self._analyze_column(self._data[column_name])

    async def _analyze_column(self, column: Series):
        column = infer_type_and_convert(column)
        type_char = column.dtype.kind
        if type_char in "iufcm":
            return await self._analyze_numeric_column(column)
        if type_char in "M":
            return await self._analyze_datetime_column(column)

        return await self._analyze_string_column(column)

    async def _analyze_numeric_column(self, column: Series) -> NumericColumn:
        return NumericColumn(
            min=numpy_min(column),
            max=numpy_max(column),
            mean=mean(column),
            median=median(column),
            stddev=std(column),
            dataType=str(column.dtype),
        )

    async def _analyze_datetime_column(self, column: Series) -> DateTimeColumn:
        INTERVALS = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

        return DateTimeColumn(
            earliest=numpy_min(column),
            latest=numpy_max(column),
            all_entries_are_unique=column.is_unique,
            monotonically_increasing=column.is_monotonic_increasing,
            monotonically_decreasing=column.is_monotonic_decreasing,
            temporalConsistencies=[compute_temporal_consistency(column, interval) for interval in INTERVALS],
            gaps={interval: compute_gaps(column, interval) for interval in INTERVALS},
        )

    async def _analyze_string_column(self, column: Series) -> StringColumn:
        return StringColumn()


def compute_temporal_consistency(column: Series, interval: timedelta) -> TemporalConsistency:
    column.index = DatetimeIndex(column)
    # TODO: Restrict to only the most abundant ones.
    abundances = unique(list(column.resample(interval).count()))
    different_abundances = len(abundances)
    return TemporalConsistency(
        interval=interval,
        stable=(different_abundances == 1),
        differentAbundancies=different_abundances,
        abundances=abundances.tolist(),
    )


def compute_gaps(column: Series, interval: timedelta) -> int:
    deltas = column.sort_values().diff()
    interval_timedelta = to_timedelta(interval)
    over_interval_size = deltas > interval_timedelta
    return count_nonzero(over_interval_size)


def infer_type_and_convert(column: Series) -> Series:
    type_character = column.dtype.kind

    if type_character in "i":
        if any(column < 0):
            return to_numeric(column, downcast="signed", errors="raise")
        return to_numeric(column, downcast="unsigned", errors="raise")

    if type_character in "u":
        return to_numeric(column, downcast="unsigned", errors="raise")

    if type_character in "f":
        return to_numeric(column, downcast="float", errors="raise")

    if type_character in "c":
        return column

    if type_character in "M":
        return to_datetime(column, errors="raise", format=DATE_TIME_FORMAT)

    if type_character in "m":
        return column

    try:
        return to_datetime(column, errors="raise", format=DATE_TIME_FORMAT)
    except (ValueError, TypeError):
        pass

    try:
        return infer_type_and_convert(to_numeric(column, errors="raise"))
    except (ValueError, TypeError):
        pass

    return column
