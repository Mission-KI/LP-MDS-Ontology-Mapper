from datetime import timedelta
from logging import getLogger
from typing import AsyncIterator, Tuple

from numpy import max as numpy_max
from numpy import mean, median
from numpy import min as numpy_min
from numpy import std, unique
from pandas import DataFrame, DatetimeIndex, Series, to_datetime, to_numeric

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
        type_character = column.dtype.kind
        if type_character in "iufc":
            return await self._analyze_numeric_column(column)
        elif type_character in "M":
            return await self._analyze_datetime_column(column)
        elif type_character in "m":
            return await self._analyze_numeric_column(column)

        try:
            datetime_converted = to_datetime(column, errors="raise", format=DATE_TIME_FORMAT)
            return await self._analyze_datetime_column(datetime_converted)
        except (ValueError, TypeError):
            pass

        try:
            numeric_converted = to_numeric(column, errors="raise")
            return await self._analyze_numeric_column(numeric_converted)
        except (ValueError, TypeError):
            pass
        return await self._analyze_string_column(column)

    async def _analyze_numeric_column(self, column: Series) -> NumericColumn:
        return NumericColumn(
            name=column.name,
            min=numpy_min(column),
            max=numpy_max(column),
            mean=mean(column),
            median=median(column),
            stddev=std(column),
            dataType=str(column.dtype),
        )

    async def _analyze_datetime_column(self, column: Series) -> DateTimeColumn:
        CONSISTENCY_INTERVALS = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

        return DateTimeColumn(
            name=column,
            earliest=numpy_min(column),
            latest=numpy_max(column),
            temporalConsistencies=[
                self._compute_temporal_consistency(column, interval) for interval in CONSISTENCY_INTERVALS
            ],
        )

    async def _analyze_string_column(self, column: Series) -> StringColumn:
        return StringColumn(name=column.name)

    def _compute_temporal_consistency(self, column: Series, interval: timedelta) -> TemporalConsistency:
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
