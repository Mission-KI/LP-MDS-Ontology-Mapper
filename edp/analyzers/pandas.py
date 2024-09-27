from datetime import timedelta
from logging import getLogger
from typing import AsyncIterator, List, Sequence

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


class Pandas(Analyzer):
    def __init__(self, data: DataFrame):
        self._logger = getLogger(__name__)
        self._data = data
        self._column_analyzers = {
            NumericColumn: self._analyze_numeric_column,
            DateTimeColumn: self._analyze_datetime_column,
            StringColumn: self._analyze_string_column,
        }

    @property
    def data_set_type(self):
        return DataSetType.structured

    async def analyze(self) -> StructuredEDPDataSet:
        columns = [column async for column in self._analyze_columns()]
        return StructuredEDPDataSet(rowCount=len(self._data.index), columns=columns)

    async def _analyze_columns(self) -> AsyncIterator[Column]:
        for column_name in self._data.columns:
            column_type = self._translate_type(column_name)
            if column_type is NumericColumn:
                yield await self._analyze_numeric_column(column_name)
            elif column_type is DateTimeColumn:
                yield await self._analyze_datetime_column(column_name)
            else:
                yield await self._analyze_string_column(column_name)

    def _translate_type(self, column_name: str):
        type_character = self._data[column_name].dtype.kind
        if type_character in "iufc":
            return NumericColumn
        elif type_character in "M":
            return DateTimeColumn
        elif type_character in "m":
            return NumericColumn
        elif type_character in "OSU":
            try:
                self._data[column_name] = to_datetime(self._data[column_name], errors="raise")
                return DateTimeColumn
            except (ValueError, TypeError):
                pass
            try:
                self._data[column_name] = to_numeric(self._data[column_name], errors="raise")
                return NumericColumn
            except (ValueError, TypeError):
                pass
        return StringColumn

    async def _analyze_numeric_column(self, name: str) -> NumericColumn:
        column = self._data[name]
        return NumericColumn(
            name=name,
            min=numpy_min(column),
            max=numpy_max(column),
            mean=mean(column),
            median=median(column),
            stddev=std(column),
            dataType=str(column.dtype),
        )

    async def _analyze_datetime_column(self, name: str) -> DateTimeColumn:
        CONSISTENCY_INTERVALS = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

        column = self._data[name]
        return DateTimeColumn(
            name=name,
            earliest=numpy_min(column),
            latest=numpy_max(column),
            temporalConsistencies=[
                self._compute_temporal_consistency(column, interval) for interval in CONSISTENCY_INTERVALS
            ],
        )

    async def _analyze_string_column(self, name: str) -> StringColumn:
        return StringColumn(name=name)

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
