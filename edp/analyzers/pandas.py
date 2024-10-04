from datetime import timedelta
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Tuple

from numpy import any as numpy_any
from numpy import count_nonzero
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
from seaborn import boxplot

from edp.analyzers.base import Analyzer
from edp.context import OutputContext
from edp.file import File
from edp.types import (
    Column,
    DataSetType,
    DateTimeColumn,
    FileReference,
    NumericColumn,
    StringColumn,
    StructuredEDPDataSet,
    TemporalConsistency,
)

DATE_TIME_FORMAT = "ISO8601"


class Pandas(Analyzer):
    def __init__(self, data: DataFrame, file: File):
        self._logger = getLogger(__name__)
        self._data = data
        self._file = file

    @property
    def data_set_type(self):
        return DataSetType.structured

    async def analyze(self, output_context: OutputContext) -> StructuredEDPDataSet:
        row_count = len(self._data.index)
        self._logger.info("Started structured data analysis with dataset containing %d rows", row_count)
        columns = {name: column async for name, column in self._analyze_columns(output_context)}
        return StructuredEDPDataSet(rowCount=row_count, columns=columns)

    async def _analyze_columns(self, output_context: OutputContext) -> AsyncIterator[Tuple[str, Column]]:
        for column_name in self._data.columns:
            yield column_name, await self._analyze_column(self._data[column_name], output_context)

    async def _analyze_column(self, column: Series, output_context: OutputContext):
        column = infer_type_and_convert(column)
        type_char = column.dtype.kind
        if type_char in "iufcm":
            return await self._analyze_numeric_column(column, output_context)
        if type_char in "M":
            return await self._analyze_datetime_column(column)

        return await self._analyze_string_column(column)

    async def _analyze_numeric_column(self, column: Series, output_context: OutputContext) -> NumericColumn:
        self._logger.debug('Analyzing column "%s" as numeric', column.name)
        upper_percentile, lower_percentile, percentile_outliers = compute_percentiles(column)
        upper_z_score, lower_z_score, z_outliers = compute_standard_score(column)
        upper_quantile, lower_quantile, iqr, upper_iqr_limit, lower_iqr_limit, iqr_outliers = (
            compute_inter_quartile_range(column)
        )
        images = [await self._generate_box_plot(column, output_context)]
        return NumericColumn(
            null_entries=number_null_entries(column),
            images=images,
            min=numpy_min(column),
            max=numpy_max(column),
            mean=mean(column),
            median=median(column),
            stddev=std(column),
            upperPercentile=upper_percentile,
            lowerPercentile=lower_percentile,
            upperQuantile=upper_quantile,
            lowerQuantile=lower_quantile,
            percentileOutlierCount=percentile_outliers,
            upperZScore=upper_z_score,
            lowerZScore=lower_z_score,
            zScoreOutlierCount=z_outliers,
            upperIQR=upper_iqr_limit,
            lowerIQR=lower_iqr_limit,
            iqr=iqr,
            iqrOutlierCount=iqr_outliers,
            dataType=str(column.dtype),
        )

    async def _analyze_datetime_column(self, column: Series) -> DateTimeColumn:
        self._logger.debug('Analyzing column "%s" as datetime', column.name)

        INTERVALS = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

        deltas = column.sort_values().diff()

        return DateTimeColumn(
            null_entries=number_null_entries(column),
            earliest=numpy_min(column),
            latest=numpy_max(column),
            all_entries_are_unique=column.is_unique,
            monotonically_increasing=column.is_monotonic_increasing,
            monotonically_decreasing=column.is_monotonic_decreasing,
            temporalConsistencies=[compute_temporal_consistency(column, interval) for interval in INTERVALS],
            gaps={interval: compute_gaps(deltas, interval) for interval in INTERVALS},
        )

    async def _analyze_string_column(self, column: Series) -> StringColumn:
        self._logger.debug('Analyzing column "%s" as string', column.name)
        return StringColumn(null_entries=number_null_entries(column))

    async def _generate_box_plot(self, column: Series, output_context: OutputContext) -> FileReference:
        plot_name = self._file.output_reference + "_" + str(column.name) + "_box_plot"
        async with output_context.get_plot(plot_name) as (axes, reference):
            boxplot(
                column,
                notch=True,
                showcaps=True,
                width=0.3,
                flierprops={"marker": "x"},
                boxprops={"facecolor": (0.3, 0.5, 0.7, 0.5)},
                medianprops={"color": "r", "linewidth": 2},
                ax=axes,
            )
        return reference


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


def compute_gaps(deltas: Series, interval: timedelta) -> int:
    interval_timedelta = to_timedelta(interval)
    over_interval_size = deltas > interval_timedelta
    return count_nonzero(over_interval_size)


def number_null_entries(column: Series) -> int:
    return column.isnull().sum()


def _get_outliers(column: Series, lower_limit: float, upper_limit: float) -> int:
    is_outlier = (column < lower_limit) | (column > upper_limit)
    return count_nonzero(is_outlier)


def compute_percentiles(column: Series) -> Tuple[float, float, int]:
    upper_percentile = column.quantile(0.99)
    lower_percentile = column.quantile(0.01)
    return upper_percentile, lower_percentile, _get_outliers(column, lower_percentile, upper_percentile)


def compute_standard_score(column: Series) -> Tuple[float, float, int]:
    column_mean = column.mean()
    column_std = column.std()
    upper_z = column_mean + 3.0 * column_std
    lower_z = column_mean - 3.0 * column_std
    return upper_z, lower_z, _get_outliers(column, lower_z, upper_z)


def compute_inter_quartile_range(column: Series) -> Tuple[float, float, float, float, float, int]:
    upper_quantile = column.quantile(0.75)
    lower_quantile = column.quantile(0.25)
    inter_quartile_range = upper_quantile - lower_quantile
    upper_iqr_limit = upper_quantile + 1.5 * inter_quartile_range
    lower_iqr_limit = lower_quantile - 1.5 * inter_quartile_range
    return (
        upper_quantile,
        lower_quantile,
        inter_quartile_range,
        upper_iqr_limit,
        lower_iqr_limit,
        _get_outliers(column, lower_iqr_limit, upper_iqr_limit),
    )


def infer_type_and_convert(column: Series) -> Series:
    type_character = column.dtype.kind

    if type_character in "i":
        if numpy_any(column < 0):
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
