from asyncio import get_running_loop
from collections.abc import Hashable
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import Enum
from logging import Logger, getLogger
from multiprocessing import cpu_count
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple
from warnings import warn

from fitter import Fitter, get_common_distributions
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from numpy import any as numpy_any
from numpy import corrcoef, count_nonzero, linspace, ones_like, triu
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    UInt64Dtype,
    concat,
    to_datetime,
    to_numeric,
)
from pydantic import BaseModel, Field
from scipy.stats import distributions
from seaborn import heatmap
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

from edp.analyzers.base import Analyzer
from edp.context import OutputContext
from edp.file import File
from edp.types import (
    DataSetType,
    DateTimeColumn,
    FileReference,
    NumericColumn,
    StringColumn,
    StructuredDataSet,
    TemporalConsistency,
    TemporalCover,
    TimeBasedGraph,
)

# Try these dateformats one after the other: ISO and a couple of usual German formats.
DATE_TIME_FORMATS = ["ISO8601", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y", "%H:%M:%S", "%H:%M"]


class FittingConfig(BaseModel):
    timeout: timedelta = Field(default=timedelta(seconds=30), description="Timeout to use for the fitting")
    error_function: str = Field(
        default="sumsquare_error", description="Error function to use to measure performance of the fits"
    )
    bins: int = Field(default=100, description="Number of bins to use for the fitting")
    distributions: List[str] = Field(
        default_factory=get_common_distributions, description="Distributions to try to fit"
    )


class _ColumnType(str, Enum):
    Numeric = "Numeric"
    String = "String"
    DateTime = "DateTime"


# Labels for fields

_COMMON_NON_NULL = "non-null-count"
_COMMON_NULL = "null-count"
_COMMON_UNIQUE = "unique-count"

_NUMERIC_MIN = "minimum"
_NUMERIC_MAX = "maximum"
_NUMERIC_LOWER_PERCENTILE = "lower-percentile"
_NUMERIC_UPPER_PERCENTILE = "upper-percentile"
_NUMERIC_PERCENTILE_OUTLIERS = "percentile-outliers"
_NUMERIC_MEAN = "mean"
_NUMERIC_MEDIAN = "median"
_NUMERIC_STD_DEV = "std-dev"
_NUMERIC_LOWER_Z = "lower-z-limit"
_NUMERIC_UPPER_Z = "upper-z-limit"
_NUMERIC_Z_OUTLIERS = "z-outlier-count"
_NUMERIC_LOWER_DIST = "lower-distribution-limit"
_NUMERIC_UPPER_DIST = "upper-distribution-limit"
_NUMERIC_LOWER_QUANT = "lower-quantile-limit"
_NUMERIC_UPPER_QUANT = "upper-quantile-limit"
_NUMERIC_LOWER_IQR = "lower-iqr-limit"
_NUMERIC_UPPER_IQR = "upper-iqr-limit"
_NUMERIC_IQR = "inter-quartile-range"
_NUMERIC_IQR_OUTLIERS = "iqr-outlier-count"
_NUMERIC_DISTRIBUTION = "distribution"
_NUMERIC_DISTRIBUTION_PARAMETERS = "distribution-parameters"
_NUMERIC_SEASONALITY = "seasonality"

_DATETIME_EARLIEST = "earliest"
_DATETIME_LATEST = "latest"
_DATETIME_ALL_ENTRIES_UNIQUE = "all-entries-are-unique"
_DATETIME_MONOTONIC_INCREASING = "monotonically-increasing"
_DATETIME_MONOTONIC_DECREASING = "monotonically-decreasing"
_DATETIME_TEMPORAL_CONSISTENCY = "temporal-consistency"


class _Distributions(str, Enum):
    SingleValue = "single value"
    TooSmallDataset = "dataset too small to determine distribution"


class _PerTimeBaseSeasonalityGraphs:
    def __init__(
        self,
        original: TimeBasedGraph,
        trend: TimeBasedGraph,
        seasonality: TimeBasedGraph,
        residual: TimeBasedGraph,
    ) -> None:
        self.original = original
        self.trend = trend
        self.seasonality = seasonality
        self.residual = residual


class _DatetimeColumnTemporalConsistency:
    def __init__(
        self, period: str, temporal_consistencies: List[TemporalConsistency], cleaned_index: DatetimeIndex
    ) -> None:
        self.period = period
        self.temporal_consistencies = temporal_consistencies
        self.cleaned_index = cleaned_index

    def __getitem__(self, period: str) -> TemporalConsistency:
        try:
            return next((consistency for consistency in self.temporal_consistencies if consistency.timeScale == period))
        except StopIteration as error:
            raise KeyError(f"{period} not in the temporal consistencies") from error


class Pandas(Analyzer):
    def __init__(self, data: DataFrame, file: File):
        self._logger = getLogger(__name__)
        self._data = data
        self._file = file
        self._fitting_config = FittingConfig()
        self._workers: int = cpu_count() - 1
        self._max_elements_per_column = 100000
        self._distribution_threshold = 30  # Number of elements in row required to cause distribution to be determined.
        self._computed_data = DataFrame(index=self._data.columns)
        self._intervals = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

    @property
    def data_set_type(self):
        return DataSetType.structured

    async def analyze(self, output_context: OutputContext):
        row_count = len(self._data.index)
        self._logger.info("Started structured data analysis with dataset containing %d rows", row_count)
        columns_by_type = await self._determine_types()
        common_fields = await self._compute_common_fields()

        datetime_ids = columns_by_type[_ColumnType.DateTime]
        datetime_columns = self._data.loc[:, datetime_ids]
        datetime_common_fields = common_fields.loc[datetime_ids]
        datetime_fields = await self._compute_datetime_fields(datetime_columns)
        datetime_fields = concat([datetime_common_fields, datetime_fields], axis=1)
        datetime_index = await _determine_datetime_index(self._logger, datetime_columns)
        datetime_column_name: Optional[str] = None
        datetime_column_periodicity: Optional[_DatetimeColumnTemporalConsistency] = None
        if datetime_index is not None:
            self._logger.info('Using "%s" as index', datetime_index.name)
            self._data.set_index(datetime_index, inplace=True)
            datetime_column_name = str(datetime_index.name)
            datetime_column_periodicity = datetime_fields[_DATETIME_TEMPORAL_CONSISTENCY][datetime_column_name]

        numeric_ids = columns_by_type[_ColumnType.Numeric]
        numeric_columns = self._data.loc[:, numeric_ids]
        numeric_common_fields = common_fields.loc[numeric_ids]
        numeric_fields = await self._compute_numeric_fields(
            numeric_columns, numeric_common_fields, datetime_column_periodicity
        )
        numeric_fields = concat([numeric_fields, numeric_common_fields], axis=1)

        string_ids = columns_by_type[_ColumnType.String]
        string_columns = self._data[string_ids]
        string_fields = common_fields.loc[string_ids]

        transformed_numeric_columns = [
            await self._transform_numeric_results(
                numeric_columns[name], _get_single_row(name, numeric_fields), output_context, datetime_column_name
            )
            for name in numeric_ids
        ]
        transformed_datetime_columns = [
            await self._transform_datetime_results(datetime_columns[name], _get_single_row(name, datetime_fields))
            for name in datetime_ids
        ]
        transformed_string_columns = [
            await self._transform_string_results(string_columns[name], _get_single_row(name, string_fields))
            for name in string_ids
        ]

        correlation_ids = numeric_ids
        correlation_columns = self._data.loc[:, correlation_ids]
        correlation_fields = common_fields.loc[correlation_ids]
        correlation_graph = await _get_correlation_graph(
            self._logger,
            self._file.output_reference + "_correlations",
            correlation_columns,
            correlation_fields,
            output_context,
        )

        transformed_numeric_column_count = len(transformed_numeric_columns)
        transformed_date_time_column_count = len(transformed_datetime_columns)
        transformed_string_column_count = len(transformed_string_columns)
        column_count = (
            transformed_numeric_column_count + transformed_date_time_column_count + transformed_string_column_count
        )

        return StructuredDataSet(
            name=PurePosixPath(self._file.relative),
            rowCount=row_count,
            columnCount=column_count,
            numericColumnCount=transformed_numeric_column_count,
            datetimeColumnCount=transformed_date_time_column_count,
            stringColumnCount=transformed_string_column_count,
            numericColumns=transformed_numeric_columns,
            datetimeColumns=transformed_datetime_columns,
            stringColumns=transformed_string_columns,
            correlationGraph=correlation_graph,
            primaryDatetimeColumn=datetime_column_name,
        )

    async def _compute_common_fields(self) -> DataFrame:
        common_fields = DataFrame(index=self._data.columns)
        common_fields[_COMMON_NON_NULL] = self._data.count()
        common_fields[_COMMON_NULL] = self._data.isna().sum()
        common_fields[_COMMON_UNIQUE] = self._data.nunique(dropna=True)
        return common_fields

    async def _compute_numeric_fields(
        self,
        columns: DataFrame,
        common_fields: DataFrame,
        index_column_periodicity: Optional[_DatetimeColumnTemporalConsistency],
    ) -> DataFrame:
        fields = DataFrame(index=columns.columns)

        fields[_NUMERIC_MIN] = columns.min()
        fields[_NUMERIC_MAX] = columns.max()
        fields[_NUMERIC_LOWER_PERCENTILE] = columns.quantile(0.01)
        fields[_NUMERIC_UPPER_PERCENTILE] = columns.quantile(0.99)
        fields[_NUMERIC_PERCENTILE_OUTLIERS] = _get_outliers(
            columns,
            fields[_NUMERIC_LOWER_PERCENTILE],
            fields[_NUMERIC_UPPER_PERCENTILE],
        )
        # Standard Distribution
        fields[_NUMERIC_MEAN] = columns.mean()
        fields[_NUMERIC_MEDIAN] = columns.median()
        fields[_NUMERIC_STD_DEV] = columns.std()
        fields[_NUMERIC_LOWER_Z] = fields[_NUMERIC_MEAN] - 3.0 * fields[_NUMERIC_STD_DEV]
        fields[_NUMERIC_UPPER_Z] = fields[_NUMERIC_MEAN] + 3.0 * fields[_NUMERIC_STD_DEV]
        fields[_NUMERIC_Z_OUTLIERS] = _get_outliers(columns, fields[_NUMERIC_LOWER_Z], fields[_NUMERIC_UPPER_Z])
        # Inter Quartile Range
        fields[_NUMERIC_LOWER_QUANT] = columns.quantile(0.25)
        fields[_NUMERIC_UPPER_QUANT] = columns.quantile(0.75)
        fields[_NUMERIC_IQR] = fields[_NUMERIC_UPPER_QUANT] - fields[_NUMERIC_LOWER_QUANT]
        fields[_NUMERIC_LOWER_IQR] = fields[_NUMERIC_LOWER_QUANT] - 1.5 * fields[_NUMERIC_IQR]
        fields[_NUMERIC_UPPER_IQR] = fields[_NUMERIC_UPPER_QUANT] + 1.5 * fields[_NUMERIC_IQR]
        fields[_NUMERIC_IQR_OUTLIERS] = _get_outliers(columns, fields[_NUMERIC_LOWER_IQR], fields[_NUMERIC_UPPER_IQR])
        # Distribution
        fields[_NUMERIC_LOWER_DIST] = fields[_NUMERIC_LOWER_IQR]
        fields.loc[fields[_NUMERIC_LOWER_IQR] < fields[_NUMERIC_MIN], _NUMERIC_LOWER_DIST] = fields[_NUMERIC_MIN]
        fields[_NUMERIC_UPPER_DIST] = fields[_NUMERIC_UPPER_IQR]
        fields.loc[fields[_NUMERIC_UPPER_IQR] > fields[_NUMERIC_MAX], _NUMERIC_UPPER_DIST] = fields[_NUMERIC_MAX]
        upper_equals_lower = fields[_NUMERIC_LOWER_DIST] == fields[_NUMERIC_UPPER_DIST]
        fields.loc[upper_equals_lower, _NUMERIC_LOWER_DIST] = fields[_NUMERIC_LOWER_DIST] * 0.9
        fields.loc[upper_equals_lower, _NUMERIC_UPPER_DIST] = fields[_NUMERIC_UPPER_DIST] * 1.1
        fields = concat(
            [
                fields,
                await _get_distributions(
                    self._logger,
                    columns,
                    concat([common_fields, fields], axis=1),
                    self._fitting_config,
                    self._distribution_threshold,
                    self._workers,
                ),
            ],
            axis=1,
        )
        if isinstance(columns.index, DatetimeIndex) and (index_column_periodicity is not None):
            fields[_NUMERIC_SEASONALITY] = await _get_seasonalities(self._logger, columns, index_column_periodicity)
        return fields

    async def _compute_datetime_fields(self, columns: DataFrame) -> DataFrame:
        computed = DataFrame(index=columns.columns)
        computed[_DATETIME_EARLIEST] = columns.min()
        computed[_DATETIME_LATEST] = columns.max()
        # TODO: Vectorize these
        computed[_DATETIME_ALL_ENTRIES_UNIQUE] = Series({name: column.is_unique for name, column in columns.items()})
        computed[_DATETIME_MONOTONIC_INCREASING] = Series(
            {name: column.is_monotonic_increasing for name, column in columns.items()}
        )
        computed[_DATETIME_MONOTONIC_DECREASING] = Series(
            {name: column.is_monotonic_decreasing for name, column in columns.items()}
        )
        computed[_DATETIME_TEMPORAL_CONSISTENCY] = await _compute_temporal_consistency(self._logger, columns)
        return computed

    async def _determine_types(self) -> Dict[_ColumnType, List[str]]:
        columns_by_type: Dict[_ColumnType, List[str]] = {
            _ColumnType.Numeric: [],
            _ColumnType.DateTime: [],
            _ColumnType.String: [],
        }

        for column_name in self._data.columns:
            columns_by_type[await self._determine_type(column_name)].append(column_name)

        self._logger.debug("Found Numeric columns: %s", columns_by_type[_ColumnType.Numeric])
        self._logger.debug("Found DateTime columns: %s", columns_by_type[_ColumnType.DateTime])
        self._logger.debug("Found String columns: %s", columns_by_type[_ColumnType.String])
        return columns_by_type

    async def _determine_type(self, column_name: str) -> _ColumnType:
        self._data[column_name] = infer_type_and_convert(self._data[column_name])
        type_char = self._data[column_name].dtype.kind
        if type_char in "iufcm":
            return _ColumnType.Numeric
        elif type_char in "M":
            return _ColumnType.DateTime
        else:
            return _ColumnType.String

    async def _transform_numeric_results(
        self,
        column: Series,
        computed_fields: Series,
        output_context: OutputContext,
        datetime_index_column: Optional[str],
    ) -> NumericColumn:
        self._logger.debug('Transforming numeric column "%s" results to EDP', column.name)
        column_plot_base = self._file.output_reference + "_" + str(column.name)
        box_plot = await _generate_box_plot(column_plot_base + "_box_plot", column, output_context)
        column_result = NumericColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
            min=computed_fields[_NUMERIC_MIN],
            max=computed_fields[_NUMERIC_MAX],
            mean=computed_fields[_NUMERIC_MEAN],
            median=computed_fields[_NUMERIC_MEDIAN],
            stddev=computed_fields[_NUMERIC_STD_DEV],
            upperPercentile=computed_fields[_NUMERIC_UPPER_PERCENTILE],
            lowerPercentile=computed_fields[_NUMERIC_LOWER_PERCENTILE],
            upperQuantile=computed_fields[_NUMERIC_UPPER_QUANT],
            lowerQuantile=computed_fields[_NUMERIC_LOWER_QUANT],
            percentileOutlierCount=computed_fields[_NUMERIC_PERCENTILE_OUTLIERS],
            upperZScore=computed_fields[_NUMERIC_UPPER_Z],
            lowerZScore=computed_fields[_NUMERIC_LOWER_Z],
            zScoreOutlierCount=computed_fields[_NUMERIC_Z_OUTLIERS],
            upperIQR=computed_fields[_NUMERIC_UPPER_IQR],
            lowerIQR=computed_fields[_NUMERIC_LOWER_IQR],
            iqr=computed_fields[_NUMERIC_IQR],
            iqrOutlierCount=computed_fields[_NUMERIC_IQR_OUTLIERS],
            distribution=computed_fields[_NUMERIC_DISTRIBUTION],
            dataType=str(column.dtype),
            boxPlot=box_plot,
        )
        if not computed_fields[_NUMERIC_DISTRIBUTION] in [
            _Distributions.SingleValue.value,
            _Distributions.TooSmallDataset.value,
        ]:
            column_result.distributionGraph = await _plot_distribution(
                column,
                computed_fields,
                self._fitting_config,
                column_plot_base + "_distribution",
                computed_fields[_NUMERIC_DISTRIBUTION],
                computed_fields[_NUMERIC_DISTRIBUTION_PARAMETERS],
                output_context,
            )

        if (
            (_NUMERIC_SEASONALITY in computed_fields.index)
            and (computed_fields[_NUMERIC_SEASONALITY] is not None)
            and (datetime_index_column is not None)
        ):
            seasonality_graphs = await _get_seasonality_graphs(
                str(column.name),
                column_plot_base,
                datetime_index_column,
                computed_fields[_NUMERIC_SEASONALITY],
                output_context,
            )
            column_result.original_series = [seasonality_graphs.original]
            column_result.trends = [seasonality_graphs.trend]
            column_result.seasonalities = [seasonality_graphs.seasonality]
            column_result.residuals = [seasonality_graphs.residual]
        return column_result

    async def _transform_datetime_results(self, column: Series, computed_fields: Series) -> DateTimeColumn:
        self._logger.debug('Transforming datetime column "%s" results to EDP', column.name)
        temporal_consistency: _DatetimeColumnTemporalConsistency = computed_fields[_DATETIME_TEMPORAL_CONSISTENCY]

        return DateTimeColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
            temporalCover=TemporalCover(
                earliest=computed_fields[_DATETIME_EARLIEST], latest=computed_fields[_DATETIME_LATEST]
            ),
            all_entries_are_unique=computed_fields[_DATETIME_ALL_ENTRIES_UNIQUE],
            monotonically_increasing=computed_fields[_DATETIME_MONOTONIC_INCREASING],
            monotonically_decreasing=computed_fields[_DATETIME_MONOTONIC_DECREASING],
            temporalConsistencies=temporal_consistency.temporal_consistencies,
            periodicity=temporal_consistency.period,
        )

    async def _transform_string_results(self, column: Series, computed_fields: Series) -> StringColumn:
        self._logger.debug('Transforming string column "%s" results to EDP', column.name)
        return StringColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
        )


async def _generate_box_plot(plot_name: str, column: Series, output_context: OutputContext) -> FileReference:
    async with output_context.get_plot(plot_name) as (axes, reference):
        if isinstance(axes.figure, Figure):
            axes.figure.set_figwidth(3.0)
        axes.boxplot(column, notch=False, tick_labels=[str(column.name)])
    return reference


async def _compute_temporal_consistency(logger: Logger, columns: DataFrame) -> Series:
    return Series(
        {
            name: _compute_temporal_consistency_for_column(logger, DatetimeIndex(columns[name]))
            for name in columns.columns
        }
    )


def _compute_temporal_consistency_for_column(
    logger: Logger, index: DatetimeIndex
) -> _DatetimeColumnTemporalConsistency:
    index = index.sort_values(ascending=True)
    row_count = len(index)

    # Remove null entries
    index = index[index.notnull()]  # type: ignore
    new_count = len(index)
    if new_count < row_count:
        empty_index_count = row_count - new_count
        message = f"Filtered out {empty_index_count} rows, because their index was empty"
        logger.warning(message)
        warn(message)
        row_count = new_count

    granularities = {
        "microseconds": Timedelta(microseconds=1),
        "milliseconds": Timedelta(milliseconds=1),
        "seconds": Timedelta(seconds=1),
        "minutes": Timedelta(minutes=1),
        "hours": Timedelta(hours=1),
        "days": Timedelta(days=1),
        "weeks": Timedelta(weeks=1),
        "months": Timedelta(days=30),
        "years": Timedelta(days=365),
    }

    deltas = index[1:] - index[:-1]
    gaps = Series(
        {label: count_nonzero(deltas > time_base) for label, time_base in granularities.items()},
        dtype=UInt64Dtype(),
    )
    distincts = Series(
        {label: len(index.round(time_base).unique()) for label, time_base in granularities.items()},  # type: ignore
        dtype=UInt64Dtype(),
    )
    stable = distincts == 1
    diff = distincts - gaps
    periodicity: str = diff.idxmax()  # type: ignore
    temporal_consistencies = [
        TemporalConsistency(
            timeScale=time_base,
            differentAbundancies=distincts[time_base],
            stable=stable[time_base],
            numberOfGaps=gaps[time_base],
        )
        for time_base in granularities
    ]

    return _DatetimeColumnTemporalConsistency(periodicity, temporal_consistencies, index)


def _get_outliers(column: DataFrame, lower_limit: Series, upper_limit: Series) -> Series:
    is_outlier = (column < lower_limit) | (column > upper_limit)
    return is_outlier.count()


async def _get_distributions(
    logger: Logger,
    columns: DataFrame,
    fields: DataFrame,
    config: FittingConfig,
    distribution_threshold: int,
    workers: int,
) -> DataFrame:
    distributions: List[Tuple[str, Dict]] = []
    for index, values in enumerate(columns.items(), start=1):
        name, column = values
        distributions.append(
            await _get_distribution(column, _get_single_row(name, fields), config, distribution_threshold, workers)
        )
        logger.debug("Computed %d/%d distributions", index, len(columns.columns))

    data_frame = DataFrame(
        distributions,
        index=columns.columns,
        columns=[_NUMERIC_DISTRIBUTION, _NUMERIC_DISTRIBUTION_PARAMETERS],
    )
    return data_frame


async def _get_distribution(
    column: Series, fields: Series, config: FittingConfig, distribution_threshold: int, workers: int
) -> Tuple[str, Dict]:
    if fields[_COMMON_UNIQUE] <= 1:
        return _Distributions.SingleValue.value, dict()

    if fields[_COMMON_NON_NULL] < distribution_threshold:
        return _Distributions.TooSmallDataset.value, dict()

    return await _find_best_distribution(column, config, fields, workers)


async def _find_best_distribution(
    column: Series, config: FittingConfig, column_fields: Series, workers: int
) -> Tuple[str, dict]:
    loop = get_running_loop()
    fitter = Fitter(
        column,
        xmin=column_fields[_NUMERIC_LOWER_DIST],
        xmax=column_fields[_NUMERIC_UPPER_DIST],
        timeout=config.timeout.total_seconds(),
        bins=config.bins,
        distributions=config.distributions,
    )

    def runner():
        return fitter.fit(max_workers=workers)

    await loop.run_in_executor(None, runner)
    # According to documentation, this ony ever contains one entry.
    best_distribution_dict = fitter.get_best(method=config.error_function)
    distribution_name, parameters = next(iter(best_distribution_dict.items()))
    return str(distribution_name), parameters


async def _plot_distribution(
    column: Series,
    column_fields: Series,
    config: FittingConfig,
    plot_name: str,
    distribution_name: str,
    distribution_parameters: dict,
    output_context: OutputContext,
):
    x_min = column_fields[_NUMERIC_LOWER_DIST]
    x_max = column_fields[_NUMERIC_UPPER_DIST]
    x_limits = (x_min, x_max)
    async with output_context.get_plot(plot_name) as (axes, reference):
        axes.set_title(f"Distribution of {column.name}")
        axes.set_xlabel(f"Value of {column.name}")
        axes.set_ylabel("Relative Density")
        axes.set_xlim(x_min, x_max)
        axes.hist(column, bins=config.bins, range=x_limits, density=True, label=f"{column.name} Value Distribution")
        x = linspace(x_min, x_max, 2048)
        distribution = getattr(distributions, distribution_name)
        distribution_y = distribution.pdf(x, **distribution_parameters)
        axes.plot(x, distribution_y, label=f"Best Fit Model Distribution {distribution_name}")
        axes.legend()
    return reference


def _get_single_row(row_name: str | Hashable, data_frame: DataFrame) -> Series:
    """This is mostly a convenience wrapper due to poor pandas stubs."""
    return data_frame.loc[str(row_name)]  # type: ignore


async def _get_correlation_graph(
    logger: Logger, plot_name: str, columns: DataFrame, fields: DataFrame, output_context: OutputContext
) -> Optional[FileReference]:
    filtered_column_names = fields.loc[fields[_COMMON_UNIQUE] > 1].index
    if len(filtered_column_names) < 2:
        return None
    filtered_columns = columns[filtered_column_names]
    logger.debug("Computing correlation between columns %s", filtered_columns.columns)
    correlation_matrix = corrcoef(filtered_columns.values, rowvar=False)
    correlation = DataFrame(
        correlation_matrix,
        columns=filtered_columns.columns,
        index=filtered_columns.columns,
    )
    mask = triu(ones_like(correlation, dtype=bool))
    async with output_context.get_plot(plot_name) as (axes, reference):
        figure = axes.figure
        if isinstance(figure, Figure):
            width_offset = 3
            height_offset = 3
            figure_size = (correlation.shape[0] + width_offset, correlation.shape[1] + height_offset)
            figure.set_size_inches(figure_size)
        heatmap(
            correlation,
            annot=True,
            mask=mask,
            fmt=".2f",
            vmin=-1.0,
            center=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            cmap=get_cmap(),
            ax=axes,
        )
    logger.debug("Finished computing correlations")
    return reference


async def _determine_datetime_index(logger: Logger, columns: DataFrame) -> Optional[DatetimeIndex]:
    # TODO: Remove this function. In future release, there will not be any single primary date time column.
    #       All date time columns will be evaluated against all numeric rows rows for analysis.
    frequency = "infer"
    number_columns = len(columns.columns)
    if number_columns == 0:
        return None

    if number_columns == 1:
        return DatetimeIndex(data=columns.iloc[:, 0], freq=frequency)

    for _, column in columns.items():
        if column.is_monotonic_increasing:
            return DatetimeIndex(data=column, freq=frequency)

    for _, column in columns.items():
        if column.is_monotonic_decreasing:
            return DatetimeIndex(data=column, freq=frequency)

    first_column = columns.iloc[:, 0]
    warning_text = f'Did not find a monotonic datetime column, will use "{first_column.name}" as index'
    logger.warning(warning_text)
    warn(warning_text, RuntimeWarning)
    return DatetimeIndex(data=first_column, freq=frequency)


async def _get_seasonalities(
    logger: Logger, columns: DataFrame, index_column_periodicity: _DatetimeColumnTemporalConsistency
) -> Series:
    row_count = len(columns.index)
    logger.info("Starting seasonality analysis on %d rows", row_count)

    filtered_columns = columns.loc[index_column_periodicity.cleaned_index]
    distincts = index_column_periodicity[index_column_periodicity.period].differentAbundancies
    series: Series = Series(
        {
            name: _seasonal_decompose_column(column=column, distinct_count=distincts)
            for name, column in filtered_columns.items()
        }
    )
    logger.info("Finished seasonality analysis, found highest periodicity at %s level", index_column_periodicity.period)
    return series


def _seasonal_decompose_column(column: Series, distinct_count: int) -> Optional[DecomposeResult]:
    non_null_column = column[column.notnull()]
    number_non_null = len(non_null_column)
    if number_non_null < 1:
        return None
    divider = min(16, number_non_null)
    period = int(distinct_count / divider)
    period = max(1, period)
    return seasonal_decompose(non_null_column, period=period, model="additive")


async def _get_seasonality_graphs(
    column_name: str,
    column_plot_base: str,
    time_base_column: str,
    seasonality: DecomposeResult,
    output_context: OutputContext,
) -> _PerTimeBaseSeasonalityGraphs:
    xlim = seasonality._observed.index[0], seasonality._observed.index[-1]

    @asynccontextmanager
    async def get_plot(plot_type: str):
        async with output_context.get_plot(column_plot_base + "_" + plot_type.lower()) as (axes, reference):
            axes.set_title(f"{plot_type} of {column_name} over {time_base_column}")
            axes.set_xlabel(time_base_column)
            axes.set_ylabel(f"{plot_type} of {column_name}")
            if axes.figure:
                axes.set_xlim(xlim)
            if isinstance(axes.figure, Figure):
                axes.figure.set_figwidth(20)
            yield axes, reference

    async with get_plot("Original") as (axes, original_reference):
        axes.plot(seasonality.observed)

    async with get_plot("Trend") as (axes, trend_reference):
        axes.plot(seasonality.trend)

    async with get_plot("Seasonality") as (axes, seasonal_reference):
        axes.plot(seasonality.seasonal)

    async with get_plot("Residual") as (axes, residual_reference):
        axes.plot(seasonality.resid, marker="o", linestyle="none")
        axes.plot(xlim, (0, 0), zorder=-3)

    return _PerTimeBaseSeasonalityGraphs(
        original=TimeBasedGraph(timeBaseColumn=time_base_column, file=original_reference),
        trend=TimeBasedGraph(timeBaseColumn=time_base_column, file=trend_reference),
        seasonality=TimeBasedGraph(timeBaseColumn=time_base_column, file=seasonal_reference),
        residual=TimeBasedGraph(timeBaseColumn=time_base_column, file=residual_reference),
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
        return try_convert_datetime(column)

    if type_character in "m":
        return column

    try:
        return try_convert_datetime(column)
    except (ValueError, TypeError):
        pass

    try:
        return infer_type_and_convert(to_numeric(column.str.replace(",", "."), errors="raise"))
    except (ValueError, TypeError):
        pass

    return column


def try_convert_datetime(column: Series) -> Series:
    for format in DATE_TIME_FORMATS:
        try:
            return to_datetime(column, errors="raise", format=format)
        except (ValueError, TypeError):
            pass
    raise TypeError
