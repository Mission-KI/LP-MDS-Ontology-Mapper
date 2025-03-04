from asyncio import get_running_loop
from collections.abc import Hashable
from datetime import timedelta
from enum import Enum
from multiprocessing import cpu_count
from pathlib import PurePosixPath
from typing import AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

from extended_dataset_profile.models.v0.edp import (
    DateTimeColumn,
    FileReference,
    NumericColumn,
    StringColumn,
    StructuredDataSet,
    TemporalCover,
)
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from numpy import linspace, ones_like, triu
from pandas import (
    DataFrame,
    Series,
    concat,
)
from scipy.stats import distributions
from seaborn import heatmap

from edps.analyzers.base import Analyzer
from edps.analyzers.pandas.fitter import Fitter, FittingConfig
from edps.analyzers.pandas.seasonality import compute_seasonality, get_seasonality_graphs
from edps.analyzers.pandas.temporal_consistency import DatetimeColumnTemporalConsistency, compute_temporal_consistency
from edps.analyzers.pandas.temporal_consistency import determine_periodicity as determine_periodicity
from edps.analyzers.pandas.type_parser import (
    DatetimeColumnInfo,
    parse_types,
)
from edps.file import File
from edps.filewriter import get_pyplot_writer
from edps.task import TaskContext

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

_DATETIME_EARLIEST = "earliest"
_DATETIME_LATEST = "latest"
_DATETIME_ALL_ENTRIES_UNIQUE = "all-entries-are-unique"
_DATETIME_MONOTONIC_INCREASING = "monotonically-increasing"
_DATETIME_MONOTONIC_DECREASING = "monotonically-decreasing"
_DATETIME_TEMPORAL_CONSISTENCY = "temporal-consistency"


class _Distributions(str, Enum):
    SingleValue = "single value"
    TooSmallDataset = "dataset too small to determine distribution"


class PandasAnalyzer(Analyzer):
    def __init__(self, data: DataFrame, file: File):
        self._data = data
        self._file = file
        self._workers: int = cpu_count() - 1
        self._max_elements_per_column = 100000
        self._distribution_threshold = 30  # Number of elements in row required to cause distribution to be determined.
        self._intervals = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

    async def analyze(self, ctx: TaskContext) -> AsyncIterator[StructuredDataSet]:
        row_count = len(self._data.index)
        ctx.logger.info(
            "Started structured data analysis with dataset containing %d rows",
            row_count,
        )

        type_parser_results = ctx.exec(parse_types, self._data)

        all_cols = type_parser_results.all_cols
        common_fields = await self._compute_common_fields(all_cols.data)

        datetime_cols = type_parser_results.datetime_cols
        datetime_common_fields = common_fields.loc[datetime_cols.ids]
        datetime_fields = await self._compute_datetime_fields(ctx, datetime_cols.data)
        datetime_fields = concat([datetime_common_fields, datetime_fields], axis=1)

        numeric_cols = type_parser_results.numeric_cols
        numeric_common_fields = common_fields.loc[numeric_cols.ids]
        numeric_fields = await self._compute_numeric_fields(ctx, numeric_cols.data, numeric_common_fields)
        numeric_fields = concat([numeric_fields, numeric_common_fields], axis=1)

        string_cols = type_parser_results.string_cols
        string_fields = common_fields.loc[string_cols.ids]

        timebase_periodicities = datetime_fields[_DATETIME_TEMPORAL_CONSISTENCY]
        seasonality_results = await compute_seasonality(ctx, datetime_cols, timebase_periodicities, numeric_cols.data)

        transformed_numeric_columns = [
            await self._transform_numeric_results(
                ctx,
                numeric_cols.get_col(id),
                _get_single_row(id, numeric_fields),
                _get_single_row(id, seasonality_results),
            )
            for id in numeric_cols.ids
        ]
        transformed_datetime_columns = [
            await self._transform_datetime_results(
                ctx,
                datetime_cols.get_col(id),
                _get_single_row(id, datetime_fields),
                datetime_cols.get_info(id),
            )
            for id in datetime_cols.ids
        ]
        transformed_string_columns = [
            await self._transform_string_results(ctx, string_cols.get_col(id), _get_single_row(id, string_fields))
            for id in string_cols.ids
        ]

        correlation_ids = numeric_cols.ids
        correlation_columns = all_cols.data.loc[:, correlation_ids]
        correlation_fields = common_fields.loc[correlation_ids]
        correlation_graph = await _get_correlation_graph(
            ctx,
            self._file.output_reference + "_correlations",
            correlation_columns,
            correlation_fields,
        )

        transformed_numeric_column_count = len(transformed_numeric_columns)
        transformed_date_time_column_count = len(transformed_datetime_columns)
        transformed_string_column_count = len(transformed_string_columns)
        column_count = (
            transformed_numeric_column_count + transformed_date_time_column_count + transformed_string_column_count
        )

        yield StructuredDataSet(
            uuid=uuid4(),
            parentUuid=None,
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
        )

    async def _compute_common_fields(self, columns: DataFrame) -> DataFrame:
        common_fields = DataFrame(index=columns.columns)
        common_fields[_COMMON_NON_NULL] = columns.count()
        common_fields[_COMMON_NULL] = columns.isna().sum()
        common_fields[_COMMON_UNIQUE] = columns.nunique(dropna=True)
        return common_fields

    async def _compute_numeric_fields(
        self,
        ctx: TaskContext,
        columns: DataFrame,
        common_fields: DataFrame,
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
                    ctx,
                    columns,
                    concat([common_fields, fields], axis=1),
                    self._distribution_threshold,
                    self._workers,
                ),
            ],
            axis=1,
        )
        return fields

    async def _compute_datetime_fields(self, ctx: TaskContext, columns: DataFrame) -> DataFrame:
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
        computed[_DATETIME_TEMPORAL_CONSISTENCY] = await compute_temporal_consistency(ctx, columns)
        return computed

    async def _transform_numeric_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series, seasonality_results: Series
    ) -> NumericColumn:
        ctx.logger.debug('Transforming numeric column "%s" results to EDP', column.name)
        column_plot_base = self._file.output_reference + "_" + str(column.name)
        box_plot = await _generate_box_plot(ctx, column_plot_base + "_box_plot", column)

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
        if computed_fields[_NUMERIC_DISTRIBUTION] not in [
            _Distributions.SingleValue.value,
            _Distributions.TooSmallDataset.value,
        ]:
            column_result.distributionGraph = await _plot_distribution(
                ctx,
                column,
                computed_fields,
                column_plot_base + "_distribution",
                computed_fields[_NUMERIC_DISTRIBUTION],
                computed_fields[_NUMERIC_DISTRIBUTION_PARAMETERS],
            )

        for datetime_column_name, seasonality in seasonality_results.items():
            seasonality_graphs = await get_seasonality_graphs(
                ctx,
                str(column.name),
                column_plot_base,
                str(datetime_column_name),
                seasonality,
            )
            column_result.original_series.append(seasonality_graphs.original)
            column_result.trends.append(seasonality_graphs.trend)
            column_result.seasonalities.append(seasonality_graphs.seasonality)
            column_result.residuals.append(seasonality_graphs.residual)
        return column_result

    async def _transform_datetime_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series, info: DatetimeColumnInfo
    ) -> DateTimeColumn:
        ctx.logger.debug('Transforming datetime column "%s" results to EDP', column.name)
        temporal_consistency: Optional[DatetimeColumnTemporalConsistency] = computed_fields[
            _DATETIME_TEMPORAL_CONSISTENCY
        ]

        return DateTimeColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
            temporalCover=TemporalCover(
                earliest=computed_fields[_DATETIME_EARLIEST],
                latest=computed_fields[_DATETIME_LATEST],
            ),
            all_entries_are_unique=computed_fields[_DATETIME_ALL_ENTRIES_UNIQUE],
            monotonically_increasing=computed_fields[_DATETIME_MONOTONIC_INCREASING],
            monotonically_decreasing=computed_fields[_DATETIME_MONOTONIC_DECREASING],
            temporalConsistencies=(temporal_consistency.temporal_consistencies if temporal_consistency else []),
            periodicity=temporal_consistency.period if temporal_consistency else None,
            format=info.get_format(),
        )

    async def _transform_string_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series
    ) -> StringColumn:
        ctx.logger.debug('Transforming string column "%s" results to EDP', column.name)
        return StringColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
        )


async def _generate_box_plot(ctx: TaskContext, plot_name: str, column: Series) -> FileReference:
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        if isinstance(axes.figure, Figure):
            axes.figure.set_figwidth(3.0)
        axes.boxplot(column, notch=False, tick_labels=[str(column.name)])
    return reference


def _get_outliers(column: DataFrame, lower_limit: Series, upper_limit: Series) -> Series:
    is_outlier = (column < lower_limit) | (column > upper_limit)
    return is_outlier.count()


async def _get_distributions(
    ctx: TaskContext,
    columns: DataFrame,
    fields: DataFrame,
    distribution_threshold: int,
    workers: int,
) -> DataFrame:
    distributions: List[Tuple[str, Dict]] = []
    for index, values in enumerate(columns.items(), start=1):
        name, column = values
        distributions.append(
            await _get_distribution(
                ctx,
                column,
                _get_single_row(name, fields),
                distribution_threshold,
                workers,
            )
        )
        ctx.logger.debug("Computed %d/%d distributions", index, len(columns.columns))

    data_frame = DataFrame(
        distributions,
        index=columns.columns,
        columns=[_NUMERIC_DISTRIBUTION, _NUMERIC_DISTRIBUTION_PARAMETERS],
    )
    return data_frame


async def _get_distribution(
    ctx: TaskContext,
    column: Series,
    fields: Series,
    distribution_threshold: int,
    workers: int,
) -> Tuple[str, Dict]:
    if fields[_COMMON_UNIQUE] <= 1:
        return _Distributions.SingleValue.value, dict()

    if fields[_COMMON_NON_NULL] < distribution_threshold:
        return _Distributions.TooSmallDataset.value, dict()

    return await _find_best_distribution(ctx, column, fields, workers)


async def _find_best_distribution(
    ctx: TaskContext, column: Series, column_fields: Series, workers: int
) -> Tuple[str, dict]:
    loop = get_running_loop()
    config = FittingConfig(min=column_fields[_NUMERIC_LOWER_DIST], max=column_fields[_NUMERIC_UPPER_DIST])
    fitter = Fitter(column, config)

    def runner():
        return fitter.fit(ctx)

    await loop.run_in_executor(None, runner)
    return await fitter.get_best(ctx)


async def _plot_distribution(
    ctx: TaskContext,
    column: Series,
    column_fields: Series,
    plot_name: str,
    distribution_name: str,
    distribution_parameters: dict,
):
    x_min = column_fields[_NUMERIC_LOWER_DIST]
    x_max = column_fields[_NUMERIC_UPPER_DIST]
    x_limits = (x_min, x_max)
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        axes.set_title(f"Distribution of {column.name}")
        axes.set_xlabel(f"Value of {column.name}")
        axes.set_ylabel("Relative Density")
        axes.set_xlim(x_min, x_max)
        axes.hist(
            column,
            bins=100,
            range=x_limits,
            density=True,
            label=f"{column.name} Value Distribution",
        )
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
    ctx: TaskContext,
    plot_name: str,
    columns: DataFrame,
    fields: DataFrame,
) -> Optional[FileReference]:
    filtered_column_names = fields.loc[fields[_COMMON_UNIQUE] > 1].index
    if len(filtered_column_names) < 2:
        return None
    filtered_columns = columns[filtered_column_names]
    ctx.logger.debug("Computing correlation between columns %s", filtered_columns.columns)
    correlation = filtered_columns.corr()
    mask = triu(ones_like(correlation, dtype=bool))
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        figure = axes.figure
        if isinstance(figure, Figure):
            width_offset = 3
            height_offset = 3
            figure_size = (
                correlation.shape[0] + width_offset,
                correlation.shape[1] + height_offset,
            )
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
    ctx.logger.debug("Finished computing correlations")
    return reference
