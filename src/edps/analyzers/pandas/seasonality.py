from contextlib import asynccontextmanager
from typing import Optional
from warnings import warn

from extended_dataset_profile.models.v0.edp import TimeBasedGraph
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

from edps.analyzers.pandas.temporal_consistency import DatetimeColumnTemporalConsistency, Granularity
from edps.analyzers.pandas.type_parser import ColumnsWrapper, DatetimeColumnInfo, DatetimeKind
from edps.filewriter import get_pyplot_writer
from edps.taskcontext import TaskContext


class PerTimeBaseSeasonalityGraphs:
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


async def compute_seasonality(
    ctx: TaskContext,
    datetime_columns: ColumnsWrapper[DatetimeColumnInfo],
    datetime_column_periodicities: Series,
    numeric_columns: DataFrame,
) -> DataFrame:
    datetime_count = len(datetime_columns.ids)
    row_count = len(numeric_columns.index)
    ctx.logger.info("Starting seasonality analysis on %d rows over %d time bases", row_count, datetime_count)

    dataframe = DataFrame(index=numeric_columns.columns, columns=datetime_columns.ids, dtype=object)

    periodicity: DatetimeColumnTemporalConsistency
    for datetime_column_name, periodicity in datetime_column_periodicities.items():
        datetime_kind = datetime_columns.get_info(str(datetime_column_name)).get_kind()

        if datetime_kind == DatetimeKind.DATE:
            message = (
                f'Column "{datetime_column_name}" has date only format. This might impede the seasonality analysis. '
                "To fix this, supply it as datetime, preferably in the ISO8601 format."
            )
            warn(message)
            ctx.logger.warning(message)
        elif datetime_kind == DatetimeKind.TIME:
            message = (
                f'Column "{datetime_column_name}" has time only format. It will be skipped for seasonality analysis'
            )
            warn(message)
            ctx.logger.warning(message)
            continue

        filtered_numeric_columns = numeric_columns.loc[periodicity.cleaned_series.index].set_index(
            periodicity.cleaned_series, inplace=False
        )

        for numeric_column_name, numeric_column in filtered_numeric_columns.items():
            dataframe.loc[str(numeric_column_name), str(datetime_column_name)] = _seasonal_decompose_column(
                ctx, numeric_column, periodicity.main_period, str(datetime_column_name)
            )

    ctx.logger.info("Finished seasonality analysis.")
    return dataframe


def _seasonal_decompose_column(
    ctx: TaskContext, column: Series, resample_period: Granularity, datetime_column_name: str
) -> Optional[DecomposeResult]:
    config = ctx.config.structured_config.seasonality
    non_null_column: Series = column[column.notnull()]
    number_non_null = len(non_null_column)
    if number_non_null > config.max_samples:
        ctx.logger.info(
            'Column "%s" contains too many entries. Running seasonality on the first %d entries only.',
            column.name,
            config.max_samples,
        )
        non_null_column = non_null_column[: config.max_samples]
    non_null_column = non_null_column.resample(resample_period.timedelta).mean().interpolate()
    resample_period_samples = resample_period.samples_per_period
    number_non_null = len(non_null_column.index)

    number_periods = number_non_null / resample_period_samples
    if number_periods < 2.0:
        ctx.logger.info(
            'Column "%s" contains only %d samples when resampled to %s for running seasonality analysis. Need at least %d samples.',
            column.name,
            number_non_null,
            resample_period,
            2 * resample_period_samples,
        )
        return None

    if number_periods > config.max_periods:
        ctx.logger.debug(
            'Column "%s" contains %d periods of data. Will be capped to %d periods.',
            column.name,
            number_periods,
            config.max_periods,
        )
        number_non_null = config.max_periods * resample_period_samples
        non_null_column = non_null_column[:number_non_null]

    try:
        return seasonal_decompose(non_null_column, model="additive", period=resample_period_samples)
    except ValueError as error:
        message = f"Seasonal decompose of {column.name} over {datetime_column_name} failed: {error}"
        ctx.logger.warning(message)
        warn(message)
        return None


async def get_seasonality_graphs(
    ctx: TaskContext,
    column_name: str,
    time_base_column_name: str,
    seasonality: DecomposeResult,
) -> PerTimeBaseSeasonalityGraphs:
    xlim = seasonality._observed.index[0], seasonality._observed.index[-1]

    @asynccontextmanager
    async def get_plot(plot_type: str):
        plot_name = ctx.build_output_reference(f"{column_name}_over_{time_base_column_name}_{plot_type.lower()}")
        async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
            axes.set_title(f"{plot_type} of {column_name} over {time_base_column_name}")
            axes.set_xlabel(time_base_column_name)
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

    return PerTimeBaseSeasonalityGraphs(
        original=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=original_reference),
        trend=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=trend_reference),
        seasonality=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=seasonal_reference),
        residual=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=residual_reference),
    )
