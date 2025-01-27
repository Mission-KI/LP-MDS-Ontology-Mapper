from contextlib import asynccontextmanager
from typing import Optional
from warnings import warn

from extended_dataset_profile.models.v0.edp import TimeBasedGraph
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

from edps.analyzers.pandas.temporal_consistency import DatetimeColumnTemporalConsistency
from edps.analyzers.pandas.type_parser import ColumnsWrapper, DatetimeColumnInfo, DatetimeKind
from edps.filewriter import get_pyplot_writer
from edps.task import TaskContext


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

        different_abundances = periodicity.get_main_temporal_consistency().differentAbundancies
        filtered_numeric_columns = numeric_columns.loc[periodicity.cleaned_series.index].set_index(
            periodicity.cleaned_series, inplace=False
        )

        for numeric_column_name, numeric_column in filtered_numeric_columns.items():
            dataframe.loc[str(numeric_column_name), str(datetime_column_name)] = _seasonal_decompose_column(
                numeric_column, different_abundances
            )

    ctx.logger.info("Finished seasonality analysis.")
    return dataframe


def _seasonal_decompose_column(column: Series, distinct_count: int) -> Optional[DecomposeResult]:
    non_null_column = column[column.notnull()]
    number_non_null = len(non_null_column)
    if number_non_null < 1:
        return None
    divider = min(16, number_non_null)
    period = int(distinct_count / divider)
    period = max(1, period)
    return seasonal_decompose(non_null_column, period=period, model="additive")


async def get_seasonality_graphs(
    ctx: TaskContext,
    column_name: str,
    column_plot_base: str,
    time_base_column_name: str,
    seasonality: DecomposeResult,
) -> PerTimeBaseSeasonalityGraphs:
    xlim = seasonality._observed.index[0], seasonality._observed.index[-1]

    @asynccontextmanager
    async def get_plot(plot_type: str):
        plot_name = column_plot_base + "_over_" + time_base_column_name + "_" + plot_type.lower()
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
