from asyncio import get_running_loop
from datetime import timedelta
from logging import getLogger
from multiprocessing import cpu_count
from typing import AsyncIterator, List, Tuple, Union

from fitter import Fitter
from numpy import any as numpy_any
from numpy import count_nonzero, linspace
from numpy import max as numpy_max
from numpy import mean, median
from numpy import min as numpy_min
from numpy import ndarray, std, unique
from numpy.random import choice as random_choice
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    to_datetime,
    to_numeric,
    to_timedelta,
)
from pydantic import BaseModel, Field
from scipy.stats import distributions

from edp.analyzers.base import Analyzer
from edp.context import OutputContext
from edp.file import File
from edp.types import (
    BaseColumnCounts,
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


def _default_distributions() -> List[str]:
    return [
        "anglit",
        "arcsine",
        "argus",
        "beta",
        "betaprime",
        "bradford",
        "burr",
        "burr12",
        "cauchy",
        "chi",
        "chi2",
        "cosine",
        "crystalball",
        "dgamma",
        "dweibull",
        "erlang",
        "expon",
        "exponnorm",
        "exponpow",
        "exponweib",
        "f",
        "fatiguelife",
        "fisk",
        "foldcauchy",
        "foldnorm",
        "frechet_l",
        "frechet_r",
        "gamma",
        "gausshyper",
        "genexpon",
        "genextreme",
        "gengamma",
        "genhalflogistic",
        "geninvgauss",
        "genlogistic",
        "gennorm",
        "genpareto",
        "gilbrat",
        "gompertz",
        "gumbel_l",
        "gumbel_r",
        "halfcauchy",
        "halfgennorm",
        "halflogistic",
        "halfnorm",
        "hypsecant",
        "invgamma",
        "invgauss",
        "invweibull",
        "johnsonsb",
        "johnsonsu",
        "kappa3",
        "kappa4",
        "ksone",
        "kstwo",
        "kstwobign",
        "laplace",
        "levy",
        "levy_l",
        "levy_stable",
        "loggamma",
        "logistic",
        "loglaplace",
        "lognorm",
        "loguniform",
        "lomax",
        "maxwell",
        "mielke",
        "moyal",
        "nakagami",
        "ncf",
        "nct",
        "ncx2",
        "norm",
        "norminvgauss",
        "pareto",
        "pearson3",
        "powerlaw",
        "powerlognorm",
        "powernorm",
        "rayleigh",
        "rdist",
        "recipinvgauss",
        "reciprocal",
        "rice",
        "rv_continuous",
        "rv_histogram",
        "semicircular",
        "skewnorm",
        "t",
        "trapz",
        "triang",
        "truncexpon",
        "truncnorm",
        "tukeylambda",
        "uniform",
        "vonmises",
        "vonmises_line",
        "wald",
        "weibull_max",
        "weibull_min",
        "wrapcauchy",
    ]


class FittingConfig(BaseModel):
    timeout: timedelta = Field(default=timedelta(seconds=10), description="Timeout to use for the fitting")
    error_function: str = Field(
        default="sumsquare_error", description="Error function to use to measure performance of the fits"
    )
    bins: int = Field(default=100, description="Number of bins to use for the fitting")
    distributions: List[str] = Field(default_factory=_default_distributions, description="Distributions to try to fit")
    workers: int = Field(
        default_factory=lambda: (cpu_count() - 1),
        description="Number of workers to use for fitting. Defaults to number cores - 1 to not block up any other processes.",
    )


class Pandas(Analyzer):
    def __init__(self, data: DataFrame, file: File):
        self._logger = getLogger(__name__)
        self._data = data
        self._file = file
        self._fitting_config = FittingConfig()
        self._max_elements_per_column = 100000

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
        column_plot_base = self._file.output_reference + "_" + str(column.name)
        upper_percentile, lower_percentile, percentile_outliers = compute_percentiles(column)
        upper_z_score, lower_z_score, z_outliers = compute_standard_score(column)
        upper_quantile, lower_quantile, iqr, upper_iqr_limit, lower_iqr_limit, iqr_outliers = (
            compute_inter_quartile_range(column)
        )
        counts = compute_counts(column)
        minimum = numpy_min(column)
        maximum = numpy_max(column)
        x_limits = _get_distribution_x_limits(minimum, maximum, lower_iqr_limit, upper_iqr_limit)

        distribution_name, distribution_plot = await compute_best_fit_distribution(
            column,
            self._fitting_config,
            column_plot_base + "_distribution",
            self._max_elements_per_column,
            output_context,
            x_limits,
        )
        images = [
            await generate_box_plot(column_plot_base + "_box_plot", column, output_context),
            distribution_plot,
        ]
        return NumericColumn(
            nonNullCount=counts.nonNullCount,
            nullCount=counts.nullCount,
            numberUnique=counts.numberUnique,
            images=images,
            min=minimum,
            max=maximum,
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
            distribution=str(distribution_name),
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

        counts = compute_counts(column)
        return DateTimeColumn(
            nonNullCount=counts.nonNullCount,
            nullCount=counts.nullCount,
            numberUnique=counts.numberUnique,
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
        counts = compute_counts(column)
        return StringColumn(
            nonNullCount=counts.nonNullCount,
            nullCount=counts.nullCount,
            numberUnique=counts.numberUnique,
        )


async def generate_box_plot(plot_name: str, column: Series, output_context: OutputContext) -> FileReference:
    async with output_context.get_plot(plot_name) as (axes, reference):
        axes.boxplot(column, notch=True)
    return reference


async def compute_best_fit_distribution(
    column: Series,
    config: FittingConfig,
    plot_name: str,
    max_elements: int,
    output_context: OutputContext,
    x_limits: Tuple[float, float],
):
    x_min, x_max = x_limits
    representatives: Union[Series, ndarray] = (
        column if column.size < max_elements else random_choice(column, max_elements)
    )

    loop = get_running_loop()
    fitter = Fitter(
        representatives,
        xmin=x_min,
        xmax=x_max,
        timeout=config.timeout.total_seconds(),
        bins=config.bins,
        distributions=config.distributions,
    )

    def runner():
        return fitter.fit(max_workers=config.workers)

    await loop.run_in_executor(None, runner)
    # According to documentation, this ony ever contains one entry.
    best_distribution_dict = fitter.get_best(method=config.error_function)
    distribution_name, distribution_parameters = next(iter(best_distribution_dict.items()))

    async with output_context.get_plot(plot_name) as (axes, reference):
        axes.set_xlim(x_min, x_max)
        axes.hist(representatives, bins=config.bins, range=x_limits, density=True)
        x = linspace(x_min, x_max, 2048)
        distribution = getattr(distributions, distribution_name)
        distribution_y = distribution.pdf(x, **distribution_parameters)
        axes.plot(x, distribution_y)

    return distribution_name, reference


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


def compute_counts(column: Series) -> BaseColumnCounts:
    non_null_count = column.count()
    null_count = count_nonzero(column.isnull())
    unique_count = column.unique().size
    return BaseColumnCounts(nonNullCount=non_null_count, nullCount=null_count, numberUnique=unique_count)


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


def _get_distribution_x_limits(
    minimum: float, maximum: float, lower_iqr: float, upper_iqr: float
) -> Tuple[float, float]:
    combined_min = minimum if lower_iqr < minimum else lower_iqr
    combined_max = maximum if upper_iqr > maximum else upper_iqr
    return combined_min, combined_max
