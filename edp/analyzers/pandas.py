from asyncio import get_running_loop
from datetime import timedelta
from enum import Enum
from logging import getLogger
from multiprocessing import cpu_count
from typing import Dict, Hashable, Iterator, List, Tuple

from fitter import Fitter
from numpy import any as numpy_any
from numpy import count_nonzero, linspace
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    concat,
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
    DataSetType,
    DateTimeColumn,
    FileReference,
    Gap,
    NumericColumn,
    StringColumn,
    StructuredDataSet,
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
        "gompertz",
        "gumbel_l",
        "gumbel_r",
        "halfcauchy",
        "halfgennorm",
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
    timeout: timedelta = Field(default=timedelta(minutes=1), description="Timeout to use for the fitting")
    error_function: str = Field(
        default="sumsquare_error", description="Error function to use to measure performance of the fits"
    )
    bins: int = Field(default=100, description="Number of bins to use for the fitting")
    distributions: List[str] = Field(default_factory=_default_distributions, description="Distributions to try to fit")


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

_DATETIME_EARLIEST = "earliest"
_DATETIME_LATEST = "latest"
_DATETIME_ALL_ENTRIES_UNIQUE = "all-entries-are-unique"
_DATETIME_MONOTONIC_INCREASING = "monotonically-increasing"
_DATETIME_MONOTONIC_DECREASING = "monotonically-decreasing"
_DATETIME_TEMPORAL_CONSISTENCY = "temporal-consistency"
_DATETIME_GAPS = "gaps"


class _Distributions(str, Enum):
    SingleValue = "single value"
    TooSmallDataset = "dataset too small to determine distribution"


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

    @property
    def data_set_type(self):
        return DataSetType.structured

    async def analyze(self, output_context: OutputContext):
        row_count = len(self._data.index)
        self._logger.info("Started structured data analysis with dataset containing %d rows", row_count)
        columns_by_type = await self._determine_types()
        common_fields = await self._compute_common_fields()

        numeric_column_ids = columns_by_type[_ColumnType.Numeric]
        numeric_columns = self._data[numeric_column_ids]
        numeric_common_fields = common_fields.loc[numeric_column_ids]
        numeric_fields = await self._compute_numeric_fields(numeric_columns, numeric_common_fields)
        numeric_fields = concat([numeric_fields, numeric_common_fields], axis=1)

        datetime_ids = columns_by_type[_ColumnType.DateTime]
        datetime_columns = self._data[datetime_ids]
        datetime_common_fields = common_fields.loc[datetime_ids]
        datetime_fields = await self._compute_datetime_fields(datetime_columns)
        datetime_fields = concat([datetime_common_fields, datetime_fields], axis=1)

        string_ids = columns_by_type[_ColumnType.String]
        string_columns = self._data[string_ids]
        string_fields = common_fields.loc[string_ids]

        transformed_numeric_columns = [
            await self._transform_numeric_results(
                numeric_columns[name], _get_single_row(name, numeric_fields), output_context
            )
            for name in numeric_column_ids
        ]
        transformed_datetime_columns = [
            await self._transform_datetime_results(datetime_columns[name], _get_single_row(name, datetime_fields))
            for name in columns_by_type[_ColumnType.DateTime]
        ]
        transformed_string_columns = [
            await self._transform_string_results(string_columns[name], _get_single_row(name, string_fields))
            for name in columns_by_type[_ColumnType.String]
        ]
        return StructuredDataSet(
            name=self._file.output_reference,
            rowCount=row_count,
            numericColumns=transformed_numeric_columns,
            datetimeColumns=transformed_datetime_columns,
            stringColumns=transformed_string_columns,
        )

    async def _compute_common_fields(self) -> DataFrame:
        common_fields = DataFrame(index=self._data.columns)
        common_fields[_COMMON_NON_NULL] = self._data.count()
        common_fields[_COMMON_NULL] = self._data.isna().sum()
        common_fields[_COMMON_UNIQUE] = self._data.nunique(dropna=True)
        return common_fields

    async def _compute_numeric_fields(self, columns: DataFrame, common_fields: DataFrame) -> DataFrame:
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
        fields = concat(
            [
                fields,
                await _get_distributions(
                    columns,
                    concat([common_fields, fields], axis=1),
                    self._fitting_config,
                    self._distribution_threshold,
                    self._workers,
                ),
            ],
            axis=1,
        )
        return fields

    async def _compute_datetime_fields(self, columns: DataFrame) -> DataFrame:
        INTERVALS = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]
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
        computed[_DATETIME_TEMPORAL_CONSISTENCY] = _get_temporal_consistencies(columns, INTERVALS)
        computed[_DATETIME_GAPS] = _get_gaps(columns, INTERVALS)
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
        self, column: Series, computed_fields: Series, output_context: OutputContext
    ) -> NumericColumn:
        self._logger.debug('Transforming numeric column "%s" results to EDP', column.name)
        column_plot_base = self._file.output_reference + "_" + str(object=column.name)
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
        return column_result

    async def _transform_datetime_results(self, column: Series, computed_fields: Series) -> DateTimeColumn:
        self._logger.debug('Transforming datetime column "%s" results to EDP', column.name)

        return DateTimeColumn(
            name=str(column.name),
            nonNullCount=computed_fields[_COMMON_NON_NULL],
            nullCount=computed_fields[_COMMON_NULL],
            numberUnique=computed_fields[_COMMON_UNIQUE],
            earliest=computed_fields[_DATETIME_EARLIEST],
            latest=computed_fields[_DATETIME_LATEST],
            all_entries_are_unique=computed_fields[_DATETIME_ALL_ENTRIES_UNIQUE],
            monotonically_increasing=computed_fields[_DATETIME_MONOTONIC_INCREASING],
            monotonically_decreasing=computed_fields[_DATETIME_MONOTONIC_DECREASING],
            temporalConsistencies=computed_fields[_DATETIME_TEMPORAL_CONSISTENCY],
            gaps=computed_fields[_DATETIME_GAPS],
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
        axes.set_title(plot_name)
        axes.boxplot(column, notch=True, tick_labels=[str(column.name)])
    return reference


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


def _get_temporal_consistencies(columns: DataFrame, intervals: List[timedelta]) -> Series:
    # TODO: Vectorize this!
    return Series({name: _get_temporal_consistencies_for_column(column, intervals) for name, column in columns.items()})


def _get_temporal_consistencies_for_column(
    column: Series, intervals: List[timedelta]
) -> List[TemporalConsistency]:
    column.index = DatetimeIndex(column)
    return [_get_temporal_consistency(column, interval) for interval in intervals]


def _get_temporal_consistency(column: Series, interval: timedelta) -> TemporalConsistency:
    # TODO: Restrict to only the most abundant ones.
    abundances = column.resample(interval).count().unique()
    different_abundances = len(abundances)
    return TemporalConsistency(
        timeScale=interval,
        stable=(different_abundances == 1),
        differentAbundancies=different_abundances,
        abundances=abundances.tolist(),
    )


def _get_gaps(columns: DataFrame, intervals: List[timedelta]) -> Series:
    # TODO: Vectorize!
    return Series([_get_gaps_per_column(column, intervals) for name, column in columns.items()])


def _get_gaps_per_column(column: Series, intervals: List[timedelta]) -> Iterator[Gap]:
    deltas = column.sort_values().diff()
    for interval in intervals:
        over_interval_size = deltas > to_timedelta(interval)
        gap_count = count_nonzero(over_interval_size)
        yield Gap(timeScale=over_interval_size, numberOfGaps=gap_count)

def _get_outliers(column: DataFrame, lower_limit: Series, upper_limit: Series) -> Series:
    is_outlier = (column < lower_limit) | (column > upper_limit)
    return is_outlier.count()


async def _get_distributions(
    columns: DataFrame, fields: DataFrame, config: FittingConfig, distribution_threshold: int, workers: int
) -> DataFrame:

    values = [
        await _get_distribution(column, _get_single_row(name, fields), config, distribution_threshold, workers)
        for name, column in columns.items()
    ]

    data_frame = DataFrame(
        values,
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
        axes.set_title(plot_name)
        axes.set_xlim(x_min, x_max)
        axes.hist(column, bins=config.bins, range=x_limits, density=True, label=str(column.name))
        x = linspace(x_min, x_max, 2048)
        distribution = getattr(distributions, distribution_name)
        distribution_y = distribution.pdf(x, **distribution_parameters)
        axes.plot(x, distribution_y, label=distribution_name)
        axes.legend()
    return reference


def _get_single_row(row_name: str | Hashable, data_frame: DataFrame) -> Series:
    """This is mostly a convenience wrapper due to poor pandas stubs."""
    return data_frame.loc[str(row_name)]  # type: ignore


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
