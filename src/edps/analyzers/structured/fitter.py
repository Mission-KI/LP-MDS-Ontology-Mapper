import asyncio
import warnings
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from pandas import Series
from pydantic import BaseModel, Field

from edps.taskcontext import TaskContext
from edps.types import DistributionConfig

Distribution = scipy.stats.rv_continuous
DistributionParameters = Dict[str, float]

COMMON_DISTRIBUTIONS: List[Distribution] = [
    scipy.stats.cauchy,
    scipy.stats.exponpow,
    scipy.stats.gamma,
    scipy.stats.norm,
    scipy.stats.powerlaw,
    scipy.stats.rayleigh,
    scipy.stats.uniform,
    scipy.stats.maxwell,
]

_KEY_PARAMS = "parameters"
_KEY_P_VALUE = "p_value"
_KEY_KS_STAT = "ks_stat"
_KEY_ERROR = "error"


class Limits(BaseModel):
    min: Optional[float] = Field(default=None, description="Minimum value to include in fitting")
    max: Optional[float] = Field(default=None, description="Maximum value to include in fitting")


class FittingError(RuntimeError):
    """Error raised, when no distribution could be fitted properly."""


@dataclass
class _PerDistributionResult:
    distribution: Distribution
    p_value: float
    ks_stat: float
    parameters: DistributionParameters


async def fit_distributions(ctx: TaskContext, data: Series, limits: Limits = Limits()) -> pd.DataFrame:
    config = ctx.config.structured_config.distribution
    data = _prepare_data(config, data, limits=limits)

    ctx.logger.debug('Starting to fit %d distributions over column "%s"', len(COMMON_DISTRIBUTIONS), data.name)
    async with asyncio.TaskGroup() as context:
        tasks = {
            distribution: context.create_task(
                _fit_single_distribution(distribution, data, config.timeout.total_seconds())
            )
            for distribution in COMMON_DISTRIBUTIONS
        }

    results = pd.DataFrame.from_dict(
        {
            name: {_KEY_P_VALUE: p_value, _KEY_KS_STAT: ks_stat, _KEY_PARAMS: params, _KEY_ERROR: error}
            async for name, p_value, ks_stat, params, error in _await_results(ctx, tasks)
        },
        orient="index",
        columns=[_KEY_P_VALUE, _KEY_KS_STAT, _KEY_PARAMS, _KEY_ERROR],
    )
    results.index.name = "distribution"
    results.sort_values(by=_KEY_KS_STAT, ascending=True, inplace=True)
    ctx.logger.debug('Finished fitting distributions over column "%s"', data.name)
    return results


async def fit_best_distribution(
    ctx: TaskContext, data: Series, limits: Limits = Limits()
) -> Tuple[str, DistributionParameters]:
    fitting_result = await fit_distributions(ctx, data, limits)

    fitting_result = fitting_result[fitting_result[_KEY_ERROR].isna()]
    if fitting_result.empty:
        raise FittingError("No distribution could be fitted without error")

    best_distribution = str(fitting_result[_KEY_KS_STAT].idxmin())
    best_row = fitting_result.loc[best_distribution]
    parameters = cast(DistributionParameters, best_row[_KEY_PARAMS])
    p_value = cast(float, best_row[_KEY_P_VALUE])
    ks_stat = cast(float, best_row[_KEY_KS_STAT])

    ctx.logger.info(
        'Found best fit for column "%s" to be distribution "%s" with attributes %s: p_value=%.6f, ks_stat=%.2f',
        data.name,
        best_distribution,
        parameters,
        p_value,
        ks_stat,
    )
    return best_distribution, parameters


def _prepare_data(config: DistributionConfig, data: Series, limits: Limits):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Prepare the data array and its limits
    min: float = data.min() if limits.min is None else limits.min
    max: float = data.max() if limits.max is None else limits.max
    data = data[(data >= min) & (data <= max)]

    # Restrict number of value entries
    if len(data.index) > config.max_samples:
        data = data.sample(config.max_samples)

    return data


async def _await_results(
    ctx: TaskContext,
    tasks: Dict[Distribution, asyncio.Task[Union[_PerDistributionResult, FittingError]]],
) -> AsyncIterator[Tuple[str, float, float, Optional[DistributionParameters], Optional[FittingError]]]:
    for distribution, task in tasks.items():
        result_or_error = await task
        if isinstance(result_or_error, FittingError):
            ctx.logger.debug("Fitting %s failed: %s", distribution.name, str(result_or_error))
            yield distribution.name, 0.0, float("inf"), None, result_or_error
        else:
            ctx.logger.debug(
                "Fitted %s distribution, p-value=%.6f, KS statistic=%.6f",
                distribution.name,
                result_or_error.p_value,
                result_or_error.ks_stat,
            )
            yield (
                distribution.name,
                result_or_error.p_value,
                result_or_error.ks_stat,
                result_or_error.parameters,
                None,
            )


def _convert_params_to_dict(
    distribution: Distribution, parameters: Tuple[np.float64 | float, ...]
) -> DistributionParameters:
    param_names = (distribution.shapes + ", loc, scale").split(", ") if distribution.shapes else ["loc", "scale"]
    return dict(zip(param_names, parameters))


def _fit(distribution: Distribution, data: np.ndarray) -> _PerDistributionResult:
    params = distribution.fit(data)
    ks_stat, p_value = scipy.stats.ks_1samp(data, distribution.cdf, args=params, method="exact", nan_policy="omit")
    return _PerDistributionResult(
        distribution=distribution,
        p_value=p_value,
        ks_stat=ks_stat,
        parameters=_convert_params_to_dict(distribution, params),
    )


async def _fit_single_distribution(
    distribution: Distribution, data: pd.Series, timeout_s: float
) -> Union[_PerDistributionResult, FittingError]:
    try:
        async with asyncio.timeout(timeout_s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parameters = await asyncio.to_thread(_fit, distribution, data.to_numpy())
            return parameters
    except TimeoutError:
        return FittingError(f"SKIPPED {distribution.name} distribution (taking more than {timeout_s} seconds)")
    except Exception as exception:
        return FittingError(f"Error while processing {distribution.name} distribution: {str(exception)}")
