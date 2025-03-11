import asyncio
import warnings
from datetime import timedelta
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from pandas import Series
from pydantic import BaseModel, Field

from edps.taskcontext import TaskContext

Distribution = scipy.stats.rv_continuous
DistributionParameters = Dict[str, float]

COMMON_DISTRIBUTIONS: List[Distribution] = [
    scipy.stats.cauchy,
    scipy.stats.chi2,
    scipy.stats.expon,
    scipy.stats.exponpow,
    scipy.stats.gamma,
    scipy.stats.lognorm,
    scipy.stats.norm,
    scipy.stats.powerlaw,
    scipy.stats.rayleigh,
    scipy.stats.uniform,
]


class _FittingError:
    def __init__(self, message: str):
        self.message = message


class FittingConfig(BaseModel):
    timeout: timedelta = Field(default=timedelta(seconds=30), description="Timeout to use for the fitting")
    min: Optional[float] = Field(default=None, description="Minimum value to include in fitting")
    max: Optional[float] = Field(default=None, description="Maximum value to include in fitting")
    max_samples: int = Field(
        default=int(1e6), description="Maximum number of values to use for determining the distribution of values."
    )


class Fitter:
    def __init__(self, data: Series, config: FittingConfig = FittingConfig()):
        self.timeout = config.timeout.total_seconds()

        self._data = data.replace([np.inf, -np.inf], np.nan)
        self._data = data.dropna()

        # Prepare the data array and its limits
        min: float = data.min() if config.min is None else config.min
        max: float = data.max() if config.max is None else config.max
        self._data = self._data[(self._data >= min) & (self._data <= max)]

        # Restrict number of value entries
        if len(self._data.index) > config.max_samples:
            self._data = self._data.sample(config.max_samples)

    async def fit(self, ctx: TaskContext) -> pd.DataFrame:
        r"""Loop over distributions and find the best parameter to fit the data for each."""
        async with asyncio.TaskGroup() as context:
            tasks = {
                distribution: context.create_task(
                    Fitter._fit_single_distribution(distribution, self._data, self.timeout)
                )
                for distribution in COMMON_DISTRIBUTIONS
            }

        results = pd.DataFrame.from_dict(
            {
                name: {"p_value": p_value, "params": params}
                async for name, p_value, params in self._await_results(ctx, tasks)
            },
            orient="index",
        )
        results.index.name = "distribution"
        results.sort_values(by="p_value", ascending=False, inplace=True)
        return results

    async def _await_results(
        self,
        ctx: TaskContext,
        tasks: Dict[Distribution, asyncio.Task[Union[np.ndarray, _FittingError]]],
    ) -> AsyncIterator[Tuple[str, np.float64, DistributionParameters]]:
        for distribution, task in tasks.items():
            parameters_or_error = await task
            if isinstance(parameters_or_error, _FittingError):
                ctx.logger.warning(parameters_or_error.message)
                warnings.warn(parameters_or_error.message)
            else:
                ks_stat, p_value = scipy.stats.kstest(self._data, distribution.name, tuple(parameters_or_error))
                ctx.logger.info(
                    "Fitted %s distribution, p-value=%.6f, KS statistic=%.6f", distribution.name, p_value, ks_stat
                )
                yield distribution.name, p_value, await self._convert_params_to_dict(distribution, parameters_or_error)

    @staticmethod
    async def _fit_single_distribution(
        distribution: Distribution, data: pd.Series, timeout_s: float
    ) -> Union[np.ndarray, _FittingError]:
        # Compute fitting params
        try:
            async with asyncio.timeout(timeout_s):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parameters = await asyncio.to_thread(distribution.fit, data.to_numpy())
                return np.array(parameters, dtype=float)
        except TimeoutError:
            return _FittingError(f"SKIPPED {distribution.name} distribution (taking more than {timeout_s} seconds)")
        except Exception as exception:
            return _FittingError(f"Error while processing {distribution.name} distribution: {str(exception)}")

    @staticmethod
    async def _convert_params_to_dict(distribution: Distribution, parameters: np.ndarray) -> DistributionParameters:
        param_names = (distribution.shapes + ", loc, scale").split(", ") if distribution.shapes else ["loc", "scale"]
        return dict(zip(param_names, parameters))

    async def get_best(self, ctx: TaskContext) -> Tuple[str, DistributionParameters]:
        fitting_result = await self.fit(ctx)
        distribution_name = fitting_result.index[0]
        return distribution_name, fitting_result.loc[distribution_name, "params"]
