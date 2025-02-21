import functools
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from pandas import Series
from pydantic import BaseModel, Field
from scipy.stats import entropy as kl_div
from scipy.stats import kstest

from edps.task import TaskContext


def get_distributions() -> List[str]:
    return [attr for attr in dir(scipy.stats) if hasattr(getattr(scipy.stats, attr), "fit")]


def get_common_distributions() -> List[str]:
    # To avoid error due to changes in scipy
    common: List[str] = [
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "rayleigh",
        "uniform",
    ]
    return [x for x in common if x in get_distributions()]


class FittingConfig(BaseModel):
    timeout: timedelta = Field(default=timedelta(seconds=30), description="Timeout to use for the fitting")
    error_function: str = Field(
        default="sumsquare_error",
        description="Error function to use to measure performance of the fits",
    )
    bins: int = Field(default=100, description="Number of bins to use for the fitting")
    distributions: List[str] = Field(
        default_factory=get_common_distributions,
        description="Distributions to try to fit",
    )


class FittingErrors(NamedTuple):
    sq_error: np.float64 = np.float64(np.inf)
    aic: np.float64 = np.float64(np.inf)
    bic: np.float64 = np.float64(np.inf)
    kullback_leibler: np.float64 = np.float64(np.inf)
    ks_stat: np.float64 = np.float64(np.inf)
    ks_pval: np.float64 = np.float64(np.inf)


class FittingResult(NamedTuple):
    params: Tuple[np.float64, ...] = ()
    pdf_fitted: np.ndarray = np.array([])
    errors: FittingErrors = FittingErrors()


class Fitter:
    def __init__(
        self,
        data: Series,
        distributions: List[str],
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        bins: int = 100,
        timeout: float = 30.0,
    ):
        self.timeout = timeout
        self.distributions = distributions

        # Prepare the data array and its limits
        self._alldata = np.asarray(data)
        self.x_min = self._alldata.min() if x_min is None else x_min
        self.x_max = self._alldata.max() if x_max is None else x_max
        self._trim_data()

        self.bins = bins
        self._update_data_pdf()

        # Containers to store fitting results
        self.fitted_param: Dict[str, Tuple[np.float64, ...]] = {}
        self.fitted_pdf: Dict[str, np.ndarray] = {}
        self._fitted_errors: Dict[str, np.float64] = {}
        self._aic: Dict[str, np.float64] = {}
        self._bic: Dict[str, np.float64] = {}
        self._kldiv: Dict[str, np.float64] = {}
        self._ks_stat: Dict[str, np.float64] = {}
        self._ks_pval: Dict[str, np.float64] = {}

        self.df_errors = pd.DataFrame()

    def _trim_data(self):
        self._data = self._alldata[np.logical_and(self._alldata >= self.x_min, self._alldata <= self.x_max)]

    def _update_data_pdf(self):
        # Histogram returns X with N+1 values. So, we rearrange the X output into only N
        self.y, self.x = np.histogram(self._data, bins=self.bins, density=True)
        self.x = np.asarray([(this + self.x[i + 1]) / 2.0 for i, this in enumerate(self.x[0:-1])])

    def fit(self, ctx: TaskContext):
        r"""Loop over distributions and find the best parameter to fit the data for each

        When a distribution is fitted onto the data, we populate a set of
        dataframes:

            - :attr:`df_errors`  :sum of the square errors between the data and the fitted
              distribution i.e., :math:`\sum_i \left( Y_i - pdf(X_i) \right)^2`
            - :attr:`fitted_param` : the parameters that best fit the data
            - :attr:`fitted_pdf` : the PDF generated with the parameters that best fit the data

        Indices of the dataframes contains the name of the distribution.

        """
        results = []
        with ProcessPoolExecutor(max_workers=len(self.distributions)) as executor:
            futures = {
                dist_name: executor.submit(
                    functools.partial(
                        Fitter._fit_single_distribution,
                        data=self._data,
                        x=self.x,
                        y=self.y,
                    ),
                    dist_name,
                )
                for dist_name in self.distributions
            }

            for dist_name, future in futures.items():
                try:
                    result = future.result(timeout=self.timeout)

                    ctx.logger.info(f"Fitted {dist_name} distribution, error={round(result.errors.sq_error, 6)}")
                    results.append((dist_name, result))
                except TimeoutError:
                    ctx.logger.warning(f"SKIPPED {dist_name} distribution (taking more than {self.timeout} seconds)")
                    future.cancel()
                    results.append((dist_name, FittingResult()))
                except Exception as exception:
                    ctx.logger.warning(f"Error while processing {dist_name} distribution: {str(exception)}")
                    results.append((dist_name, FittingResult()))

        for dist_name, values in results:
            self.fitted_param[dist_name] = values.params
            self.fitted_pdf[dist_name] = values.pdf_fitted
            self._fitted_errors[dist_name] = values.errors.sq_error
            self._aic[dist_name] = values.errors.aic
            self._bic[dist_name] = values.errors.bic
            self._kldiv[dist_name] = values.errors.kullback_leibler
            self._ks_stat[dist_name] = values.errors.ks_stat
            self._ks_pval[dist_name] = values.errors.ks_pval

        self.df_errors = pd.DataFrame(
            {
                "sumsquare_error": self._fitted_errors,
                "aic": self._aic,
                "bic": self._bic,
                "kl_div": self._kldiv,
                "ks_statistic": self._ks_stat,
                "ks_pvalue": self._ks_pval,
            }
        )
        self.df_errors.sort_index(inplace=True)

    @staticmethod
    def _fit_single_distribution(
        dist_name: str,
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> FittingResult:
        # Compute fitting params
        dist_model = getattr(scipy.stats, dist_name)
        params = dist_model.fit(data)

        # Compute probability density function
        pdf_fitted = dist_model.pdf(x, *params)

        # Compute fitting errors
        errors = Fitter._compute_fitting_errors(dist_model, pdf_fitted, params, data, x, y)

        return FittingResult(
            params=params,
            pdf_fitted=pdf_fitted,
            errors=errors,
        )

    @staticmethod
    def _compute_fitting_errors(
        dist_model: scipy.stats.rv_continuous,
        pdf_fitted: np.ndarray,
        params: Tuple[np.float64, ...],
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> FittingErrors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in exp")
            warnings.filterwarnings("ignore", message="overflow encountered in log")
            warnings.filterwarnings("ignore", message="underflow encountered in log")
            warnings.filterwarnings("ignore", message="divide by zero encountered in")
            warnings.filterwarnings("ignore", message="invalid value encountered")

            # Sum of squared errors
            sq_error = np.sum((pdf_fitted - y) ** 2)

            # Calculate information criteria
            log_lik = np.sum(dist_model.logpdf(x, *params))
            k = len(params)
            n = len(data)
            aic = 2 * k - 2 * log_lik

            # Special case of gaussian distribution
            # bic = n * np.log(sq_error / n) + k * np.log(n)
            # general case:
            bic = k * np.log(n) - 2 * log_lik

            # Calculate kullback leibler divergence
            kullback_leibler = np.float64(kl_div(pdf_fitted, y))

            # Calculate goodness-of-fit statistic
            dist_fitted = dist_model(*params)
            ks_stat, ks_pval = kstest(data, dist_fitted.cdf)

        return FittingErrors(
            sq_error,
            aic,
            bic,
            kullback_leibler,
            ks_stat,
            ks_pval,
        )

    def get_best(self, method: str = "sumsquare_error") -> Dict[str, Dict[str, np.float64]]:
        # self.df_errors should be sorted, so then us take the first one as the best
        best = self.df_errors.sort_values(method).iloc[0]
        best_name = str(best.name)
        params = self.fitted_param[best_name]

        dist = getattr(scipy.stats, best_name)
        if dist.shapes:
            param_names = (dist.shapes + ", loc, scale").split(", ")
        else:
            param_names = ["loc", "scale"]

        return {best_name: dict(zip(param_names, params))}
