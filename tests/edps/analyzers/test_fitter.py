import math
from typing import Dict, Tuple

import pytest
import scipy
from pandas import Series

from edps.analyzers.pandas.fitter import (
    COMMON_DISTRIBUTIONS,
    Distribution,
    DistributionParameters,
    Fitter,
    FittingError,
)

N_SAMPLES = 25000
LOC = 10.0
SCALE = 20.0
A = 2.0
B = 5.0
NOISE_AMPLITUDE = 0.1 * SCALE

REL_TOL = 0.15


@pytest.fixture(scope="session")
def static_random_seed():
    return 1


@pytest.fixture(scope="session")
def noise(static_random_seed):
    return scipy.stats.distributions.uniform(loc=0.0, scale=NOISE_AMPLITUDE).rvs(N_SAMPLES, static_random_seed)


TEST_PARAMETERS: Dict[Distribution, DistributionParameters] = {
    scipy.stats.cauchy: dict(
        loc=LOC,
        scale=SCALE,
    ),
    scipy.stats.exponpow: dict(loc=LOC, scale=SCALE, b=B),
    scipy.stats.gamma: dict(loc=LOC, scale=SCALE, a=A),
    scipy.stats.norm: dict(
        loc=LOC,
        scale=SCALE,
    ),
    scipy.stats.powerlaw: dict(loc=LOC, scale=SCALE, a=A),
    scipy.stats.rayleigh: dict(
        loc=LOC,
        scale=SCALE,
    ),
    scipy.stats.uniform: dict(
        loc=LOC,
        scale=SCALE,
    ),
    scipy.stats.maxwell: dict(
        loc=LOC,
        scale=SCALE,
    ),
}


@pytest.fixture
def noisy_distribution(
    distribution: Distribution, static_random_seed, noise
) -> Tuple[Distribution, DistributionParameters, Series]:
    parameters = TEST_PARAMETERS[distribution]
    data = distribution.rvs(**parameters, size=N_SAMPLES, random_state=static_random_seed)
    noisy_data = data + noise
    return distribution, parameters, Series(noisy_data, name=distribution.name)


def test_every_distribution_has_1_to_1_params():
    assert set(TEST_PARAMETERS.keys()) == set(COMMON_DISTRIBUTIONS)


@pytest.mark.parametrize(
    "distribution", COMMON_DISTRIBUTIONS, ids=[distribution.name for distribution in COMMON_DISTRIBUTIONS]
)
def test_fit_distribution(noisy_distribution):
    distribution, parameters, data = noisy_distribution
    result = Fitter._fit(distribution, data.to_numpy())
    assert result.distribution.name == distribution.name
    for parameter_name, parameter_value in parameters.items():
        assert math.isclose(result.parameters[parameter_name], parameter_value, rel_tol=REL_TOL)


@pytest.mark.parametrize(
    "distribution", COMMON_DISTRIBUTIONS, ids=[distribution.name for distribution in COMMON_DISTRIBUTIONS]
)
async def test_detect_all_distributions(ctx, noisy_distribution):
    distribution, parameters, data = noisy_distribution
    fitter = Fitter(data, ctx)
    name, params = await fitter.get_best()
    assert name == distribution.name, f'Expected to detect "{distribution.name}", got "{name}".'
    for parameter_name, parameter_value in parameters.items():
        assert math.isclose(params[parameter_name], parameter_value, rel_tol=REL_TOL)


async def test_no_distribution_fits(ctx, monkeypatch):
    data = Series([1, 2, 3, 4, 5, 6])
    fitter = Fitter(data, ctx)

    async def dummy_fit_single_distribution(distribution, data, timeout_s):
        return FittingError("DummyError")

    monkeypatch.setattr(Fitter, "_fit_single_distribution", dummy_fit_single_distribution)
    with pytest.raises(FittingError):
        await fitter.get_best()
