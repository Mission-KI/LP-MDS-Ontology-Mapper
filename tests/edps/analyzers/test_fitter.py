import numpy as np
from pandas import Series

from edps.analyzers.pandas.fitter import Fitter
from edps.importers.structured import csv_import_dataframe


async def test_detect_norm(ctx):
    np.random.seed(1)
    n_samples = 25000
    data = np.random.normal(loc=10.0, scale=20.0, size=n_samples)

    noise = np.random.uniform(low=-0.05, high=0.05, size=n_samples)
    noisy_data = data * (1 + noise)
    series = Series(noisy_data)

    fitter = Fitter(series, ctx)
    name, params = await fitter.get_best(ctx)
    assert name == "norm"
    assert list(params.keys()) == ["loc", "scale"]
    assert abs(params.get("loc", 0.0) - 10.12) < 0.01
    assert abs(params.get("scale", 0.0) - 20.01) < 0.01


async def test_detect_chi_square(ctx):
    np.random.seed(1)
    n_samples = 25000
    data = np.random.chisquare(df=10.0, size=n_samples)

    noise = np.random.uniform(low=-0.05, high=0.05, size=n_samples)
    noisy_data = data * (1 + noise)
    series = Series(noisy_data)

    fitter = Fitter(series, ctx)
    name, params = await fitter.get_best(ctx)
    assert name == "chi2"
    assert list(params.keys()) == ["df", "loc", "scale"]
    assert abs(params.get("df", 0.0) - 9.84) < 0.01
    assert abs(params.get("loc", 0.0) - 0.03) < 0.01
    assert abs(params.get("scale", 0.0) - 1.01) < 0.01


async def test_detect_gamma(ctx):
    np.random.seed(1)
    n_samples = 25000
    data = np.random.gamma(shape=20.0, scale=20.0, size=n_samples)

    noise = np.random.uniform(low=-0.05, high=0.05, size=n_samples)
    noisy_data = data * (1 + noise)
    series = Series(noisy_data)

    fitter = Fitter(series, ctx)
    name, params = await fitter.get_best(ctx)
    assert name == "gamma"
    assert list(params.keys()) == ["a", "loc", "scale"]
    assert abs(params.get("a", 0.0) - 19.07) < 0.01
    assert abs(params.get("loc", 0.0) - 5.05) < 0.01
    assert abs(params.get("scale", 0.0) - 20.69) < 0.01


async def test_same_as_original_fitter(path_data_test_csv, ctx):
    data = await csv_import_dataframe(ctx, path_data_test_csv)
    column = data["aufenthalt"].astype(int)

    fitter = Fitter(column, ctx)
    name, params = await fitter.get_best(ctx)
    assert name == "lognorm"
    assert list(params.keys()) == ["s", "loc", "scale"]
    assert abs(params.get("s", 0.0) - 0.98) < 0.01
    assert abs(params.get("loc", 0.0) - -599.22) < 0.01
    assert abs(params.get("scale", 0.0) - 6846.56) < 0.01
