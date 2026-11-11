from dataclasses import replace

import numpy as np

from cricket.bayes_model import XI_PARAMETERS, build_model, fit
from test_24_bayes_model import small_data


def with_xi(data, seed=2):
    rng = np.random.default_rng(seed)
    n = len(data.result)
    return replace(data, xi_bat=rng.normal(0, 0.5, n), xi_bowl=rng.normal(0, 0.5, n))


def test_xi_parameters_only_when_xi_gaps_are_given():
    without = build_model(small_data())
    with_them = build_model(with_xi(small_data()))
    for name in XI_PARAMETERS:
        assert name not in without.named_vars
        assert name in with_them.named_vars


def test_short_fit_with_xi_returns_xi_draws():
    samples, _ = fit(build_model(with_xi(small_data())), draws=30, tune=30, chains=2, cores=1,
                     progressbar=False)
    for name in XI_PARAMETERS:
        assert samples[name].shape == (2, 30)