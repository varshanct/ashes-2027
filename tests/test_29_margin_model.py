from dataclasses import replace

import numpy as np
import pymc as pm
from scipy.stats import norm

from cricket.bayes_model import MARGIN_PARAMETERS, build_model, fit
from test_24_bayes_model import small_data


def with_margins(data, seed=1):
    rng = np.random.default_rng(seed)
    won = np.flatnonzero(data.result != 1)
    return replace(data, margin_idx=won, margin=rng.uniform(0.1, 2.5, len(won)),
                   margin_sign=np.where(data.result[won] == 2, 1.0, -1.0))


def test_margin_likelihood_matches_hand_formula():
    # A normal cut off at 0: density at x, divided by the chance of being above 0.
    mu, sigma = np.array([0.5, -0.3, 1.2]), np.array([0.8, 0.8, 0.5])
    x = np.array([0.2, 1.0, 2.0])
    from_pymc = pm.logp(pm.TruncatedNormal.dist(mu=mu, sigma=sigma, lower=0.0), x).eval()
    by_hand = norm.logpdf(x, mu, sigma) - norm.logsf(0.0, mu, sigma)
    np.testing.assert_allclose(from_pymc, by_hand, rtol=1e-6)


def test_margin_parameters_only_when_margins_are_given():
    without = build_model(small_data())
    with_them = build_model(with_margins(small_data()))
    for name in MARGIN_PARAMETERS:
        assert name not in without.named_vars
        assert name in with_them.named_vars


def test_short_fit_with_margins_returns_margin_draws():
    samples, _ = fit(build_model(with_margins(small_data())), draws=30, tune=30, chains=2, cores=1,
                     progressbar=False)
    for name in MARGIN_PARAMETERS:
        assert samples[name].shape == (2, 30)
    assert (samples["margin_slope"] > 0).all() and (samples["margin_sigma"] > 0).all()