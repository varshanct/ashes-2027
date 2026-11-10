import numpy as np
import pymc as pm

from cricket.bayes_model import PARAMETERS, build_model, fit
from cricket.model_data import ModelData
from cricket.predict import result_probs


def small_data(seed=0):
    rng = np.random.default_rng(seed)
    n = 40
    team_a = rng.integers(0, 3, n)
    team_b = (team_a + rng.integers(1, 3, n)) % 3
    years = rng.integers(0, 3, n)
    return ModelData(teams=["A", "B", "C"], years=[2024, 2025, 2026], venues=["V1", "V2"],
                     team_a=team_a, team_b=team_b, year_idx=years, venue_idx=rng.integers(0, 2, n),
                     home=rng.integers(-1, 2, n), decade=(years + 2024 - 2019) / 10,
                     result=rng.integers(0, 3, n))


def test_pymc_likelihood_matches_numpy_version():
    eta = np.array([-1.0, 0.0, 0.5, 2.0])
    width = np.array([0.8, 1.2, 0.5, 2.0])
    observed = np.array([0, 1, 2, 1])
    half = width / 2
    dist = pm.OrderedLogistic.dist(eta=eta, cutpoints=np.stack([-half, half], axis=-1))
    from_pymc = pm.logp(dist, observed).eval()
    from_numpy = np.log(result_probs(eta, width)[np.arange(4), observed])
    np.testing.assert_allclose(from_pymc, from_numpy, rtol=1e-6)


def test_model_has_every_parameter():
    model = build_model(small_data())
    for name in PARAMETERS:
        assert name in model.named_vars


def test_short_fit_returns_draws_of_the_right_shape():
    data = small_data()
    samples, divergences = fit(build_model(data), draws=30, tune=30, chains=2, cores=1, progressbar=False)
    assert samples["strength"].shape == (2, 30, 3, 3)       # chains, draws, years, teams
    assert samples["venue_effect"].shape == (2, 30, 2)
    assert samples["home_advantage"].shape == (2, 30)
    np.testing.assert_allclose(samples["strength"].sum(axis=3), 0.0, atol=1e-8)   # centred every year
    assert divergences >= 0