import numpy as np
import pytest

from cricket.diagnostics import diagnose, effective_sample_size, split_rhat


def ar1(rho, chains=4, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, draws))
    for t in range(1, draws):
        x[:, t] = rho * x[:, t - 1] + rng.standard_normal(chains)
    return x


def test_rhat_near_one_for_good_chains():
    x = np.random.default_rng(0).standard_normal((4, 1000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)


def test_rhat_large_when_chains_disagree():
    x = np.random.default_rng(0).standard_normal((4, 1000))
    x[0] += 3     # one chain stuck somewhere else
    assert split_rhat(x) > 1.1


def test_rhat_catches_a_drifting_chain():
    x = np.random.default_rng(0).standard_normal((4, 1000))
    x[1] += np.linspace(0, 4, 1000)   # a chain that never settles
    assert split_rhat(x) > 1.05


def test_ess_close_to_draw_count_for_independent_draws():
    x = np.random.default_rng(0).standard_normal((4, 1000))
    assert effective_sample_size(x) == pytest.approx(4000, rel=0.15)


def test_ess_matches_theory_for_correlated_draws():
    # For an AR(1) chain the ESS is about N * (1 - rho) / (1 + rho).
    x = ar1(0.9)
    expected = 8000 * 0.1 / 1.9
    assert effective_sample_size(x) == pytest.approx(expected, rel=0.25)


def test_diagnose_reports_worst_element():
    rng = np.random.default_rng(0)
    good = rng.standard_normal((4, 500, 3))
    bad = good.copy()
    bad[0, :, 2] += 5
    table = diagnose({"good": good, "bad": bad}).set_index("parameter")
    assert table.loc["good", "elements"] == 3
    assert table.loc["good", "max_rhat"] < 1.01
    assert table.loc["bad", "max_rhat"] > 1.1
