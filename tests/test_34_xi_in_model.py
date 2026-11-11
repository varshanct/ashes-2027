import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from cricket import rolling_backtest
from cricket.model_data import XI_SCALE, add_xi, build_model_table, to_model_data
from cricket.predict import result_probs
from cricket.rolling_backtest import backtest_year, forecast_matches, version_table
from test_26_forecast_command import fake_sampler
from test_30_rolling_backtest import ALIASES, made_up_matches


def xi_for(rows):
    """XI ratings from (match_id, team, bat_xi, bowl_xi) rows."""
    return pd.DataFrame(rows, columns=["match_id", "team", "bat_xi", "bowl_xi"])


def test_add_xi_gives_gaps_from_team1_point_of_view_in_units_of_ten_runs():
    table = pd.DataFrame({"match_id": ["1", "2"], "team1": ["A", "B"], "team2": ["B", "A"]})
    xi = xi_for([("1", "A", 36.0, 26.0), ("1", "B", 31.0, 31.0)])       # match 2 has no XI ratings
    out = add_xi(table, xi)
    assert out.loc[0, "xi_bat"] == pytest.approx((36 - 31) / XI_SCALE)    # A bats 5 runs better
    assert out.loc[0, "xi_bowl"] == pytest.approx((31 - 26) / XI_SCALE)   # A concedes 5 fewer: better
    assert out.loc[1, "xi_bat"] == 0 and out.loc[1, "xi_bowl"] == 0      # no ratings: treated as even


def test_model_data_carries_xi_gaps_only_when_present():
    matches, _ = made_up_matches()
    table = build_model_table(matches, ALIASES)
    assert len(to_model_data(table).xi_bat) == 0
    xi = xi_for([("1", "A", 40.0, 25.0), ("1", "B", 30.0, 30.0)])
    data = to_model_data(add_xi(table, xi))
    assert len(data.xi_bat) == len(table)
    assert data.xi_bat[0] == pytest.approx(1.0) and data.xi_bowl[0] == pytest.approx(0.5)


def test_forecast_adds_the_xi_effect_for_each_test():
    n = 40
    post = {"teams": ["A", "B"], "years": [2020], "venues": ["V"],
            "strength": np.zeros((n, 1, 2)), "sigma_drift": np.zeros(n), "sigma_venue": np.zeros(n),
            "home_advantage": np.zeros(n), "draw_base": np.full(n, np.log(0.8)), "draw_trend": np.zeros(n),
            "venue_effect": np.zeros((n, 1)), "xi_bat_effect": np.full(n, 0.3),
            "xi_bowl_effect": np.full(n, 0.2)}
    tests = pd.DataFrame({"team1": ["A"], "team2": ["B"], "home": [0], "venue_key": ["V"],
                          "xi_bat": [1.0], "xi_bowl": [2.0]})
    probs = forecast_matches(post, tests, 2021)
    np.testing.assert_allclose(probs[0], result_probs(0.3 * 1 + 0.2 * 2, 0.8))   # edge from the XIs alone


def test_each_version_sees_only_its_own_columns():
    table = pd.DataFrame({"result": [2], "margin": [1.0], "xi_bat": [0.5], "xi_bowl": [0.1]})
    assert list(version_table(table, "results").columns) == ["result"]
    assert list(version_table(table, "margins").columns) == ["result", "margin"]
    assert list(version_table(table, "xi").columns) == ["result", "xi_bat", "xi_bowl"]


def test_xi_version_learns_from_xi_gaps(monkeypatch):
    matches, innings = made_up_matches()
    xi = xi_for([(m, t, 30.0, 30.0) for m in "1234567" for t in "ABC"])
    full_table = add_xi(build_model_table(matches, ALIASES), xi)
    seen = []

    def recording_sampler(data, draws, tune, chains, seed):
        seen.append((len(data.result), len(data.margin_idx), len(data.xi_bat)))
        return fake_sampler(data, draws, tune, chains, seed)

    monkeypatch.setattr(rolling_backtest, "sample_posterior", recording_sampler)
    args = argparse.Namespace(draws=20, tune=20, chains=2, seed=1, versions=["results", "xi"])
    rows, checks = backtest_year(2021, matches, innings, ALIASES, full_table, args, xi)
    assert seen == [(4, 0, 0), (4, 0, 4)]               # the XI version gets an XI gap for each 2020 Test
    assert [c["version"] for c in checks] == ["Results only", "With XI ratings"]
    assert "log_loss_xi" in rows.columns


def test_command_asks_for_the_xi_ratings_first(tmp_path, monkeypatch):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    processed.mkdir()
    manual.mkdir()
    matches, innings = made_up_matches()
    matches.to_csv(processed / "matches.csv", index=False)
    innings.to_csv(processed / "innings.csv", index=False)
    ALIASES.to_csv(manual / "venue_aliases.csv", index=False)
    argv = ["rolling_backtest", "--processed", str(processed), "--manual", str(manual),
            "--figures", str(tmp_path / "figures")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="cricket.lineups"):
        rolling_backtest.main()