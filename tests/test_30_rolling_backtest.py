import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from cricket import rolling_backtest
from cricket.model_data import build_model_table
from cricket.predict import result_probs
from cricket.rolling_backtest import (backtest_year, forecast_matches, past_results_forecast,
                                      scores_by_year)
from test_26_forecast_command import fake_sampler

ALIASES = pd.DataFrame({"cricsheet_venue": ["A Ground"], "ground": ["A Ground"]})


def made_up_matches():
    """A plays at home in A City every time; 2020 is for training, 2021 is forecast."""
    def m(mid, date, team1, team2, result, winner=None, runs=None, wkts=None, inns=None, city="A City"):
        return {"match_id": mid, "date_start": date, "venue": city.replace("City", "Ground"), "city": city,
                "team1": team1, "team2": team2, "result_type": result, "winner": winner,
                "margin_runs": runs, "margin_wkts": wkts, "margin_innings": inns}
    matches = pd.DataFrame([
        m("1", "2020-01-10", "A", "B", "win", "A", runs=50),
        m("2", "2020-03-10", "A", "B", "draw"),
        m("3", "2020-06-10", "B", "A", "win", "B", wkts=5, city="B City"),
        m("4", "2020-09-10", "B", "A", "win", "A", runs=20, inns=1, city="B City"),
        m("5", "2021-02-10", "A", "B", "win", "A", wkts=3),
        m("6", "2021-05-10", "B", "A", "draw", city="B City"),
        m("7", "2021-07-10", "A", "C", "win", "A", runs=100),     # C has never played before: skipped
    ])
    innings = pd.DataFrame({"match_id": ["1", "1", "3", "3", "5"], "runs": [300, 250, 200, 210, 280],
                            "wickets": [10, 10, 10, 5, 10]})
    return matches, innings


def test_forecast_matches_matches_hand_calculation():
    n = 50
    post = {"teams": ["A", "B"], "years": [2020], "venues": ["V"],
            "strength": np.zeros((n, 1, 2)), "sigma_drift": np.zeros(n), "sigma_venue": np.zeros(n),
            "home_advantage": np.full(n, 0.5), "draw_base": np.full(n, np.log(0.8)),
            "draw_trend": np.zeros(n), "venue_effect": np.zeros((n, 1))}
    tests = pd.DataFrame({"team1": ["A", "B"], "team2": ["B", "A"], "home": [1, -1],
                          "venue_key": ["V", "V"]})
    probs = forecast_matches(post, tests, 2021)
    np.testing.assert_allclose(probs[0], result_probs(0.5, 0.8))      # A at home: edge +0.5
    np.testing.assert_allclose(probs[1], probs[0][::-1])               # same Test seen from the away side


def test_past_results_forecast_turns_round_for_the_away_side():
    # Home win, home win, draw, away win.
    train = pd.DataFrame({"home": [1, -1, 1, 1], "result": [2, 0, 1, 0]})
    probs = past_results_forecast(train, pd.Series([1, -1, 0]))
    np.testing.assert_allclose(probs[0], [0.25, 0.25, 0.5])     # team1 at home
    np.testing.assert_allclose(probs[1], [0.5, 0.25, 0.25])     # team1 away
    np.testing.assert_allclose(probs[2], [0.375, 0.25, 0.375])  # neutral ground


def test_backtest_year_uses_only_earlier_tests_and_skips_new_teams(monkeypatch):
    matches, innings = made_up_matches()
    full_table = build_model_table(matches, ALIASES)
    seen = []

    def recording_sampler(data, draws, tune, chains, seed):
        seen.append((len(data.result), len(data.margin_idx)))
        return fake_sampler(data, draws, tune, chains, seed)

    monkeypatch.setattr(rolling_backtest, "sample_posterior", recording_sampler)
    args = argparse.Namespace(draws=20, tune=20, chains=2, seed=1, versions=["results", "margins"])
    rows, checks = backtest_year(2021, matches, innings, ALIASES, full_table, args)
    # Four 2020 Tests to learn from; the results-only fit ignores the three margins, the other uses them.
    assert seen == [(4, 0), (4, 3)]
    assert rows["match_id"].tolist() == ["5", "6"]
    assert [c["version"] for c in checks] == ["Results only", "With margins"]
    np.testing.assert_allclose(rows["log_loss_thirds"], np.log(3))


def test_scores_by_year_averages_each_year_and_all_together():
    tests = pd.DataFrame({"year": [2020, 2020, 2021]})
    names = rolling_backtest.forecasts(["results", "margins"])
    for key in names:
        tests[f"log_loss_{key}"] = [1.0, 0.5, 2.0]
        tests[f"brier_{key}"] = [0.6, 0.4, 0.8]
    scores = scores_by_year(tests, names)
    assert scores.loc[2020, "tests"] == 2
    assert scores.loc[2020, "With margins log loss"] == pytest.approx(0.75)
    assert scores.loc["All years", "With margins log loss"] == pytest.approx(3.5 / 3)
    assert scores.loc["All years", "tests"] == 3


def test_command_saves_every_file(monkeypatch, tmp_path, capsys):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    processed.mkdir()
    manual.mkdir()
    matches, innings = made_up_matches()
    matches.to_csv(processed / "matches.csv", index=False)
    innings.to_csv(processed / "innings.csv", index=False)
    ALIASES.to_csv(manual / "venue_aliases.csv", index=False)
    monkeypatch.setattr(rolling_backtest, "sample_posterior", fake_sampler)
    argv = ["rolling_backtest", "--first", "2021", "--last", "2021", "--versions", "results,margins",
            "--draws", "20", "--chains", "2",
            "--processed", str(processed), "--manual", str(manual), "--figures", str(tmp_path / "figures")]
    monkeypatch.setattr(sys, "argv", argv)
    rolling_backtest.main()
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "rolling_backtest.png", "rolling_backtest_checks.csv", "rolling_backtest_scores.csv",
        "rolling_backtest_tests.csv"]
    out = capsys.readouterr().out
    assert "skipped 1 involving a team with no earlier Tests" in out
    assert "Average log loss per Test" in out