import sys

import numpy as np
import pandas as pd
import pytest

from cricket import backtest
from cricket.backtest import base_rates, score, series_tests
from cricket.ingest import ingest_folder, save_tables
from test_26_forecast_command import fake_sampler


def table(rows):
    columns = ["match_id", "date_start", "team1", "team2", "home", "venue_key", "result"]
    return pd.DataFrame(rows, columns=columns)


def test_series_tests_picks_the_window_and_reads_results_from_the_home_side():
    t = table([
        ["1", "2019-07-24", "England", "Ireland", 1, "Lord's", 2],          # wrong opponent
        ["2", "2019-08-01", "England", "Australia", 1, "Edgbaston", 0],     # Australia won
        ["3", "2019-08-14", "Australia", "England", -1, "Lord's", 1],       # draw
        ["4", "2019-08-22", "Australia", "England", -1, "Headingley", 0],   # Australia (team1) lost: England won
        ["5", "2019-10-01", "England", "Australia", 1, "Kia Oval", 2],      # after the window
    ])
    tests = series_tests(t, "England", "Australia", "2019-08-01", "2019-09-16")
    assert tests["match_id"].tolist() == ["2", "3", "4"]
    assert tests["outcome"].tolist() == [2, 1, 0]
    assert tests["test_no"].tolist() == [1, 2, 3]
    assert tests["venue"].tolist() == ["Edgbaston", "Lord's", "Headingley"]


def test_base_rates_count_wins_from_the_home_side():
    t = table([
        ["1", "2020-01-01", "A", "B", 1, "V", 2],     # home side (team1) won
        ["2", "2020-01-02", "A", "B", -1, "V", 0],    # home side (team2) won
        ["3", "2020-01-03", "A", "B", 1, "V", 1],     # draw
        ["4", "2020-01-04", "A", "B", 1, "V", 0],     # away side won
        ["5", "2020-01-05", "A", "B", 0, "V", 2],     # neutral ground: left out
    ])
    np.testing.assert_allclose(base_rates(t), [0.5, 0.25, 0.25])


def test_base_rates_with_no_home_sides_are_equal_thirds():
    t = table([["1", "2020-01-01", "A", "B", 0, "V", 2]])
    np.testing.assert_allclose(base_rates(t), [1 / 3] * 3)


def test_score_of_a_certain_correct_forecast_and_of_equal_thirds():
    log_loss, brier = score([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]], [0, 1])
    np.testing.assert_allclose(log_loss, [0.0, np.log(3)])
    np.testing.assert_allclose(brier, [0.0, 2 / 3])


def test_score_punishes_confident_wrong_forecasts_most():
    loss, _ = score([[0.6, 0.2, 0.2], [0.9, 0.05, 0.05]], [2, 2])
    assert loss[1] > loss[0]


@pytest.fixture
def project(raw_dir, tmp_path):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    manual.mkdir()
    pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]}).to_csv(
        manual / "venue_aliases.csv", index=False)
    return tmp_path


def run(monkeypatch, project, sampler=fake_sampler):
    # The made-up matches: one finished Test in June 2023, then a win and a draw in July.
    monkeypatch.setattr(backtest, "SERIES", {"Made-up series": ("2023-07-01", "2023-08-31")})
    monkeypatch.setattr(backtest, "sample_posterior", sampler)
    argv = ["backtest", "--draws", "50", "--chains", "2", "--processed", str(project / "processed"),
            "--manual", str(project / "manual"), "--figures", str(project / "figures")]
    monkeypatch.setattr(sys, "argv", argv)
    backtest.main()


def test_model_is_trained_only_on_tests_before_the_series(monkeypatch, project):
    seen = {}

    def recording_sampler(data, draws, tune, chains, seed):
        seen["tests"] = len(data.result)
        return fake_sampler(data, draws, tune, chains, seed)

    run(monkeypatch, project, recording_sampler)
    assert seen["tests"] == 1


def test_command_saves_every_file_and_scores_each_test(monkeypatch, project):
    run(monkeypatch, project)
    figures = project / "figures"
    assert sorted(p.name for p in figures.iterdir()) == [
        "backtest.png", "backtest_scores.csv", "backtest_series.csv", "backtest_tests.csv"]
    tests = pd.read_csv(figures / "backtest_tests.csv")
    assert tests["result"].tolist() == ["Australia won", "Draw"]
    np.testing.assert_allclose(tests[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=0.002)
    np.testing.assert_allclose(tests["log_loss_thirds"], np.log(3), atol=0.001)
    scores = pd.read_csv(figures / "backtest_scores.csv")
    assert scores["series"].tolist() == ["Made-up series"] * 2 + ["Both series"] * 2
    log_loss = scores[scores["measure"] == "log loss"]
    np.testing.assert_allclose(log_loss["Equal thirds"], np.log(3), atol=0.001)
    series = pd.read_csv(figures / "backtest_series.csv")
    assert series["Actual"].tolist() == ["England 0-1 Australia"]