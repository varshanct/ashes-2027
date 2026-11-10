import sys

import numpy as np
import pandas as pd
import pytest

from cricket import forecast
from cricket.ingest import ingest_folder
from cricket.model_data import build_model_table, margin_in_wickets, runs_per_wicket, to_model_data
from test_26_forecast_command import fake_sampler, project  # noqa: F401  (project is a fixture)

ALIASES = pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]})


def test_runs_per_wicket():
    innings = pd.DataFrame({"runs": [300, 200], "wickets": [10, 10]})
    assert runs_per_wicket(innings) == 25


def test_margin_in_wickets_converts_runs_and_keeps_wickets():
    matches = pd.DataFrame({"margin_runs": [100.0, None, None], "margin_wkts": [None, 6.0, None]})
    result = margin_in_wickets(matches, rpw=25)
    assert result.iloc[0] == 4          # 100 runs at 25 runs per wicket
    assert result.iloc[1] == 6          # 6 wickets stays 6
    assert np.isnan(result.iloc[2])     # a draw has no margin


def test_model_table_adds_log_margin_only_when_asked(raw_dir):
    matches = ingest_folder(raw_dir)["matches"]
    assert "margin" not in build_model_table(matches, ALIASES).columns
    table = build_model_table(matches, ALIASES, rpw=6.0).set_index("match_id")
    assert table.loc["3001", "margin"] == pytest.approx(np.log1p(12 / 6))   # innings and 12 runs = 2 wickets
    assert table.loc["3002", "margin"] == pytest.approx(np.log1p(9))        # won by 9 wickets
    assert np.isnan(table.loc["3003", "margin"])                             # draw


def test_model_data_records_each_win_and_who_won(raw_dir):
    matches = ingest_folder(raw_dir)["matches"]
    data = to_model_data(build_model_table(matches, ALIASES, rpw=6.0))
    # Tests in date order: 3001 England (team1) won, 3002 Australia (team2) won, 3003 drawn.
    assert data.margin_idx.tolist() == [0, 1]
    assert data.margin_sign.tolist() == [1.0, -1.0]
    np.testing.assert_allclose(data.margin, [np.log1p(2), np.log1p(9)])


def test_model_data_without_margins_has_none(raw_dir):
    matches = ingest_folder(raw_dir)["matches"]
    data = to_model_data(build_model_table(matches, ALIASES))
    assert len(data.margin_idx) == 0 and len(data.margin) == 0


def run_forecast(monkeypatch, project, *extra):
    seen = {}

    def recording_sampler(data, draws, tune, chains, seed):
        seen["margins"] = len(data.margin_idx)
        return fake_sampler(data, draws, tune, chains, seed)

    monkeypatch.setattr(forecast, "sample_posterior", recording_sampler)
    argv = ["forecast", "--draws", "50", "--chains", "2", "--processed", str(project / "processed"),
            "--manual", str(project / "manual"), "--figures", str(project / "figures"),
            "--outputs", str(project / "outputs"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    forecast.main()
    return seen["margins"]


def test_forecast_uses_margins_by_default(monkeypatch, project, capsys):
    assert run_forecast(monkeypatch, project) == 2
    assert "Winning margins: 2 wins" in capsys.readouterr().out


def test_forecast_can_leave_margins_out(monkeypatch, project, capsys):
    assert run_forecast(monkeypatch, project, "--results-only") == 0
    assert "Winning margins: left out" in capsys.readouterr().out