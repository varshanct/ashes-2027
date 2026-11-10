import sys

import numpy as np
import pandas as pd
import pytest

from cricket import forecast
from cricket.ingest import ingest_folder, save_tables

GROUNDS = ["Trent Bridge", "Lord's", "Edgbaston", "Utilita Bowl", "Kia Oval"]


def fake_sampler(data, draws, tune, chains, seed):
    """Stands in for PyMC: random draws of the right shapes, so the command can be tested quickly."""
    rng = np.random.default_rng(seed)
    years, teams, venues = len(data.years), len(data.teams), len(data.venues)
    strength = rng.normal(0, 0.2, size=(chains, draws, years, teams))
    strength -= strength.mean(axis=3, keepdims=True)
    scalar = lambda mean: rng.normal(mean, 0.05, size=(chains, draws))  # noqa: E731
    samples = {"strength": strength, "sigma_start": np.abs(scalar(0.5)), "sigma_drift": np.abs(scalar(0.1)),
               "home_advantage": scalar(0.3), "draw_base": scalar(-0.2), "draw_trend": scalar(-0.3),
               "sigma_venue": np.abs(scalar(0.3)), "venue_effect": rng.normal(0, 0.3, size=(chains, draws, venues))}
    return samples, 0


@pytest.fixture
def project(raw_dir, tmp_path):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    manual.mkdir()
    pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]}).to_csv(
        manual / "venue_aliases.csv", index=False)
    pd.DataFrame({"test_no": [1, 2, 3, 4, 5], "venue": GROUNDS}).to_csv(manual / "fixtures_2027.csv", index=False)
    pd.DataFrame({"test_no": [1, 2, 3, 4, 5], "p_england": [0.4] * 5, "p_draw": [0.2] * 5,
                  "p_australia": [0.4] * 5}).to_csv(manual / "my_forecast.csv", index=False)
    return tmp_path


def run(monkeypatch, project):
    monkeypatch.setattr(forecast, "sample_posterior", fake_sampler)
    argv = ["forecast", "--draws", "50", "--chains", "2", "--processed", str(project / "processed"),
            "--manual", str(project / "manual"), "--figures", str(project / "figures"),
            "--outputs", str(project / "outputs")]
    monkeypatch.setattr(sys, "argv", argv)
    forecast.main()


def test_command_saves_every_file(monkeypatch, project):
    run(monkeypatch, project)
    assert sorted(p.name for p in (project / "figures").iterdir()) == [
        "forecast_2027.csv", "forecast_2027.png", "sampler_checks.csv", "series_2027.csv", "team_strength.png"]
    assert (project / "outputs" / "posterior_draws.npz").exists()


def test_forecast_file_has_one_row_per_test_adding_to_one(monkeypatch, project):
    run(monkeypatch, project)
    tests = pd.read_csv(project / "figures" / "forecast_2027.csv")
    assert tests["venue"].tolist() == GROUNDS
    np.testing.assert_allclose(tests[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=0.002)


def test_series_file_compares_three_forecasts(monkeypatch, project):
    run(monkeypatch, project)
    series = pd.read_csv(project / "figures" / "series_2027.csv")
    assert series["forecast"].tolist() == ["Model", "Model, if Tests were independent", "My forecast"]
    total = series["England win series"] + series["Drawn series"] + series["Australia win series"]
    np.testing.assert_allclose(total, 1.0, atol=0.002)


def test_printout_shows_checks_and_forecast(monkeypatch, project, capsys):
    run(monkeypatch, project)
    out = capsys.readouterr().out
    for heading in ["Sampler checks", "Divergent transitions: 0", "Model parameters", "Strength in",
                    "Ground effect", "Forecast for each Test", "Series chances"]:
        assert heading in out