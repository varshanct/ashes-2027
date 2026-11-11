import numpy as np
import pandas as pd

from cricket.charts import plot_forecast, plot_strength


def test_strength_chart_is_saved(tmp_path):
    rng = np.random.default_rng(0)
    strength = rng.normal(0, 0.2, size=(200, 5, 3))
    path = plot_strength(strength, [2022, 2023, 2024, 2025, 2026], ["Australia", "England", "India"],
                         tmp_path / "strength.png", "Test title")
    assert path.exists() and path.stat().st_size > 10_000


def test_forecast_chart_is_saved(tmp_path):
    rows = pd.DataFrame({"label": ["Test 1 model", "Test 1 mine"], "p_home": [0.4, 0.42],
                         "p_draw": [0.2, 0.15], "p_away": [0.4, 0.43]})
    path = plot_forecast(rows, tmp_path / "forecast.png", "Test title")
    assert path.exists() and path.stat().st_size > 10_000
