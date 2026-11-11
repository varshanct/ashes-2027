import pandas as pd

from cricket.charts import plot_ground_profiles, plot_results

GROUNDS = ["Trent Bridge", "Lord's", "Edgbaston", "Utilita Bowl", "Kia Oval"]


def profiles():
    return pd.DataFrame({"ground": GROUNDS, "tests": [20, 30, 25, 0, 28],
                         "draw_rate": [0.2, 0.3, 0.15, None, 0.25],
                         "avg_first_innings_runs": [350, 380, 320, None, 360]})


def records():
    return pd.DataFrame({"ground": GROUNDS, "tests": [6, 7, 6, 0, 6], "wins": [3, 2, 3, 0, 4],
                         "draws": [1, 2, 1, 0, 0], "losses": [2, 3, 2, 0, 2]})


def test_profile_chart_is_saved(tmp_path):
    path = plot_ground_profiles(profiles(), tmp_path / "profiles.png", "Test title")
    assert path.exists() and path.stat().st_size > 10_000


def test_results_chart_counts_is_saved(tmp_path):
    path = plot_results(records(), tmp_path / "counts.png", "Test title")
    assert path.exists() and path.stat().st_size > 10_000


def test_results_chart_shares_handles_ground_with_no_tests(tmp_path):
    path = plot_results(records(), tmp_path / "shares.png", "Test title", as_share=True)
    assert path.exists()
