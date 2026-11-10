"""Forecast past Ashes series using only the Tests played before each one, and score the forecasts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.charts import plot_forecast
from cricket.diagnostics import diagnose
from cricket.forecast import flatten, sample_posterior, series_row
from cricket.model_data import build_model_table, runs_per_wicket, to_model_data
from cricket.predict import forecast_tests, summarise_tests

HOME, AWAY = "England", "Australia"
SERIES = {"2019 Ashes": ("2019-08-01", "2019-09-16"),
          "2023 Ashes": ("2023-06-16", "2023-07-31")}
RESULT_NAMES = ["England won", "Draw", "Australia won"]
VERSIONS = {"results": "Results only", "margins": "With margins"}


def series_tests(table, home, away, start, end):
    """The Tests between two teams in a date window. Outcome is from the home side: 0 win, 1 draw, 2 loss."""
    dates = pd.to_datetime(table["date_start"])
    pair = (((table["team1"] == home) & (table["team2"] == away))
            | ((table["team1"] == away) & (table["team2"] == home)))
    rows = table[pair & (dates >= start) & (dates <= end)].copy()
    home_is_team1 = rows["team1"] == home
    team1_won = rows["result"] == 2
    rows["outcome"] = np.where(rows["result"] == 1, 1, np.where(team1_won == home_is_team1, 0, 2))
    rows["test_no"] = np.arange(1, len(rows) + 1)
    rows = rows.rename(columns={"venue_key": "venue"})
    return rows[["test_no", "match_id", "date_start", "venue", "outcome"]].reset_index(drop=True)


def base_rates(table):
    """Share of home wins, draws and away wins in past Tests that had a home side."""
    homed = table[table["home"] != 0]
    if homed.empty:
        return np.full(3, 1 / 3)
    home_won = ((homed["home"] == 1) & (homed["result"] == 2)) | ((homed["home"] == -1) & (homed["result"] == 0))
    p_home, p_draw = home_won.mean(), (homed["result"] == 1).mean()
    return np.array([p_home, p_draw, 1 - p_home - p_draw])


def score(probs, outcomes):
    """Log loss and Brier score for each Test. Lower is better for both."""
    probs, outcomes = np.asarray(probs, dtype=float), np.asarray(outcomes)
    log_loss = -np.log(probs[np.arange(len(outcomes)), outcomes])
    brier = ((probs - np.eye(3)[outcomes]) ** 2).sum(axis=1)
    return log_loss, brier


def score_table(tests):
    """Average score per Test for each forecast, for each series and for both together."""
    forecasts = {**VERSIONS, "base_rates": "Past results", "thirds": "Equal thirds"}
    groups = [*tests.groupby("series"), ("Both series", tests)]
    rows = []
    for series, group in groups:
        for measure, prefix in [("log loss", "log_loss"), ("Brier", "brier")]:
            row = {"series": series, "measure": measure}
            row.update({name: group[f"{prefix}_{key}"].mean() for key, name in forecasts.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def backtest_series(name, start, end, matches, innings, aliases, full_table, args):
    """Fit both versions of the model on Tests before the series, then forecast and score every Test."""
    before = matches[pd.to_datetime(matches["date_start"]) < start]
    rpw = runs_per_wicket(innings[innings["match_id"].isin(before["match_id"])])
    train = build_model_table(before, aliases, rpw=rpw)
    tests = series_tests(full_table, HOME, AWAY, start, end)
    print(f"\n{name}: trained on {len(train):,} Tests before {start}, forecasting {len(tests)} Tests")

    rows = tests[["test_no", "venue", "date_start"]].copy()
    rows.insert(0, "series", name)
    rows["result"] = [RESULT_NAMES[k] for k in tests["outcome"]]
    england_wins, australia_wins = (tests["outcome"] == 0).sum(), (tests["outcome"] == 2).sum()
    series_rows = []
    for key, label in VERSIONS.items():
        table = train if key == "margins" else train.drop(columns="margin")
        data = to_model_data(table, extra_venues=tests["venue"])
        samples, divergences = sample_posterior(data, args.draws, args.tune, args.chains, args.seed)
        checks = diagnose(samples)
        print(f"  {label}: worst R-hat {checks['max_rhat'].max():.3f}   "
              f"smallest ESS {checks['min_ess'].min():.0f}   divergences {divergences}")

        post = flatten(samples, data)
        probs = forecast_tests(post, tests, HOME, AWAY, int(start[:4]), seed=args.seed)
        mean = probs.mean(axis=0)
        rows[f"p_home_{key}"], rows[f"p_draw_{key}"], rows[f"p_away_{key}"] = mean.T
        rows[f"log_loss_{key}"], rows[f"brier_{key}"] = score(mean, tests["outcome"])
        series = series_row(f"{name}, {label.lower()}", probs)
        series["Actual"] = f"England {england_wins}-{australia_wins} Australia"
        series_rows.append(series)

    rates = np.tile(base_rates(train), (len(tests), 1))
    thirds = np.full((len(tests), 3), 1 / 3)
    for label, p in [("base_rates", rates), ("thirds", thirds)]:
        rows[f"log_loss_{label}"], rows[f"brier_{label}"] = score(p, tests["outcome"])
    return rows, series_rows


def main():
    parser = argparse.ArgumentParser(description="Back-test the model on past Ashes series.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    figures = Path(args.figures)
    figures.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    matches = pd.read_csv(Path(args.processed) / "matches.csv", dtype={"match_id": str})
    aliases = pd.read_csv(Path(args.manual) / "venue_aliases.csv")
    innings = pd.read_csv(Path(args.processed) / "innings.csv", dtype={"match_id": str})
    full_table = build_model_table(matches, aliases)

    all_rows, all_series = [], []
    for name, (start, end) in SERIES.items():
        rows, series = backtest_series(name, start, end, matches, innings, aliases, full_table, args)
        all_rows.append(rows)
        all_series.extend(series)
    tests = pd.concat(all_rows, ignore_index=True)
    series = pd.DataFrame(all_series)

    print("\nEvery Test (chances from the model with margins, then what happened):")
    columns = ["series", "test_no", "venue", "p_home_margins", "p_draw_margins", "p_away_margins", "result"]
    print(tests[columns].round(2).to_string(index=False))

    scores = score_table(tests)
    print("\nAverage score per Test (lower is better):")
    print(scores.round(3).to_string(index=False))

    print("\nSeries chances against the actual result:")
    print(series.round(3).to_string(index=False))

    tests.round(3).to_csv(figures / "backtest_tests.csv", index=False)
    scores.round(3).to_csv(figures / "backtest_scores.csv", index=False)
    series.round(3).to_csv(figures / "backtest_series.csv", index=False)
    chart = tests.assign(label=tests["series"].str[:4] + " Test " + tests["test_no"].astype(str) + ", "
                         + tests["venue"] + " (" + tests["result"] + ")",
                         p_home=tests["p_home_margins"], p_draw=tests["p_draw_margins"],
                         p_away=tests["p_away_margins"])
    plot_forecast(chart, figures / "backtest.png",
                  "Back-test: chances before each Ashes Test (model with margins), and what happened")

    print("\nSaved:")
    for file in ["backtest_tests.csv", "backtest_scores.csv", "backtest_series.csv", "backtest.png"]:
        print(f"  {figures / file}")


if __name__ == "__main__":
    main()