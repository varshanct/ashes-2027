"""Back-test on every year of Tests: fit on all earlier Tests, then forecast and score that year's Tests."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.backtest import VERSIONS, base_rates, score
from cricket.charts import BAR, BASE, GRID, INK, INK_2, MUTED, SURFACE, _style, plt
from cricket.diagnostics import diagnose
from cricket.forecast import flatten, sample_posterior
from cricket.model_data import TREND_BASE_YEAR, build_model_table, runs_per_wicket, to_model_data
from cricket.predict import result_probs, strength_in_year, venue_draws

FORECASTS = {**VERSIONS, "base_rates": "Past results", "thirds": "Equal thirds"}


def forecast_matches(post, tests, year, seed=2027):
    """Chances of [team1 loses, draw, team1 wins] for any Tests in one year, averaged over posterior draws."""
    rng = np.random.default_rng(seed)
    strength = strength_in_year(post, year, rng)
    number = {team: i for i, team in enumerate(post["teams"])}
    a, b = tests["team1"].map(number).to_numpy(), tests["team2"].map(number).to_numpy()
    eta = strength[:, a] - strength[:, b] + post["home_advantage"][:, None] * tests["home"].to_numpy()
    venues = {v: venue_draws(post, v, rng) for v in tests["venue_key"].unique()}
    venue = np.stack([venues[v] for v in tests["venue_key"]], axis=1)
    decade = (year - TREND_BASE_YEAR) / 10
    log_width = post["draw_base"][:, None] + venue + post["draw_trend"][:, None] * decade
    return result_probs(eta, np.exp(log_width)).mean(axis=0)


def past_results_forecast(train, home):
    """The simple forecast from past results, turned round to team1's point of view for each Test."""
    p_home, p_draw, p_away = base_rates(train)
    neutral = [(1 - p_draw) / 2, p_draw, (1 - p_draw) / 2]
    rows = {1: [p_away, p_draw, p_home], -1: [p_home, p_draw, p_away], 0: neutral}
    return np.array([rows[h] for h in home])


def backtest_year(year, matches, innings, aliases, full_table, args):
    """Fit both versions on Tests before the year, then forecast and score every Test in it."""
    start = f"{year}-01-01"
    before = matches[pd.to_datetime(matches["date_start"]) < start]
    rpw = runs_per_wicket(innings[innings["match_id"].isin(before["match_id"])])
    train = build_model_table(before, aliases, rpw=rpw)
    known = set(train["team1"]) | set(train["team2"])
    in_year = full_table[full_table["year"] == year]
    tests = in_year[in_year["team1"].isin(known) & in_year["team2"].isin(known)].reset_index(drop=True)
    skipped = len(in_year) - len(tests)
    print(f"\n{year}: trained on {len(train):,} Tests, forecasting {len(tests)}"
          + (f" (skipped {skipped} involving a team with no earlier Tests)" if skipped else ""))

    rows = tests[["match_id", "date_start", "team1", "team2", "venue_key", "result"]].copy()
    rows.insert(0, "year", year)
    checks = []
    for key, label in VERSIONS.items():
        table = train if key == "margins" else train.drop(columns="margin")
        data = to_model_data(table, extra_venues=tests["venue_key"])
        samples, divergences = sample_posterior(data, args.draws, args.tune, args.chains, args.seed)
        diag = diagnose(samples)
        checks.append({"year": year, "version": label, "max_rhat": diag["max_rhat"].max(),
                       "min_ess": diag["min_ess"].min(), "divergences": divergences})
        print(f"  {label}: worst R-hat {diag['max_rhat'].max():.3f}   "
              f"smallest ESS {diag['min_ess'].min():.0f}   divergences {divergences}")
        probs = forecast_matches(flatten(samples, data), tests, year, seed=args.seed)
        rows[f"log_loss_{key}"], rows[f"brier_{key}"] = score(probs, tests["result"])

    simple = {"base_rates": past_results_forecast(train, tests["home"]),
              "thirds": np.full((len(tests), 3), 1 / 3)}
    for key, probs in simple.items():
        rows[f"log_loss_{key}"], rows[f"brier_{key}"] = score(probs, tests["result"])
    return rows, checks


def scores_by_year(tests):
    """Average log loss and Brier score per Test, for each year and for all years together."""
    columns = {f"{measure}_{key}": f"{name} {label}" for key, name in FORECASTS.items()
               for measure, label in [("log_loss", "log loss"), ("brier", "Brier")]}
    table = tests.groupby("year")[list(columns)].mean()
    table.insert(0, "tests", tests.groupby("year").size())
    total = tests[list(columns)].mean()
    total["tests"] = len(tests)
    table.loc["All years"] = total
    return table.rename(columns=columns)


def plot_log_loss_by_year(scores, path, title):
    """One line per forecast: average log loss in each year. Lower is better."""
    years = scores.drop(index="All years")
    styles = {"With margins": (BAR, "-", 2.2), "Results only": (INK_2, "--", 1.4),
              "Past results": (MUTED, ":", 1.4), "Equal thirds": (BASE, "-", 1.2)}
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    for name, (colour, line, width) in styles.items():
        ax.plot(years.index.astype(int), years[f"{name} log loss"], color=colour, linestyle=line,
                linewidth=width, marker="o" if name == "With margins" else None, markersize=4, label=name)
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylabel("Average log loss per Test (lower is better)", color=MUTED)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Back-test the model on every year of Tests.")
    parser.add_argument("--first", type=int, default=2015)
    parser.add_argument("--last", type=int, default=2026)
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
    innings = pd.read_csv(Path(args.processed) / "innings.csv", dtype={"match_id": str})
    aliases = pd.read_csv(Path(args.manual) / "venue_aliases.csv")
    full_table = build_model_table(matches, aliases)

    all_rows, all_checks = [], []
    for year in range(args.first, args.last + 1):
        rows, checks = backtest_year(year, matches, innings, aliases, full_table, args)
        all_rows.append(rows)
        all_checks.extend(checks)
    tests = pd.concat(all_rows, ignore_index=True)
    checks = pd.DataFrame(all_checks)
    scores = scores_by_year(tests)

    print("\nSampler checks across every fit:")
    print(f"  worst R-hat {checks['max_rhat'].max():.3f}   smallest ESS {checks['min_ess'].min():.0f}   "
          f"total divergences {checks['divergences'].sum()}")
    print("\nAverage log loss per Test (lower is better):")
    print(scores[["tests"] + [f"{name} log loss" for name in FORECASTS.values()]].round(3).to_string())
    print("\nAverage Brier score per Test (lower is better):")
    print(scores[["tests"] + [f"{name} Brier" for name in FORECASTS.values()]].round(3).to_string())

    tests.round(3).to_csv(figures / "rolling_backtest_tests.csv", index=False)
    scores.round(3).to_csv(figures / "rolling_backtest_scores.csv")
    checks.round(3).to_csv(figures / "rolling_backtest_checks.csv", index=False)
    plot_log_loss_by_year(scores, figures / "rolling_backtest.png",
                          f"Back-test on every year of Tests, {args.first} to {args.last}")
    print("\nSaved:")
    for file in ["rolling_backtest_tests.csv", "rolling_backtest_scores.csv", "rolling_backtest_checks.csv",
                 "rolling_backtest.png"]:
        print(f"  {figures / file}")


if __name__ == "__main__":
    main()
