"""Fit the team-strength model and forecast the 2027 Ashes, Test by Test and for the series."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.charts import plot_forecast, plot_strength
from cricket.diagnostics import diagnose
from cricket.model_data import build_model_table, runs_per_wicket, to_model_data
from cricket.predict import forecast_tests, series_scorelines, series_summary, summarise_tests

FORECAST_YEAR = 2027
HOME, AWAY = "England", "Australia"
SCALARS = ["sigma_start", "sigma_drift", "home_advantage", "draw_base", "draw_trend", "sigma_venue",
           "margin_base", "margin_slope", "margin_sigma"]


def sample_posterior(data, draws, tune, chains, seed):
    """Build and fit the PyMC model. PyMC is imported here, so the rest of this file works without it."""
    from cricket.bayes_model import build_model, fit
    return fit(build_model(data), draws=draws, tune=tune, chains=chains, seed=seed)


def flatten(samples, data):
    """Join the chains into one long run of draws and attach the team, year and venue labels."""
    post = {name: values.reshape(-1, *values.shape[2:]) for name, values in samples.items()}
    post.update(teams=data.teams, years=data.years, venues=data.venues)
    return post


def describe(values):
    """Mean and 90% interval of a set of draws."""
    return {"mean": values.mean(), "low_5%": np.percentile(values, 5), "high_95%": np.percentile(values, 95)}


def series_row(name, probs):
    summary = series_summary(series_scorelines(probs))
    return {"forecast": name, "England win series": summary["home_win_series"],
            "Drawn series": summary["drawn_series"], "Australia win series": summary["away_win_series"],
            "Australia retain the urn": summary["drawn_series"] + summary["away_win_series"]}


def main():
    parser = argparse.ArgumentParser(description="Fit the team-strength model and forecast the 2027 Ashes.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--results-only", action="store_true", help="leave out the winning margins")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    processed, manual = Path(args.processed), Path(args.manual)
    figures, outputs = Path(args.figures), Path(args.outputs)
    figures.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    matches = pd.read_csv(processed / "matches.csv", dtype={"match_id": str})
    aliases = pd.read_csv(manual / "venue_aliases.csv")
    fixtures = pd.read_csv(manual / "fixtures_2027.csv")
    mine = pd.read_csv(manual / "my_forecast.csv")
    innings = pd.read_csv(processed / "innings.csv", dtype={"match_id": str})

    rpw = None if args.results_only else runs_per_wicket(innings)
    table = build_model_table(matches, aliases, rpw=rpw)
    data = to_model_data(table, extra_venues=fixtures["venue"])
    print(f"Tests in the model: {len(table):,}   teams: {len(data.teams)}   "
          f"years: {data.years[0]}-{data.years[-1]}   venues: {len(data.venues)}")
    if rpw is None:
        print("Winning margins: left out")
    else:
        print(f"Winning margins: {len(data.margin_idx):,} wins, at {rpw:.1f} runs per wicket "
              f"(a 10-wicket win counts like a {10 * rpw:.0f}-run win)")

    samples, divergences = sample_posterior(data, args.draws, args.tune, args.chains, args.seed)

    checks = diagnose(samples)
    checks.round(3).to_csv(figures / "sampler_checks.csv", index=False)
    print("\nSampler checks (want max_rhat below 1.01 and min_ess above 400):")
    print(checks.round(3).to_string(index=False))
    print(f"Divergent transitions: {divergences}  (want 0)")

    post = flatten(samples, data)
    np.savez_compressed(outputs / "posterior_draws.npz", **{k: post[k] for k in samples})

    params = pd.DataFrame({name: describe(post[name]) for name in SCALARS if name in post}).T
    print("\nModel parameters (mean and 90% interval):")
    print(params.round(3).to_string())

    last = data.years[-1]
    strengths = pd.DataFrame({team: describe(post["strength"][:, -1, data.teams.index(team)])
                              for team in [HOME, AWAY]}).T
    print(f"\nStrength in {last} (0 = average Test side):")
    print(strengths.round(3).to_string())

    grounds = pd.DataFrame({venue: describe(post["venue_effect"][:, data.venues.index(venue)])
                            for venue in fixtures["venue"]}).T
    print("\nGround effect on the draw band (above 0 = more draws than an average venue):")
    print(grounds.round(3).to_string())

    test_probs = forecast_tests(post, fixtures, HOME, AWAY, FORECAST_YEAR, seed=args.seed)
    tests = summarise_tests(test_probs, fixtures)
    tests.round(3).to_csv(figures / "forecast_2027.csv", index=False)
    mine_probs = mine.sort_values("test_no")[["p_england", "p_draw", "p_australia"]].to_numpy()[None, :, :]
    mine_named = mine.rename(columns={"p_england": "my_england", "p_draw": "my_draw", "p_australia": "my_australia"})
    comparison = tests.merge(mine_named, on="test_no")
    print("\nForecast for each Test (model, then my own forecast):")
    print(comparison.round(2).to_string(index=False))

    series = pd.DataFrame([
        series_row("Model", test_probs),
        series_row("Model, if Tests were independent", test_probs.mean(axis=0, keepdims=True)),
        series_row("My forecast", mine_probs),
    ])
    series.round(3).to_csv(figures / "series_2027.csv", index=False)
    print("\nSeries chances:")
    print(series.round(3).to_string(index=False))

    rows = []
    for _, t in comparison.iterrows():
        name = f"Test {t['test_no']}, {t['venue']}"
        rows.append({"label": f"{name}: model", "p_home": t["p_home"], "p_draw": t["p_draw"], "p_away": t["p_away"]})
        rows.append({"label": f"{name}: my forecast", "p_home": t["my_england"], "p_draw": t["my_draw"],
                     "p_away": t["my_australia"]})
    saved = [
        plot_strength(post["strength"], data.years, data.teams, figures / "team_strength.png",
                      "Test team strength over time (median and 90% interval)"),
        plot_forecast(pd.DataFrame(rows), figures / "forecast_2027.png",
                      "2027 Ashes: model forecast and my own forecast, Test by Test"),
    ]
    print("\nSaved:")
    for path in [figures / "sampler_checks.csv", figures / "forecast_2027.csv", figures / "series_2027.csv",
                 *saved, outputs / "posterior_draws.npz"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()