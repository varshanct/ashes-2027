"""Forecast the 2027 Ashes from the two squads: who is fit for each Test, who comes in for anyone missing,
and how strong each XI is. Uses the model version with XI ratings, which forecast as well as results only."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.charts import plot_forecast
from cricket.diagnostics import diagnose
from cricket.forecast import AWAY, FORECAST_YEAR, HOME, describe, flatten, sample_posterior, series_row
from cricket.lineups import MAIN_BOWLERS
from cricket.manual_data import AVAIL_COLS, TEAMS
from cricket.model_data import XI_SCALE, add_xi, build_model_table, to_model_data
from cricket.players import check_first_class, count_tests, player_ratings, squad_ratings, with_first_class
from cricket.predict import forecast_tests, summarise_tests

SAME_FITNESS = 0.5    # chance a player's fitness for a Test follows his fitness for the whole series
BATTING_SLOTS = 6     # an allrounder batting in the top six is replaced by a batter; lower down, by a bowler


def need_for(player):
    """The kind of player who replaces him if he is missing: keeper, batter, pace or spin."""
    if player["role"] == "keeper":
        return "keeper"
    if player["role"] == "batter" or player["bowling_type"] not in ("pace", "spin"):
        return "batter"
    if player["role"] == "allrounder" and player["expected_batting_position"] <= BATTING_SLOTS:
        return "batter"
    return player["bowling_type"]


def availability_draws(squads, availability, n, rng, without=()):
    """Whether each squad player is available for each Test, in n simulated series: shape (n, tests, players).

    Each player's chance for each Test comes from availability.csv. Players not listed are always available,
    and players named in `without` never are. Half the time a player's fitness for a Test follows one draw
    for the whole series, so a player who misses one Test is more likely to miss others.
    """
    ids = squads["cricsheet_id"].to_numpy()
    listed = availability.set_index("cricsheet_id")[AVAIL_COLS].reindex(ids).fillna(1.0)
    chances = np.array(listed, dtype=float).T                             # shape (tests, players)
    chances[:, squads["player_name"].isin(list(without)).to_numpy()] = 0.0
    tests, players = chances.shape
    series = rng.random((n, 1, players))
    fresh = rng.random((n, tests, players))
    follow = rng.random((n, tests, players)) < SAME_FITNESS
    return np.where(follow, series, fresh) < chances[None]


def pick_xi(team_squad, cover, missing):
    """The XI for one Test: the expected XI, with each missing player replaced.

    The first available player in the cover plan for his role comes in. A keeper's cover is usually already
    in the XI (he takes the gloves), so a batter comes in instead. If the cover plan runs out, any available
    squad player of the same bowling type comes in, then anyone available, and if nobody is left, a
    replacement rated at the average for the role.
    """
    core = team_squad[team_squad["expected_batting_position"].notna()]
    core = core.sort_values("expected_batting_position")
    xi = [pid for pid in core["cricsheet_id"] if pid not in missing]
    for _, player in core[core["cricsheet_id"].isin(missing)].iterrows():
        xi.append(replacement(player, team_squad, cover, missing, xi))
    return xi


def replacement(player, team_squad, cover, missing, xi):
    """The player who comes in for one missing player (see pick_xi)."""
    def free(pid):
        return pid not in missing and pid not in xi

    need = need_for(player)
    for kind in ([need, "batter"] if need == "keeper" else [need]):
        plan = cover[cover["need"] == kind].sort_values("cover_order")
        for pid in plan["cricsheet_id"]:
            if free(pid):
                return pid
    rest = team_squad[team_squad["cricsheet_id"].map(free)]
    for pool in (rest[rest["bowling_type"] == player["bowling_type"]], rest):
        if len(pool):
            return pool["cricsheet_id"].iloc[0]
    return f"replacement {need}"


def role_averages(attrs):
    """Ratings for a replacement nobody has picked: the average for his role."""
    def player(bat):
        return pd.Series({"bat_rating": bat, "bowl_rating": attrs["bowl_mean"], "balls": 0})
    top, lower = player(attrs["bat_mean_top"]), player(attrs["bat_mean_lower"])
    return {"replacement batter": top, "replacement keeper": top,
            "replacement pace": lower, "replacement spin": lower}


def rate_xi(xi, rated, averages):
    """Batting and bowling rating of one XI, the same way as for past Tests: the batting rating of all
    eleven, and the bowling rating of the five who have bowled most balls in Tests."""
    rows = pd.DataFrame([rated.loc[pid] if pid in rated.index else averages[pid] for pid in xi])
    bowlers = rows.sort_values("balls", ascending=False, kind="stable").head(MAIN_BOWLERS)
    return rows["bat_rating"].mean(), bowlers["bowl_rating"].mean()


def simulate_xis(squads, cover, rated, averages, available):
    """Each team's XI ratings in every simulated series and Test, and how many Tests each player plays.

    Returns {team: {"bat": array (n, tests), "bowl": array (n, tests)}} and the average number of Tests
    played by each player, out of the five.
    """
    n, n_tests, _ = available.shape
    ids = squads["cricsheet_id"].to_numpy()
    xis, played = {}, Counter()
    for team in TEAMS:
        in_team = (squads["team"] == team).to_numpy()
        team_squad, team_cover, team_ids = squads[in_team], cover[cover["team"] == team], ids[in_team]
        bat, bowl = np.zeros((n, n_tests)), np.zeros((n, n_tests))
        known = {}                                    # each set of missing players only needs picking once
        for s in range(n):
            for t in range(n_tests):
                missing = frozenset(team_ids[~available[s, t, in_team]])
                if missing not in known:
                    xi = pick_xi(team_squad, team_cover, missing)
                    known[missing] = (xi, *rate_xi(xi, rated, averages))
                xi, bat[s, t], bowl[s, t] = known[missing]
                played.update(xi)
        xis[team] = {"bat": bat, "bowl": bowl}
    return xis, pd.Series(played, dtype=float) / n


def xi_edge(post, xis):
    """How much England's XI is worth over Australia's in each Test, for every posterior draw."""
    bat_gap = (xis[HOME]["bat"] - xis[AWAY]["bat"]) / XI_SCALE
    bowl_gap = (xis[AWAY]["bowl"] - xis[HOME]["bowl"]) / XI_SCALE      # a lower bowling rating is better
    return post["xi_bat_effect"][:, None] * bat_gap + post["xi_bowl_effect"][:, None] * bowl_gap


def rated_players(table, squads, column_suffix=""):
    """Squad ratings indexed by player ID: batting, bowling and balls bowled."""
    rated = pd.DataFrame({"bat_rating": table[f"bat_rating{column_suffix}"].to_numpy(),
                          "bowl_rating": table[f"bowl_rating{column_suffix}"].to_numpy(),
                          "balls": table["balls"].to_numpy()}, index=squads["cricsheet_id"].to_numpy())
    return rated


def main():
    parser = argparse.ArgumentParser(description="Forecast the 2027 Ashes from the two squads.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--as-of", default="2026-11-10")
    parser.add_argument("--without", default="", help="players to leave out of a what-if forecast, "
                                                       "comma-separated, e.g. 'Pat Cummins,Josh Hazlewood'")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    processed, manual, figures = Path(args.processed), Path(args.manual), Path(args.figures)
    figures.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    xi_path = processed / "xi_ratings.csv"
    if not xi_path.exists():
        raise SystemExit(f"{xi_path} is missing. Run `uv run python -m cricket.lineups` first.")
    matches = pd.read_csv(processed / "matches.csv", dtype={"match_id": str})
    deliveries = pd.read_csv(processed / "deliveries.csv", dtype={"match_id": str}, low_memory=False)
    appearances = pd.read_csv(processed / "appearances.csv", dtype={"match_id": str})
    xi = pd.read_csv(xi_path, dtype={"match_id": str})
    aliases = pd.read_csv(manual / "venue_aliases.csv")
    fixtures = pd.read_csv(manual / "fixtures_2027.csv").sort_values("test_no").reset_index(drop=True)
    squads = pd.read_csv(manual / "squads_2027.csv", dtype={"cricsheet_id": str})
    availability = pd.read_csv(manual / "availability.csv", dtype={"cricsheet_id": str})
    cover = pd.read_csv(manual / "cover_plan.csv", dtype={"cricsheet_id": str})
    first_class = pd.read_csv(manual / "first_class.csv", dtype={"cricsheet_id": str})
    without = [name.strip() for name in args.without.split(",") if name.strip()]
    unknown = sorted(set(without) - set(squads["player_name"]))
    if unknown:
        raise SystemExit(f"Not in squads_2027.csv: {unknown}. Use the names exactly as they appear there.")
    problems = check_first_class(first_class, squads)
    if problems:
        raise SystemExit("Fix first_class.csv first:\n  " + "\n  ".join(problems))

    # 1. Fit the model with XI ratings on every Test.
    table = add_xi(build_model_table(matches, aliases), xi)
    data = to_model_data(table, extra_venues=fixtures["venue"])
    print(f"Fitting the model with XI ratings on {len(table):,} Tests.")
    samples, divergences = sample_posterior(data, args.draws, args.tune, args.chains, args.seed)
    checks = diagnose(samples)
    print(f"Sampler checks: worst R-hat {checks['max_rhat'].max():.3f}   "
          f"smallest ESS {checks['min_ess'].min():.0f}   divergences {divergences}")
    post = flatten(samples, data)
    effects = pd.DataFrame({name: describe(post[name]) for name in ["xi_bat_effect", "xi_bowl_effect"]}).T
    print("\nWorth of 10 runs of XI batting or bowling average (mean and 90% interval):")
    print(effects.round(3).to_string())

    # 2. Rate every squad player, from Test records only and with first-class records too.
    ratings = player_ratings(deliveries, appearances, matches, args.as_of)
    players = squad_ratings(squads, ratings, count_tests(appearances, matches, args.as_of))
    players = with_first_class(players, ratings, squads, first_class)
    tests_only = rated_players(players, squads, "_tests_only")
    with_fc = rated_players(players, squads)
    averages = role_averages(ratings.attrs)

    # 3. Simulate who plays each Test, rate the XIs, and forecast. Simulation i goes with posterior draw i.
    n = len(post["home_advantage"])
    everyone = np.ones((n, len(fixtures), len(squads)), dtype=bool)
    available = availability_draws(squads, availability, n, np.random.default_rng(args.seed))
    scenarios = [("Expected XIs, everyone fit", everyone, tests_only),
                 ("Availability simulated", available, tests_only),
                 ("Availability simulated, with first-class records", available, with_fc)]
    if without:
        missing = availability_draws(squads, availability, n, np.random.default_rng(args.seed), without)
        scenarios.append((f"Availability simulated, without {' and '.join(without)}", missing, tests_only))

    series, tests, strengths, played_by = [], [], [], {}
    for name, avail, rated in scenarios:
        xis, played = simulate_xis(squads, cover, rated, averages, avail)
        edge = xi_edge(post, xis)
        probs = forecast_tests(post, fixtures, HOME, AWAY, FORECAST_YEAR, seed=args.seed, edge=edge)
        series.append(series_row(name, probs))
        tests.append(summarise_tests(probs, fixtures).assign(forecast=name))
        strengths.append({"forecast": name, **{f"{team} {kind}": xis[team][kind].mean()
                                               for team in TEAMS for kind in ("bat", "bowl")}})
        played_by[name] = played

    strengths = pd.DataFrame(strengths)
    print("\nAverage XI ratings (bat: runs per dismissal, higher is better; "
          "bowl: runs per wicket, lower is better):")
    print(strengths.round(1).to_string(index=False))

    appearances_table = squads[["team", "player_name"]].copy()
    main_played = played_by["Availability simulated"]
    appearances_table["tests_played"] = squads["cricsheet_id"].map(main_played).fillna(0)
    print(f"\nAverage Tests played out of {len(fixtures)}, with availability simulated:")
    for team in TEAMS:
        in_team = appearances_table["team"] == team
        shown = appearances_table[in_team & (appearances_table["tests_played"] > 0)]
        print(f"\n{team}:")
        print(shown.sort_values("tests_played", ascending=False)[["player_name", "tests_played"]]
              .round(2).to_string(index=False))

    tests = pd.concat(tests, ignore_index=True)
    main_tests = tests[tests["forecast"] == "Availability simulated"]
    print("\nForecast for each Test, with availability simulated:")
    print(main_tests[["test_no", "venue", "p_home", "p_draw", "p_away"]].round(2).to_string(index=False))

    series = pd.DataFrame(series)
    print("\nSeries chances:")
    print(series.round(3).to_string(index=False))

    rows = series.rename(columns={"forecast": "label", "England win series": "p_home",
                                  "Drawn series": "p_draw", "Australia win series": "p_away"})
    saved = [figures / "squad_forecast_series.csv", figures / "squad_forecast_tests.csv",
             figures / "squad_forecast_players.csv", figures / "squad_forecast.png"]
    series.round(3).to_csv(saved[0], index=False)
    tests.round(3).to_csv(saved[1], index=False)
    appearances_table.round(2).to_csv(saved[2], index=False)
    labels = ("England win series", "Drawn series", "Australia win series")
    plot_forecast(rows, saved[3], "2027 Ashes series forecast from the two squads", labels=labels)
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()