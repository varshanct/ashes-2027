"""Player ratings from ball-by-ball data: batting and bowling averages, weighted towards recent Tests and
pulled towards the average for a player of the same type when a player has little data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_YEARS = 2.0     # a Test two years ago counts half as much as one today
PRIOR_OUTS = 10           # a batter with no record counts as 10 dismissals at the average for his position
TOP_ORDER = 7             # batting positions 1 to 7 are the top order; 8 to 11 the lower order
PRIOR_WICKETS = 10        # a bowler with no record counts as 10 wickets at the average
NOT_BOWLERS_WICKETS = {"run out", "retired out", "obstructing the field", "timed out", "handled the ball"}


def recency_weights(dates, as_of, half_life=HALF_LIFE_YEARS):
    """Weight for each date: 1 today, 0.5 one half-life ago, 0.25 two half-lives ago."""
    age_years = (pd.Timestamp(as_of) - pd.to_datetime(dates)).dt.days / 365.25
    return 0.5 ** (age_years / half_life)


def batting_positions(deliveries):
    """Each player's usual batting position: the median order in which he first faced in each innings."""
    first = deliveries.dropna(subset=["batter_id"]).drop_duplicates(["match_id", "innings_no", "batter_id"])
    order = first.groupby(["match_id", "innings_no"]).cumcount() + 1
    return order.groupby(first["batter_id"]).median()


def player_ratings(deliveries, appearances, matches, as_of, half_life=HALF_LIFE_YEARS):
    """Every player's batting and bowling record from Tests that started before as_of.

    bat_rating: runs per dismissal, recency-weighted and pulled towards the average for his usual batting
    position, top order or lower order (higher is better).
    bowl_rating: runs conceded per wicket, recency-weighted and pulled towards the average (lower is better).
    """
    start = matches.set_index("match_id")["date_start"]
    d = deliveries[pd.to_datetime(deliveries["match_id"].map(start)) < pd.Timestamp(as_of)].copy()
    d["w"] = recency_weights(d["match_id"].map(start), as_of, half_life).to_numpy()

    # Who was out: Cricsheet gives a name, so look up the ID from that match's team sheet.
    ids = appearances[["match_id", "player_name", "player_id"]]
    outs = d[d["is_wicket"]].merge(ids, left_on=["match_id", "player_out"],
                                   right_on=["match_id", "player_name"])
    bowled = d["is_wicket"] & ~d["wicket_kind"].isin(NOT_BOWLERS_WICKETS)
    # Runs charged to the bowler: off the bat, plus wides and no-balls (byes and leg byes are not his fault).
    conceded = d["runs_bat"] + np.where(d["is_wide"] | d["is_noball"], d["runs_extras"], 0)

    table = pd.DataFrame({
        "runs": d.groupby("batter_id")["runs_bat"].sum(),
        "outs": outs.groupby("player_id").size(),
        "w_runs": (d["runs_bat"] * d["w"]).groupby(d["batter_id"]).sum(),
        "w_outs": outs.groupby("player_id")["w"].sum(),
        "balls": d[d["is_legal"]].groupby("bowler_id").size(),
        "conceded": conceded.groupby(d["bowler_id"]).sum(),
        "wickets": bowled.groupby(d["bowler_id"]).sum(),
        "w_conceded": (conceded * d["w"]).groupby(d["bowler_id"]).sum(),
        "w_wickets": (bowled * d["w"]).groupby(d["bowler_id"]).sum(),
    }).fillna(0)

    table["position"] = batting_positions(d)
    top = (table["position"] <= TOP_ORDER).to_numpy()
    overall = table["w_runs"].sum() / table["w_outs"].sum()
    bat_mean_top = group_mean(table[top], overall)
    bat_mean_lower = group_mean(table[~top], overall)
    bowl_mean = table["w_conceded"].sum() / table["w_wickets"].sum()
    table["bat_avg"] = table["runs"] / table["outs"].replace(0, np.nan)
    table["bowl_avg"] = table["conceded"] / table["wickets"].replace(0, np.nan)
    bat_prior = np.where(top, bat_mean_top, bat_mean_lower)
    table["bat_rating"] = (table["w_runs"] + PRIOR_OUTS * bat_prior) / (table["w_outs"] + PRIOR_OUTS)
    table["bowl_rating"] = ((table["w_conceded"] + PRIOR_WICKETS * bowl_mean)
                            / (table["w_wickets"] + PRIOR_WICKETS))
    table.index.name = "player_id"
    table.attrs.update(bat_mean_top=bat_mean_top, bat_mean_lower=bat_mean_lower, bowl_mean=bowl_mean)
    return table


def group_mean(group, fallback):
    """Weighted runs per dismissal for a group of batters, or the fallback if the group has no dismissals."""
    outs = group["w_outs"].sum()
    return group["w_runs"].sum() / outs if outs > 0 else fallback


def count_tests(appearances, matches, as_of):
    """Number of Tests each player appeared in before as_of."""
    start = matches.set_index("match_id")["date_start"]
    before = appearances[pd.to_datetime(appearances["match_id"].map(start)) < pd.Timestamp(as_of)]
    return before.groupby("player_id").size()


def squad_ratings(squads, ratings, tests):
    """The ratings of every squad player, in squad order.

    A player with no Tests gets the average for his role: top order for batters, keepers and allrounders,
    lower order for bowlers.
    """
    out = squads[["team", "player_name", "role", "squad_tier"]].copy()
    ids = squads["cricsheet_id"]
    out["tests"] = ids.map(tests).fillna(0).astype(int)
    for column in ["position", "bat_avg", "bat_rating", "balls", "wickets", "bowl_avg", "bowl_rating"]:
        out[column] = ids.map(ratings[column])
    top, lower = ratings.attrs["bat_mean_top"], ratings.attrs["bat_mean_lower"]
    role_mean = np.where(squads["role"] == "bowler", lower, top)
    out["bat_rating"] = out["bat_rating"].fillna(pd.Series(role_mean, index=out.index))
    out["bowl_rating"] = out["bowl_rating"].fillna(ratings.attrs["bowl_mean"])
    out[["balls", "wickets"]] = out[["balls", "wickets"]].fillna(0).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Rate every squad player from ball-by-ball Test data.")
    parser.add_argument("--as-of", default="2026-11-10")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    processed = Path(args.processed)
    pd.set_option("display.width", 200)

    matches = pd.read_csv(processed / "matches.csv", dtype={"match_id": str})
    deliveries = pd.read_csv(processed / "deliveries.csv", dtype={"match_id": str}, low_memory=False)
    appearances = pd.read_csv(processed / "appearances.csv", dtype={"match_id": str})
    squads = pd.read_csv(Path(args.manual) / "squads_2027.csv", dtype={"cricsheet_id": str})

    ratings = player_ratings(deliveries, appearances, matches, args.as_of)
    table = squad_ratings(squads, ratings, count_tests(appearances, matches, args.as_of))
    print(f"Ratings from every Test before {args.as_of}, each Test weighted by how recent it is "
          f"(half-life {HALF_LIFE_YEARS:g} years).")
    print("bat_avg and bowl_avg are plain career averages; the ratings are what the model will use.")
    print(f"Average for pulling batters in: top order {ratings.attrs['bat_mean_top']:.1f}, "
          f"lower order {ratings.attrs['bat_mean_lower']:.1f} runs per dismissal.")
    for team in ["England", "Australia"]:
        print(f"\n{team}:")
        print(table[table["team"] == team].drop(columns="team").round(1).to_string(index=False))

    Path(args.figures).mkdir(exist_ok=True)
    path = Path(args.figures) / "player_ratings_2027_squads.csv"
    table.round(2).to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()