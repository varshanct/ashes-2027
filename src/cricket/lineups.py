"""Rate each side's XI in every Test from its players' ratings, using only Tests from earlier years."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.model_data import build_model_table
from cricket.players import TOP_ORDER, player_ratings

MAIN_BOWLERS = 5          # a side's bowling is rated on the five players who have bowled most
FIRST_YEAR = 2004         # earlier years have too few previous Tests to rate players from


def positions_in_match(deliveries):
    """The order in which each player first batted in each Test (his best position across both innings)."""
    first = deliveries.dropna(subset=["batter_id"]).drop_duplicates(["match_id", "innings_no", "batter_id"])
    first = first.assign(position=first.groupby(["match_id", "innings_no"]).cumcount() + 1)
    return first.groupby(["match_id", "batter_id"])["position"].min()


def rate_xi(player_ids, ratings, match_positions):
    """Batting and bowling rating of one XI.

    bat_xi: the average batting rating of the XI (higher is better).
    bowl_xi: the average bowling rating of the five who have bowled most (lower is better).
    A player with no earlier Tests gets the average: for batting, the top or lower order depending on where he
    batted in this match, and for bowling, the average bowler.
    """
    rated = ratings.reindex(player_ids)
    top, lower = ratings.attrs["bat_mean_top"], ratings.attrs["bat_mean_lower"]
    position = match_positions.reindex(player_ids).to_numpy()
    debut_bat = np.where(position <= TOP_ORDER, top, lower)          # did not bat counts as lower order
    bat = rated["bat_rating"].fillna(pd.Series(debut_bat, index=rated.index))
    balls = rated["balls"].fillna(0)
    bowl = rated["bowl_rating"].fillna(ratings.attrs["bowl_mean"])
    main = balls.sort_values(ascending=False, kind="stable").index[:MAIN_BOWLERS]
    return {"bat_xi": bat.mean(), "bowl_xi": bowl.loc[main].mean(),
            "new_players": int(rated["bat_rating"].isna().sum())}


def xi_ratings(deliveries, appearances, matches, first_year=FIRST_YEAR, last_year=None, progress=print):
    """One row per side per Test from first_year on, rated from players' Tests before 1 January that year."""
    years = pd.to_datetime(matches.set_index("match_id")["date_start"]).dt.year
    last_year = last_year or int(years.max())
    match_positions = positions_in_match(deliveries)
    rows = []
    for year in range(first_year, last_year + 1):
        ratings = player_ratings(deliveries, appearances, matches, f"{year}-01-01")
        in_year = appearances[appearances["match_id"].map(years) == year]
        for (match_id, team), xi in in_year.groupby(["match_id", "team"], sort=False):
            positions = match_positions.loc[match_id] if match_id in match_positions.index else pd.Series()
            rows.append({"match_id": match_id, "team": team, "players": len(xi),
                         **rate_xi(xi["player_id"].to_numpy(), ratings, positions)})
        progress(f"  {year}: rated {in_year['match_id'].nunique()} Tests")
    return pd.DataFrame(rows)


def add_xi_gaps(table, xi):
    """Add each Test's XI gaps from team1's point of view: positive means team1's XI is stronger.

    bat_gap: team1's batting rating minus team2's. bowl_gap: team2's bowling rating minus team1's (a lower
    bowling rating is better). Tests without XI ratings are dropped.
    """
    one = xi.rename(columns={"team": "team1", "bat_xi": "bat1", "bowl_xi": "bowl1"})
    two = xi.rename(columns={"team": "team2", "bat_xi": "bat2", "bowl_xi": "bowl2"})
    out = table.merge(one[["match_id", "team1", "bat1", "bowl1"]], on=["match_id", "team1"])
    out = out.merge(two[["match_id", "team2", "bat2", "bowl2"]], on=["match_id", "team2"])
    out["bat_gap"] = out["bat1"] - out["bat2"]
    out["bowl_gap"] = out["bowl2"] - out["bowl1"]
    return out.drop(columns=["bat1", "bat2", "bowl1", "bowl2"])


def results_by_gap(table, column, groups=5):
    """Share of team1 wins, draws and losses, in groups from the most negative gap to the most positive."""
    bands = pd.qcut(table[column], groups, labels=[f"{i + 1}" for i in range(groups)])
    summary = table.groupby(bands, observed=True).agg(
        tests=("result", "size"), average_gap=(column, "mean"),
        team1_won=("result", lambda r: (r == 2).mean()), drawn=("result", lambda r: (r == 1).mean()),
        team1_lost=("result", lambda r: (r == 0).mean()))
    summary.index.name = f"{column} group"
    return summary


def main():
    parser = argparse.ArgumentParser(description="Rate each side's XI in every Test.")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    args = parser.parse_args()
    processed = Path(args.processed)
    pd.set_option("display.width", 200)

    matches = pd.read_csv(processed / "matches.csv", dtype={"match_id": str})
    deliveries = pd.read_csv(processed / "deliveries.csv", dtype={"match_id": str}, low_memory=False)
    appearances = pd.read_csv(processed / "appearances.csv", dtype={"match_id": str})
    aliases = pd.read_csv(Path(args.manual) / "venue_aliases.csv")

    print(f"Rating every XI from {FIRST_YEAR}, using only Tests before 1 January of each year:")
    xi = xi_ratings(deliveries, appearances, matches, first_year=FIRST_YEAR)
    path = processed / "xi_ratings.csv"
    xi.round(3).to_csv(path, index=False)
    print(f"\nRated {len(xi):,} XIs in {xi['match_id'].nunique():,} Tests. Saved: {path}")
    print(f"XIs with 11 players: {(xi['players'] == 11).mean():.0%}. "
          f"Average new players per XI: {xi['new_players'].mean():.2f}")

    table = add_xi_gaps(build_model_table(matches, aliases), xi)
    print(f"\nTests with both XIs rated: {len(table):,}")
    print("\nDoes the side with the better batting XI do better?")
    print("(group 1: team1's XI much weaker ... group 5: team1's XI much stronger)")
    print(results_by_gap(table, "bat_gap").round(2).to_string())
    print("\nDoes the side with the better bowling XI do better?")
    print(results_by_gap(table, "bowl_gap").round(2).to_string())


if __name__ == "__main__":
    main()