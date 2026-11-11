"""Find each squad player's Cricsheet ID from Test appearances and the People register."""

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["player_id", "player_name", "tests", "last_test"]


def surname(full_name):
    """The last word of a name: 'Steve Smith' -> 'Smith'."""
    return full_name.strip().split()[-1]


def capped_candidates(name_part, team, appearances, matches):
    """Test players for this team whose name contains name_part, most recent first."""
    found = appearances[(appearances["team"] == team)
                        & appearances["player_name"].str.contains(name_part, case=False, regex=False)]
    if found.empty:
        return pd.DataFrame(columns=COLUMNS)
    found = found.merge(matches[["match_id", "date_start"]], on="match_id")
    summary = (found.groupby(["player_id", "player_name"])
                    .agg(tests=("match_id", "nunique"), last_test=("date_start", "max"))
                    .reset_index()
                    .sort_values("last_test", ascending=False)
                    .reset_index(drop=True))
    return summary[COLUMNS]


def register_candidates(name_part, people):
    """Anyone in the People register whose name contains name_part (for players with no Tests yet)."""
    found = people[people["name"].str.contains(name_part, case=False, regex=False)]
    return found[["identifier", "name"]].reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Suggest Cricsheet IDs for every player in the squads file.")
    parser.add_argument("--squads", default="data/manual/squads_2027.csv")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--people", default="data/raw/people.csv")
    args = parser.parse_args()

    squads = pd.read_csv(args.squads, dtype={"cricsheet_id": str})
    appearances = pd.read_csv(Path(args.processed) / "appearances.csv", dtype={"match_id": str})
    matches = pd.read_csv(Path(args.processed) / "matches.csv", dtype={"match_id": str})
    people = pd.read_csv(args.people, dtype=str)

    for _, row in squads.iterrows():
        name_part = surname(row["player_name"])
        print(f"\n=== {row['team']}: {row['player_name']}  (searching for '{name_part}')")
        capped = capped_candidates(name_part, row["team"], appearances, matches)
        if not capped.empty:
            print(capped.head(5).to_string(index=False))
        else:
            print(f"No Test appearances for {row['team']}. Candidates from the People register:")
            print(register_candidates(name_part, people).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
