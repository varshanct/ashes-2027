"""Sanity checks on the real processed Test tables."""

import argparse
from pathlib import Path

import pandas as pd


def margin_mismatches(matches, innings):
    """Wins where the recorded margin does not match the innings totals.

    Win by runs (including an innings win): winner's total runs minus loser's total runs = margin_runs.
    Win by wickets: 10 minus wickets lost in the winner's final innings = margin_wkts.
    """
    totals = innings.groupby(["match_id", "batting_team"])["runs"].sum()
    rows = []
    for _, m in matches[matches["result_type"] == "win"].iterrows():
        winner = m["winner"]
        loser = m["team2"] if winner == m["team1"] else m["team1"]
        if pd.notna(m["margin_runs"]):
            difference = totals.get((m["match_id"], winner), 0) - totals.get((m["match_id"], loser), 0)
            if difference != m["margin_runs"]:
                rows.append({"match_id": m["match_id"], "kind": "runs",
                             "recorded": m["margin_runs"], "from_totals": difference})
        elif pd.notna(m["margin_wkts"]):
            final = innings[(innings["match_id"] == m["match_id"]) & (innings["batting_team"] == winner)]
            if final.empty:
                continue
            last = final.loc[final["innings_no"].idxmax()]
            remaining = 10 - last["wickets"]
            if remaining != m["margin_wkts"]:
                rows.append({"match_id": m["match_id"], "kind": "wickets",
                             "recorded": m["margin_wkts"], "from_totals": remaining})
    return pd.DataFrame(rows, columns=["match_id", "kind", "recorded", "from_totals"])


def main():
    parser = argparse.ArgumentParser(description="Sanity checks on the processed tables.")
    parser.add_argument("--processed", default="data/processed")
    args = parser.parse_args()
    folder = Path(args.processed)
    matches = pd.read_csv(folder / "matches.csv", dtype={"match_id": str})
    innings = pd.read_csv(folder / "innings.csv", dtype={"match_id": str})

    print(f"Tests: {len(matches):,}  from {matches['date_start'].min()} to {matches['date_start'].max()}")
    print("Results:", matches["result_type"].value_counts().to_dict())

    wins = matches[matches["result_type"] == "win"]
    with_margin = wins["margin_runs"].notna() | wins["margin_wkts"].notna()
    bad = margin_mismatches(matches, innings)
    print(f"Wins checked against the recorded margin: {with_margin.sum():,}  mismatches: {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string(index=False))

    pair = ["England", "Australia"]
    ashes = matches[matches["team1"].isin(pair) & matches["team2"].isin(pair)]
    print(f"\nEngland v Australia Tests: {len(ashes)}")
    print("  results:", ashes["winner"].fillna(ashes["result_type"]).value_counts().to_dict())

    teams = sorted(set(matches["team1"]) | set(matches["team2"]))
    print(f"\n{len(teams)} team names. Look for the same team spelled two ways:")
    print(", ".join(teams))


if __name__ == "__main__":
    main()