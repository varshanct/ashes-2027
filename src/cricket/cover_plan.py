"""Check the cover plan: who comes in, in order of preference, when a player is unavailable."""

import argparse

import pandas as pd

from cricket.manual_data import TEAMS

NEEDS = {"batter", "pace", "spin", "keeper"}
COLUMNS = ["team", "need", "cover_order", "cricsheet_id", "player_name", "reason"]


def check_cover_plan(plan, squads):
    """Return a list of problems with the cover plan. An empty list means it is complete and consistent."""
    f = "cover_plan.csv"
    missing = [c for c in COLUMNS if c not in plan.columns]
    if missing:
        return [f"{f}: missing columns {missing}"]
    problems = []
    for need in sorted(set(plan["need"].dropna()) - NEEDS):
        problems.append(f"{f}: unknown need '{need}'; allowed {sorted(NEEDS)}")
    for team in sorted(set(plan["team"].dropna()) - set(TEAMS)):
        problems.append(f"{f}: unknown team '{team}'; use England or Australia")

    known = squads.dropna(subset=["cricsheet_id"]).set_index("cricsheet_id")
    for _, row in plan.iterrows():
        pid, name, need = row["cricsheet_id"], row["player_name"], row["need"]
        if pd.isna(pid) or pid not in known.index:
            problems.append(f"{f}: {name} is not in squads_2027.csv")
            continue
        player = known.loc[pid]
        if player["team"] != row["team"]:
            problems.append(f"{f}: {name} is in the {player['team']} squad, not {row['team']}")
        if need in ("pace", "spin") and player["bowling_type"] != need:
            problems.append(f"{f}: {name} is {need} cover but bowls {player['bowling_type']}")
        if need == "batter" and player["role"] not in ("batter", "allrounder"):
            problems.append(f"{f}: {name} is batter cover but is a {player['role']}")
        if need == "keeper" and player["role"] == "keeper":
            problems.append(f"{f}: {name} is a first-choice keeper; keeper cover must be someone else")
        if need != "keeper" and player["squad_tier"] == "core":
            problems.append(f"{f}: {name} is already in the expected XI, so cannot be {need} cover")

    for (team, need), group in plan.groupby(["team", "need"]):
        orders = sorted(pd.to_numeric(group["cover_order"], errors="coerce").tolist())
        if orders != list(range(1, len(group) + 1)):
            problems.append(f"{f}: {team} {need} cover_order must be 1 to {len(group)}; found {orders}")
    for team in TEAMS:
        for need in sorted(NEEDS):
            if ((plan["team"] == team) & (plan["need"] == need)).sum() == 0:
                problems.append(f"{f}: {team} has no {need} cover")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Check the cover plan against the squads.")
    parser.add_argument("--plan", default="data/manual/cover_plan.csv")
    parser.add_argument("--squads", default="data/manual/squads_2027.csv")
    args = parser.parse_args()
    plan = pd.read_csv(args.plan, dtype={"cricsheet_id": str})
    squads = pd.read_csv(args.squads, dtype={"cricsheet_id": str})
    problems = check_cover_plan(plan, squads)
    if problems:
        print(f"{len(problems)} thing(s) to fix in the cover plan:")
        for number, problem in enumerate(problems, start=1):
            print(f"  {number}. {problem}")
        raise SystemExit(1)
    print("The cover plan is complete and passed every check.")


if __name__ == "__main__":
    main()