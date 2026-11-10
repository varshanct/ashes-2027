"""Turn raw Cricsheet Test-match files into four tidy tables."""

import argparse
import json
from pathlib import Path

import pandas as pd

# One official spelling per team.
TEAM_ALIASES = {
    "U.A.E.": "United Arab Emirates",
    "UAE": "United Arab Emirates",
}


def clean_team(name):
    """Return the one official spelling of a team name."""
    name = name.strip()
    return TEAM_ALIASES.get(name, name)

def parse_outcome(outcome):
    """Read the result of a match into simple columns."""
    result = {
        "winner": None,
        "margin_runs": None,
        "margin_wkts": None,
        "margin_innings": None,
        "result_type": "win",
    }
    if outcome.get("winner") is not None:
        result["winner"] = clean_team(outcome["winner"])
        by = outcome.get("by", {})
        result["margin_runs"] = by.get("runs")
        result["margin_wkts"] = by.get("wickets")
        result["margin_innings"] = by.get("innings")
    else:
        # Cricsheet writes "draw", "tie" or "no result" here.
        result["result_type"] = outcome.get("result", "unknown").replace(" ", "_")
    return result

    # Ways of leaving the crease that are not dismissals.
NOT_DISMISSALS = {"retired hurt", "retired not out"}


def parse_match(path):
    """Read one Cricsheet Test-match file into rows for four tables."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    info = data["info"]
    match_id = Path(path).stem
    teams = [clean_team(t) for t in info["teams"]]
    registry = info.get("registry", {}).get("people", {})
    toss = info.get("toss", {})
    dates = sorted(info["dates"])

    innings_rows = []
    delivery_rows = []
    previous_team = None

    for innings_no, inn in enumerate(data.get("innings", []), start=1):
        batting_team = clean_team(inn["team"])
        runs = 0
        wickets = 0
        legal_balls = 0

        for over_block in inn.get("overs", []):
            over = over_block["over"]
            ball_in_over = 0
            for d in over_block["deliveries"]:
                extras = d.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not (is_wide or is_noball)
                if is_legal:
                    ball_in_over += 1
                    legal_balls += 1

                dismissals = [w for w in d.get("wickets", []) if w["kind"] not in NOT_DISMISSALS]
                runs_total = d["runs"]["total"]
                runs += runs_total
                wickets += len(dismissals)

                delivery_rows.append({
                    "match_id": match_id,
                    "innings_no": innings_no,
                    "batting_team": batting_team,
                    "over": over,
                    "ball_in_over": ball_in_over,
                    "batter": d["batter"],
                    "batter_id": registry.get(d["batter"]),
                    "bowler": d["bowler"],
                    "bowler_id": registry.get(d["bowler"]),
                    "runs_bat": d["runs"]["batter"],
                    "runs_extras": d["runs"]["extras"],
                    "runs_total": runs_total,
                    "is_wide": is_wide,
                    "is_noball": is_noball,
                    "is_legal": is_legal,
                    "is_wicket": len(dismissals) > 0,
                    "wicket_kind": dismissals[0]["kind"] if dismissals else None,
                    "player_out": dismissals[0]["player_out"] if dismissals else None,
                })

        penalty = inn.get("penalty_runs", {})
        penalty_runs = penalty.get("pre", 0) + penalty.get("post", 0)
        target = inn.get("target", {})
        innings_rows.append({
            "match_id": match_id,
            "innings_no": innings_no,
            "batting_team": batting_team,
            "runs": runs + penalty_runs,
            "penalty_runs": penalty_runs,
            "wickets": wickets,
            "legal_balls": legal_balls,
            "declared": bool(inn.get("declared", False)),
            "forfeited": bool(inn.get("forfeited", False)),
            "follow_on": innings_no == 3 and batting_team == previous_team,
            "target_runs": target.get("runs"),
        })
        previous_team = batting_team

    total_legal_balls = sum(r["legal_balls"] for r in innings_rows)
    match_row = {
        "match_id": match_id,
        "date_start": dates[0],
        "date_end": dates[-1],
        "days_played": len(dates),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": clean_team(toss["winner"]) if "winner" in toss else None,
        "toss_decision": toss.get("decision"),
        "overs_bowled": round(total_legal_balls / 6, 1),
        **parse_outcome(info.get("outcome", {})),
    }

    appearance_rows = []
    for team, players in info.get("players", {}).items():
        for name in players:
            appearance_rows.append({
                "match_id": match_id,
                "team": clean_team(team),
                "player_name": name,
                "player_id": registry.get(name),
            })

    return {
        "matches": [match_row],
        "innings": innings_rows,
        "deliveries": delivery_rows,
        "appearances": appearance_rows,
    }

def ingest_folder(raw_dir):
    """Read every men's Test file in a folder into four tables."""
    tables = {"matches": [], "innings": [], "deliveries": [], "appearances": []}
    files = sorted(Path(raw_dir).glob("*.json"))
    skipped = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)["info"]
        if info.get("match_type") != "Test" or info.get("gender") != "male":
            skipped += 1
            continue
        for name, rows in parse_match(path).items():
            tables[name].extend(rows)

    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    for col in ["date_start", "date_end"]:
        frames["matches"][col] = pd.to_datetime(frames["matches"][col])
    frames["matches"] = frames["matches"].sort_values(["date_start", "match_id"]).reset_index(drop=True)
    print(f"Read {len(files)} files, kept {len(frames['matches'])} men's Tests, skipped {skipped}.")
    return frames


def save_tables(frames, out_dir):
    """Write each table to a CSV file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv  ({len(df):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Turn Cricsheet Test files into tables.")
    parser.add_argument("--raw", default="data/raw/tests_json")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    save_tables(ingest_folder(args.raw), args.out)


if __name__ == "__main__":
    main()  