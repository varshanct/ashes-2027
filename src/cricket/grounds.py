"""Profile the five 2027 Ashes grounds from every Test played there."""

import argparse
from pathlib import Path

import pandas as pd

from cricket.charts import plot_ground_profiles, plot_results

# Words to search for in Cricsheet's venue names, for each ECB ground name.
SEARCH_WORDS = {
    "Trent Bridge": "Trent Bridge",
    "Lord's": "Lord",
    "Edgbaston": "Edgbaston",
    "Utilita Bowl": "Bowl",
    "Kia Oval": "Oval",
}


def venue_candidates(matches, word):
    """Cricsheet venue names containing a word, with their city, number of Tests and years played."""
    found = matches[matches["venue"].str.contains(word, case=False, regex=False, na=False)]
    dates = pd.to_datetime(found["date_start"])
    summary = (found.assign(year=dates.dt.year)
                    .groupby(["venue", "city"], dropna=False)
                    .agg(tests=("match_id", "nunique"), first=("year", "min"), last=("year", "max"))
                    .reset_index()
                    .sort_values("tests", ascending=False)
                    .reset_index(drop=True))
    return summary


def check_aliases(aliases, venues, matches):
    """Every ECB ground needs at least one Cricsheet name, and every Cricsheet name must exist."""
    f = "venue_aliases.csv"
    problems = []
    for col in ["cricsheet_venue", "ground"]:
        if col not in aliases.columns:
            return [f"{f}: missing column '{col}'"]
    for ground in sorted(set(aliases["ground"]) - set(venues["venue"])):
        problems.append(f"{f}: '{ground}' is not a ground in venues_2027.csv")
    for ground in venues["venue"]:
        if (aliases["ground"] == ground).sum() == 0:
            problems.append(f"{f}: {ground} has no Cricsheet venue name yet")
    for name in sorted(set(aliases["cricsheet_venue"]) - set(matches["venue"])):
        problems.append(f"{f}: '{name}' does not appear as a venue in matches.csv")
    for name in sorted(set(aliases.loc[aliases.duplicated("cricsheet_venue"), "cricsheet_venue"])):
        problems.append(f"{f}: '{name}' is listed more than once")
    return problems


def attach_ground(matches, aliases):
    """Add a 'ground' column with the ECB name; empty for Tests played anywhere else."""
    lookup = dict(zip(aliases["cricsheet_venue"], aliases["ground"]))
    return matches.assign(ground=matches["venue"].map(lookup))

def ground_profiles(matches, innings, deliveries, aliases, order, since=None):
    """One row per ground: how Tests there have tended to play out."""
    m = attach_ground(matches, aliases)
    m = m[m["ground"].notna() & m["result_type"].isin(["win", "draw"])]
    if since is not None:
        m = m[pd.to_datetime(m["date_start"]) >= pd.Timestamp(since)]

    first = innings[innings["innings_no"] == 1].set_index("match_id")
    m = m.assign(bat_first=m["match_id"].map(first["batting_team"]),
                 first_innings_runs=m["match_id"].map(first["runs"]))
    wickets = deliveries[deliveries["is_wicket"] & deliveries["match_id"].isin(m["match_id"])]
    wickets = wickets.assign(ground=wickets["match_id"].map(m.set_index("match_id")["ground"]))

    rows = []
    for ground in order:
        g = m[m["ground"] == ground]
        w = wickets[wickets["ground"] == ground]
        kinds = w["wicket_kind"].replace({"caught and bowled": "caught"})
        rows.append({
            "ground": ground,
            "tests": len(g),
            "draw_rate": (g["result_type"] == "draw").mean() if len(g) else float("nan"),
            "bat_first_win_rate": (g["winner"] == g["bat_first"]).mean() if len(g) else float("nan"),
            "toss_winner_win_rate": (g["winner"] == g["toss_winner"]).mean() if len(g) else float("nan"),
            "avg_first_innings_runs": g["first_innings_runs"].mean(),
            "avg_overs_bowled": g["overs_bowled"].mean(),
            "wickets": len(w),
            "share_caught": (kinds == "caught").mean() if len(w) else float("nan"),
            "share_bowled": (kinds == "bowled").mean() if len(w) else float("nan"),
            "share_lbw": (kinds == "lbw").mean() if len(w) else float("nan"),
        })
    return pd.DataFrame(rows)


def team_record_at_grounds(matches, aliases, team, order, opponent=None, since=None):
    """One row per ground: a team's wins, draws and losses there, optionally against one opponent."""
    m = attach_ground(matches, aliases)
    m = m[m["ground"].notna() & m["result_type"].isin(["win", "draw"])]
    m = m[(m["team1"] == team) | (m["team2"] == team)]
    if opponent is not None:
        m = m[(m["team1"] == opponent) | (m["team2"] == opponent)]
    if since is not None:
        m = m[pd.to_datetime(m["date_start"]) >= pd.Timestamp(since)]
    rows = []
    for ground in order:
        g = m[m["ground"] == ground]
        wins = (g["winner"] == team).sum()
        draws = (g["result_type"] == "draw").sum()
        losses = len(g) - wins - draws
        rows.append({"ground": ground, "tests": len(g), "wins": wins, "draws": draws, "losses": losses})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Profile the five Ashes grounds and draw the charts.")
    parser.add_argument("--find", action="store_true", help="list Cricsheet venue names for each ground")
    parser.add_argument("--since", default=None, help="only use Tests starting on or after this date (YYYY-MM-DD)")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--manual", default="data/manual")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    processed, manual, figures = Path(args.processed), Path(args.manual), Path(args.figures)
    matches = pd.read_csv(processed / "matches.csv", dtype={"match_id": str})

    if args.find:
        for ground, word in SEARCH_WORDS.items():
            print(f"\n=== {ground}  (searching for '{word}')")
            print(venue_candidates(matches, word).to_string(index=False))
        return

    venues = pd.read_csv(manual / "venues_2027.csv")
    aliases = pd.read_csv(manual / "venue_aliases.csv")
    problems = check_aliases(aliases, venues, matches)
    if problems:
        print(f"{len(problems)} thing(s) to fix in venue_aliases.csv:")
        for number, problem in enumerate(problems, start=1):
            print(f"  {number}. {problem}")
        raise SystemExit(1)

    innings = pd.read_csv(processed / "innings.csv", dtype={"match_id": str})
    deliveries = pd.read_csv(processed / "deliveries.csv", dtype={"match_id": str},
                             usecols=["match_id", "is_wicket", "wicket_kind"])
    order = venues["venue"].tolist()
    figures.mkdir(exist_ok=True)
    suffix = f"_since_{args.since[:4]}" if args.since else ""
    period = f"Tests since {args.since[:4]}" if args.since else "all Tests in the data"
    pd.set_option("display.width", 200)

    profiles = ground_profiles(matches, innings, deliveries, aliases, order, since=args.since)
    profiles.round(3).to_csv(figures / f"ground_profiles{suffix}.csv", index=False)
    print(f"\nGround profiles ({period}):")
    print(profiles.round(2).to_string(index=False))

    england = team_record_at_grounds(matches, aliases, "England", order, since=args.since)
    print(f"\nEngland at each ground, all opponents ({period}):")
    print(england.to_string(index=False))

    ashes = team_record_at_grounds(matches, aliases, "England", order, opponent="Australia", since=args.since)
    print(f"\nEngland v Australia at each ground ({period}); losses are Australia wins:")
    print(ashes.to_string(index=False))

    saved = [
        plot_ground_profiles(profiles, figures / f"ground_profiles{suffix}.png",
                             f"How the five Ashes grounds have played ({period})"),
        plot_results(england, figures / f"england_at_grounds{suffix}.png",
                     f"England's results at each ground, all opponents ({period})",
                     labels=("England win", "Draw", "England loss"), as_share=True),
        plot_results(ashes, figures / f"ashes_at_grounds{suffix}.png",
                     f"England v Australia at each ground ({period})",
                     labels=("England win", "Draw", "Australia win")),
    ]
    print("\nSaved:")
    for path in [figures / f"ground_profiles{suffix}.csv", *saved]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
