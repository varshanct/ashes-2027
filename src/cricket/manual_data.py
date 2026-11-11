"""Load and check the hand-filled CSV files in data/manual/ for the 2027 Ashes."""

import argparse
from pathlib import Path

import pandas as pd

TEAMS = ["England", "Australia"]
N_TESTS = 5
TEST_LENGTH_DAYS = 5
MIN_BOWLING_OPTIONS = 4
AVAIL_COLS = ["p_available_t1", "p_available_t2", "p_available_t3", "p_available_t4", "p_available_t5"]

# The only values allowed in these columns.
ALLOWED = {
    "role": {"batter", "bowler", "allrounder", "keeper"},
    "bowling_type": {"pace", "spin", "none"},
    "squad_tier": {"core", "fringe", "outside"},
    "status": {"fit", "injured", "doubtful", "retired_test", "unavailable"},
}

# Every file and the columns it must have, in order.
REQUIRED_COLUMNS = {
    "squads_2027.csv": ["team", "player_name", "cricsheet_id", "role", "bowling_type", "squad_tier",
                        "expected_batting_position", "source_url", "last_updated"],
    "availability.csv": ["cricsheet_id", "player_name", "status", "issue", "expected_return",
                         *AVAIL_COLS, "source_url", "last_updated"],
    "fixtures_2027.csv": ["test_no", "date_start", "date_end", "venue", "home_team", "away_team",
                          "source_url"],
    "venues_2027.csv": ["venue", "city", "source_url"],
    "weather_climate.csv": ["venue", "month", "p_rain_day", "avg_max_temp_c", "source_url"],
}


def load_all(folder):
    """Read the five files into a dictionary of tables, keeping player IDs as text."""
    return {name: pd.read_csv(Path(folder) / name, dtype={"cricsheet_id": str})
            for name in REQUIRED_COLUMNS}


def bad_dates(series):
    """Return the values that are filled in but are not YYYY-MM-DD dates."""
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    return list(series[parsed.isna() & series.notna()])


def check_columns(t):
    """Every file must have every column it needs."""
    problems = []
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in t[name].columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
    return problems

def check_squads(sq):
    """Squad rows: allowed values, real teams, IDs present and unique, sensible batting positions."""
    problems = []
    f = "squads_2027.csv"
    for col in ["role", "bowling_type", "squad_tier"]:
        unknown = sorted(set(sq[col].dropna()) - ALLOWED[col])
        if unknown:
            problems.append(f"{f}: column '{col}' has unknown values {unknown}; "
                            f"allowed {sorted(ALLOWED[col])}")
    for team in sorted(set(sq["team"].dropna()) - set(TEAMS)):
        problems.append(f"{f}: unknown team '{team}'; use England or Australia")
    for team in TEAMS:
        if (sq["team"] == team).sum() == 0:
            problems.append(f"{f}: {team} has no players yet")
    for name in sq.loc[sq["cricsheet_id"].isna(), "player_name"]:
        problems.append(f"{f}: '{name}' has no cricsheet_id")
    repeated = sq.loc[sq.duplicated("cricsheet_id", keep=False), "cricsheet_id"].dropna()
    for pid in sorted(set(repeated)):
        problems.append(f"{f}: player ID {pid} appears more than once")
    for raw in bad_dates(sq["last_updated"]):
        problems.append(f"{f}: '{raw}' in last_updated is not a YYYY-MM-DD date")

    pos = pd.to_numeric(sq["expected_batting_position"], errors="coerce")
    invalid = sq["expected_batting_position"].notna() & (
        pos.isna() | (pos % 1 != 0) | (pos < 1) | (pos > 11))
    for _, row in sq[invalid].iterrows():
        problems.append(f"{f}: {row['player_name']} has expected_batting_position "
                        f"'{row['expected_batting_position']}', must be blank or 1-11")
    return problems

def check_xi(sq):
    """Each team's expected XI must be a balanced Test side."""
    problems = []
    f = "squads_2027.csv"
    pos = pd.to_numeric(sq["expected_batting_position"], errors="coerce")
    valid = pos.notna() & (pos % 1 == 0) & (pos >= 1) & (pos <= 11)
    for team in TEAMS:
        in_team = sq["team"] == team
        if in_team.sum() == 0:
            continue  # check_squads already reports a team with no players
        xi = sq[in_team & valid]
        slots = pos[in_team & valid].astype(int)
        missing = sorted(set(range(1, 12)) - set(slots))
        doubled = sorted(set(slots[slots.duplicated()]))
        if missing:
            problems.append(f"{f}: {team} expected XI is missing batting positions {missing}")
        if doubled:
            problems.append(f"{f}: {team} has more than one player at batting positions {doubled}")
        if (xi["role"] == "keeper").sum() == 0:
            problems.append(f"{f}: {team} expected XI has no wicketkeeper")
        bowlers = xi["role"].isin(["bowler", "allrounder"]).sum()
        if bowlers < MIN_BOWLING_OPTIONS:
            problems.append(f"{f}: {team} expected XI has {bowlers} bowling options; a Test side "
                            f"needs at least {MIN_BOWLING_OPTIONS} (role bowler or allrounder)")
    return problems

def check_availability(av, sq):
    """Injury rows: allowed status, probabilities between 0 and 1, players who are in a squad."""
    problems = []
    f = "availability.csv"
    unknown = sorted(set(av["status"].dropna()) - ALLOWED["status"])
    if unknown:
        problems.append(f"{f}: column 'status' has unknown values {unknown}; "
                        f"allowed {sorted(ALLOWED['status'])}")
    for col in AVAIL_COLS:
        values = pd.to_numeric(av[col], errors="coerce")
        out = av[values.isna() | (values < 0) | (values > 1)]
        for _, row in out.iterrows():
            problems.append(f"{f}: {row['player_name']} has {col}={row[col]}, must be between 0 and 1")
    probs = av[AVAIL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    retired = av[(av["status"] == "retired_test") & (probs > 0).any(axis=1)]
    for name in retired["player_name"]:
        problems.append(f"{f}: {name} is retired_test, so every p_available column must be 0")
    for name in av.loc[~av["cricsheet_id"].isin(sq["cricsheet_id"]), "player_name"]:
        problems.append(f"{f}: {name} is not in squads_2027.csv")
    for col in ["expected_return", "last_updated"]:
        for raw in bad_dates(av[col]):
            problems.append(f"{f}: '{raw}' in {col} is not a YYYY-MM-DD date")
    return problems

def check_fixtures(fx, ven):
    """The schedule: five Tests, each five days, in date order, at known grounds, England at home."""
    problems = []
    f = "fixtures_2027.csv"
    if sorted(fx["test_no"].tolist()) != list(range(1, N_TESTS + 1)):
        problems.append(f"{f}: test_no must be 1 to {N_TESTS}, each once; "
                        f"found {sorted(fx['test_no'].tolist())}")
    for col in ["date_start", "date_end"]:
        for raw in bad_dates(fx[col]):
            problems.append(f"{f}: '{raw}' in {col} is not a YYYY-MM-DD date")
    start = pd.to_datetime(fx["date_start"], errors="coerce", format="%Y-%m-%d")
    end = pd.to_datetime(fx["date_end"], errors="coerce", format="%Y-%m-%d")
    length = (end - start).dt.days + 1
    for no, days in zip(fx["test_no"], length):
        if pd.notna(days) and days != TEST_LENGTH_DAYS:
            problems.append(f"{f}: Test {no} lasts {int(days)} days; "
                            f"a Test is scheduled for {TEST_LENGTH_DAYS}")
    if start.notna().all() and list(fx.loc[start.sort_values().index, "test_no"]) != sorted(fx["test_no"]):
        problems.append(f"{f}: Tests are not in date order")
    for venue in sorted(set(fx["venue"]) - set(ven["venue"])):
        problems.append(f"{f}: venue '{venue}' is not in venues_2027.csv")
    if not ((fx["home_team"] == "England") & (fx["away_team"] == "Australia")).all():
        problems.append(f"{f}: every Test should have home_team England and away_team Australia")
    for venue in sorted(set(ven.loc[ven.duplicated("venue"), "venue"])):
        problems.append(f"venues_2027.csv: '{venue}' is listed more than once")
    return problems


def months_needed(fx):
    """Every (ground, month) pair that any day of any Test falls in."""
    needed = set()
    for _, row in fx.iterrows():
        start = pd.to_datetime(row["date_start"], errors="coerce", format="%Y-%m-%d")
        end = pd.to_datetime(row["date_end"], errors="coerce", format="%Y-%m-%d")
        if pd.isna(start) or pd.isna(end):
            continue
        for day in pd.date_range(start, end):
            needed.add((row["venue"], day.month))
    return needed

def check_weather(wx, fx):
    """Climate rows: filled in, rain chance between 0 and 1, and one row for every ground-month played."""
    problems = []
    f = "weather_climate.csv"
    for _, row in wx.iterrows():
        where = f"{row['venue']} in month {row['month']}"
        if not 1 <= row["month"] <= 12:
            problems.append(f"{f}: month {row['month']} for {row['venue']} is not 1-12")
        if pd.isna(row["p_rain_day"]):
            problems.append(f"{f}: fill in p_rain_day for {where}")
        elif not 0 <= row["p_rain_day"] <= 1:
            problems.append(f"{f}: p_rain_day {row['p_rain_day']} for {where} must be 0-1 "
                            f"(write 0.35, not 35)")
        if pd.isna(row["avg_max_temp_c"]):
            problems.append(f"{f}: fill in avg_max_temp_c for {where}")
        if pd.isna(row["source_url"]):
            problems.append(f"{f}: fill in source_url for {where}")
    have = set(zip(wx["venue"], wx["month"]))
    for venue, month in sorted(months_needed(fx) - have):
        problems.append(f"{f}: no row for {venue} in month {month}")
    return problems

def validate(t):
    """Run every check. An empty list means everything is complete and consistent."""
    problems = check_columns(t)
    if problems:
        return problems  # without the right columns, nothing else can be checked safely
    sq, av = t["squads_2027.csv"], t["availability.csv"]
    fx, ven, wx = t["fixtures_2027.csv"], t["venues_2027.csv"], t["weather_climate.csv"]
    problems += check_squads(sq)
    problems += check_xi(sq)
    problems += check_availability(av, sq)
    problems += check_fixtures(fx, ven)
    problems += check_weather(wx, fx)
    return problems


def main():
    parser = argparse.ArgumentParser(description="Check the hand-filled files in data/manual/.")
    parser.add_argument("--folder", default="data/manual")
    args = parser.parse_args()
    problems = validate(load_all(args.folder))
    if problems:
        print(f"{len(problems)} thing(s) to fix or fill in:")
        for number, problem in enumerate(problems, start=1):
            print(f"  {number}. {problem}")
        raise SystemExit(1)
    print("All manual data files are complete and passed every check.")


if __name__ == "__main__":
    main()
