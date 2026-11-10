"""Small, hand-built Test matches in the real Cricsheet JSON layout.

Each fake match is tiny but exercises the tricky parts of Test cricket:
innings victories, declarations, the follow-on, a fourth-innings chase,
penalty runs, a batter retiring hurt, a forfeited innings, a draw and a washout.
Every total below is worked out by hand in the comments, so each test
knows exactly what the right answer is.
"""

import json
from pathlib import Path

import pytest

ENG, AUS = "England", "Australia"
REGISTRY = {
    "E Bat": "eeee0001", "E Bat2": "eeee0002", "E Bowl": "eeee0003",
    "A Bat": "aaaa0001", "A Bat2": "aaaa0002", "A Bowl": "aaaa0003",
}


def ball(batter, bowler, bat_runs=0, extras=None, wicket=None):
    """Build one delivery the way Cricsheet writes it."""
    extras = extras or {}
    extra_runs = sum(extras.values())
    d = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "x",
        "runs": {"batter": bat_runs, "extras": extra_runs, "total": bat_runs + extra_runs},
    }
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"player_out": batter, "kind": wicket}]
    return d


def eng_ball(runs=0, **kw):
    return ball("E Bat", "A Bowl", runs, **kw)


def aus_ball(runs=0, **kw):
    return ball("A Bat", "E Bowl", runs, **kw)


def innings(team, deliveries, **flags):
    """Put deliveries into one over (enough for these tiny tests)."""
    inn = {"team": team, "overs": [{"over": 0, "deliveries": deliveries}] if deliveries else []}
    inn.update(flags)
    return inn


def match(outcome, innings_list, dates, match_type="Test", gender="male"):
    return {
        "meta": {"data_version": "1.1.0"},
        "info": {
            "dates": dates,
            "event": {"name": "The Ashes"},
            "gender": gender,
            "match_type": match_type,
            "outcome": outcome,
            "players": {ENG: ["E Bat", "E Bat2", "E Bowl"], AUS: ["A Bat", "A Bat2", "A Bowl"]},
            "registry": {"people": REGISTRY},
            "teams": [ENG, AUS],
            "toss": {"decision": "bat", "winner": ENG},
            "venue": "Edgbaston, Birmingham",
            "city": "Birmingham",
        },
        "innings": innings_list,
    }


def build_matches():
    return {
        # England win by an innings and 12 runs, enforcing the follow-on.
        #   Inns 1 England 30 declared (5 sixes)
        #   Inns 2 Australia 10 (4, 4, wicket, 2)
        #   Inns 3 Australia 8, following on (4, 4, wicket)
        #   England 30 vs Australia 10 + 8 = 18, so an innings and 12 runs.
        "3001": match({"winner": ENG, "by": {"innings": 1, "runs": 12}},
                      [innings(ENG, [eng_ball(6) for _ in range(5)], declared=True),
                       innings(AUS, [aus_ball(4), aus_ball(4), aus_ball(0, wicket="bowled"), aus_ball(2)]),
                       innings(AUS, [aus_ball(4), aus_ball(4), aus_ball(0, wicket="lbw")])],
                      ["2023-06-16", "2023-06-17", "2023-06-18"]),

        # Australia win a fourth-innings chase.
        #   Inns 1 England 20 (6, 6, 4, 4)
        #   Inns 2 Australia 15 (6, 6, 3)
        #   Inns 3 England: 6, wide (1), 4, no-ball (1), retired hurt (0) = 12 off the balls
        #           + 5 penalty runs = 17. Retired hurt is NOT a wicket. Legal balls = 3.
        #   Target = 20 + 17 - 15 + 1 = 23
        #   Inns 4 Australia 24 (6, 6, 6, wicket, 6)
        "3002": match({"winner": AUS, "by": {"wickets": 9}},
                      [innings(ENG, [eng_ball(6), eng_ball(6), eng_ball(4), eng_ball(4)]),
                       innings(AUS, [aus_ball(6), aus_ball(6), aus_ball(3)]),
                       innings(ENG, [eng_ball(6), eng_ball(0, extras={"wides": 1}), eng_ball(4),
                                     eng_ball(0, extras={"noballs": 1}), eng_ball(0, wicket="retired hurt")],
                               penalty_runs={"pre": 5}),
                       innings(AUS, [aus_ball(6), aus_ball(6), aus_ball(6), aus_ball(0, wicket="caught"),
                                     aus_ball(6)], target={"runs": 23})],
                      ["2023-07-06", "2023-07-07", "2023-07-08", "2023-07-09"]),

        # A rain-affected draw over all five days, with England forfeiting their second innings.
        "3003": match({"result": "draw"},
                      [innings(ENG, [eng_ball(6) for _ in range(3)]),
                       innings(AUS, [aus_ball(6), aus_ball(6)]),
                       innings(ENG, [], forfeited=True),
                       innings(AUS, [aus_ball(6)])],
                      ["2023-07-27", "2023-07-28", "2023-07-29", "2023-07-30", "2023-07-31"]),

        # Washed out without a ball bowled.
        "3004": match({"result": "no result"}, [], ["2023-08-10"]),

        # These two must be thrown away: an ODI and a women's Test.
        "2001": match({"winner": ENG, "by": {"runs": 5}}, [innings(ENG, [eng_ball(6)])],
                      ["2023-09-01"], match_type="ODI"),
        "2002": match({"winner": ENG, "by": {"runs": 5}}, [innings(ENG, [eng_ball(6)])],
                      ["2023-06-22"], gender="female"),
    }


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    for name, content in build_matches().items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content), encoding="utf-8")
    return tmp_path

# ---------------------------------------------------------------------------
# A complete, correct set of made-up manual files.
# Each test starts from this and breaks one thing on purpose.
# ---------------------------------------------------------------------------

import pandas as pd

# A balanced made-up XI: five batters, an allrounder, a keeper at 7, an allrounder, three bowlers.
FAKE_XI = [(1, "batter", "none"), (2, "batter", "none"), (3, "batter", "none"),
           (4, "batter", "none"), (5, "batter", "none"), (6, "allrounder", "pace"),
           (7, "keeper", "none"), (8, "allrounder", "spin"), (9, "bowler", "pace"),
           (10, "bowler", "pace"), (11, "bowler", "pace")]


def fake_squad(team, prefix):
    rows = []
    for pos, role, bowl in FAKE_XI:
        rows.append({"team": team, "player_name": f"{team} Player {pos}",
                     "cricsheet_id": f"{prefix}{pos:07d}", "role": role, "bowling_type": bowl,
                     "squad_tier": "core", "expected_batting_position": pos,
                     "source_url": "https://example.com", "last_updated": "2026-11-09"})
    rows.append({"team": team, "player_name": f"{team} Reserve Seamer",
                 "cricsheet_id": f"{prefix}0000099", "role": "bowler", "bowling_type": "pace",
                 "squad_tier": "fringe", "expected_batting_position": None,
                 "source_url": "https://example.com", "last_updated": "2026-11-09"})
    return rows


@pytest.fixture
def manual():
    grounds = ["Trent Bridge", "Lord's", "Edgbaston", "Utilita Bowl", "Kia Oval"]
    cities = ["Nottingham", "London", "Birmingham", "Southampton", "London"]
    starts = ["2027-06-18", "2027-06-30", "2027-07-08", "2027-07-21", "2027-07-29"]
    ends = ["2027-06-22", "2027-07-04", "2027-07-12", "2027-07-25", "2027-08-02"]
    weather_rows = [("Trent Bridge", 6), ("Lord's", 6), ("Lord's", 7), ("Edgbaston", 7),
                    ("Utilita Bowl", 7), ("Kia Oval", 7), ("Kia Oval", 8)]
    return {
        "squads_2027.csv": pd.DataFrame(fake_squad("England", "e") + fake_squad("Australia", "a")),
        "availability.csv": pd.DataFrame([{
            "cricsheet_id": "e0000009", "player_name": "England Player 9", "status": "injured",
            "issue": "side strain", "expected_return": "2027-06-01",
            "p_available_t1": 0.6, "p_available_t2": 0.8, "p_available_t3": 0.9,
            "p_available_t4": 0.9, "p_available_t5": 0.9,
            "source_url": "https://example.com", "last_updated": "2026-11-09"}]),
        "fixtures_2027.csv": pd.DataFrame({
            "test_no": [1, 2, 3, 4, 5], "date_start": starts, "date_end": ends, "venue": grounds,
            "home_team": "England", "away_team": "Australia", "source_url": "https://example.com"}),
        "venues_2027.csv": pd.DataFrame({"venue": grounds, "city": cities,
                                         "source_url": "https://example.com"}),
        "weather_climate.csv": pd.DataFrame([
            {"venue": v, "month": m, "p_rain_day": 0.4, "avg_max_temp_c": 21.0,
             "source_url": "https://example.com"} for v, m in weather_rows]),
    }