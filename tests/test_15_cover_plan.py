import pandas as pd
import pytest

from cricket.cover_plan import check_cover_plan


def reserve(team, name, pid, role, bowl):
    return {"team": team, "player_name": name, "cricsheet_id": pid, "role": role, "bowling_type": bowl,
            "squad_tier": "fringe", "expected_batting_position": None,
            "source_url": "https://example.com", "last_updated": "2026-11-10"}


@pytest.fixture
def squads(manual):
    extra = [reserve("England", "England Reserve Batter", "e0000098", "batter", "none"),
             reserve("England", "England Reserve Spinner", "e0000097", "bowler", "spin"),
             reserve("Australia", "Australia Reserve Batter", "a0000098", "batter", "none"),
             reserve("Australia", "Australia Reserve Spinner", "a0000097", "bowler", "spin")]
    return pd.concat([manual["squads_2027.csv"], pd.DataFrame(extra)], ignore_index=True)


@pytest.fixture
def plan():
    rows = []
    for team, p in [("England", "e"), ("Australia", "a")]:
        rows += [
            {"team": team, "need": "batter", "cover_order": 1, "cricsheet_id": f"{p}0000098",
             "player_name": f"{team} Reserve Batter", "reason": "next batter in line"},
            {"team": team, "need": "pace", "cover_order": 1, "cricsheet_id": f"{p}0000099",
             "player_name": f"{team} Reserve Seamer", "reason": "next seamer in line"},
            {"team": team, "need": "spin", "cover_order": 1, "cricsheet_id": f"{p}0000097",
             "player_name": f"{team} Reserve Spinner", "reason": "only specialist spin reserve"},
            {"team": team, "need": "keeper", "cover_order": 1, "cricsheet_id": f"{p}0000003",
             "player_name": f"{team} Player 3", "reason": "keeps for his county"},
        ]
    return pd.DataFrame(rows)


def test_complete_plan_passes(plan, squads):
    assert check_cover_plan(plan, squads) == []


def test_player_not_in_squad(plan, squads):
    plan.loc[0, "cricsheet_id"] = "zzzz9999"
    assert check_cover_plan(plan, squads) == ["cover_plan.csv: England Reserve Batter is not in squads_2027.csv"]


def test_wrong_team(plan, squads):
    plan.loc[0, "team"] = "Australia"
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: England Reserve Batter is in the England squad, not Australia" in problems


def test_spinner_listed_as_pace_cover(plan, squads):
    plan.loc[1, ["cricsheet_id", "player_name"]] = ["e0000097", "England Reserve Spinner"]
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: England Reserve Spinner is pace cover but bowls spin" in problems


def test_bowler_listed_as_batter_cover(plan, squads):
    plan.loc[0, ["cricsheet_id", "player_name"]] = ["e0000099", "England Reserve Seamer"]
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: England Reserve Seamer is batter cover but is a bowler" in problems


def test_first_choice_keeper_cannot_be_keeper_cover(plan, squads):
    plan.loc[3, ["cricsheet_id", "player_name"]] = ["e0000007", "England Player 7"]
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: England Player 7 is a first-choice keeper; keeper cover must be someone else" in problems


def test_xi_player_cannot_be_bowling_cover(plan, squads):
    plan.loc[1, ["cricsheet_id", "player_name"]] = ["e0000009", "England Player 9"]
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: England Player 9 is already in the expected XI, so cannot be pace cover" in problems


def test_cover_order_must_run_from_1(plan, squads):
    plan.loc[1, "cover_order"] = 2
    assert check_cover_plan(plan, squads) == ["cover_plan.csv: England pace cover_order must be 1 to 1; found [2]"]


def test_every_need_must_be_covered(plan, squads):
    plan = plan[~((plan["team"] == "Australia") & (plan["need"] == "spin"))]
    assert check_cover_plan(plan, squads) == ["cover_plan.csv: Australia has no spin cover"]


def test_unknown_need(plan, squads):
    plan.loc[0, "need"] = "opener"
    problems = check_cover_plan(plan, squads)
    assert "cover_plan.csv: unknown need 'opener'; allowed ['batter', 'keeper', 'pace', 'spin']" in problems