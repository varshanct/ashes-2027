import pytest

from cricket.manual_data import check_squads


def problems_with(manual, text):
    return [p for p in check_squads(manual["squads_2027.csv"]) if text in p]


def test_complete_squads_pass(manual):
    assert check_squads(manual["squads_2027.csv"]) == []


def test_unknown_team(manual):
    manual["squads_2027.csv"].loc[0, "team"] = "India"
    assert problems_with(manual, "unknown team 'India'")


def test_team_with_no_players(manual):
    sq = manual["squads_2027.csv"]
    manual["squads_2027.csv"] = sq[sq["team"] == "England"]
    assert problems_with(manual, "Australia has no players yet")


def test_unknown_role(manual):
    manual["squads_2027.csv"].loc[0, "role"] = "batsman"
    assert problems_with(manual, "unknown values ['batsman']")


def test_missing_id(manual):
    manual["squads_2027.csv"].loc[0, "cricsheet_id"] = None
    assert problems_with(manual, "has no cricsheet_id")


def test_duplicate_player(manual):
    sq = manual["squads_2027.csv"]
    sq.loc[1, "cricsheet_id"] = sq.loc[0, "cricsheet_id"]
    assert problems_with(manual, "appears more than once")


def test_bad_last_updated_date(manual):
    manual["squads_2027.csv"].loc[0, "last_updated"] = "09/11/2026"
    assert problems_with(manual, "'09/11/2026' in last_updated")


@pytest.mark.parametrize("value", [0, 12, 2.5])
def test_bad_batting_position(manual, value):
    manual["squads_2027.csv"].loc[0, "expected_batting_position"] = value
    assert problems_with(manual, "must be blank or 1-11")