import numpy as np
import pandas as pd
import pytest

from cricket.model_data import build_model_table, home_team_by_place, to_model_data


def match(mid, date, venue, city, t1, t2, result_type, winner=None):
    return {"match_id": mid, "date_start": date, "venue": venue, "city": city, "team1": t1, "team2": t2,
            "result_type": result_type, "winner": winner}


@pytest.fixture
def matches():
    rows = [
        # Lord's: England in all 5 Tests (one is a neutral final), so England are at home there.
        match("1", "2015-07-16", "Lord's", "London", "England", "Australia", "win", "Australia"),
        match("2", "2019-08-14", "Lord's", "London", "England", "Australia", "draw"),
        match("3", "2020-07-08", "Lord's", "London", "West Indies", "England", "win", "England"),
        match("4", "2021-06-18", "Lord's", "London", "India", "England", "win", "India"),
        match("5", "2025-06-11", "Lord's", "London", "South Africa", "Australia", "win", "South Africa"),
        match("6", "2023-06-01", "Lord's", "London", "England", "Ireland", "win", "England"),
        # Adelaide with no city recorded: falls back to the venue name.
        match("7", "2017-12-02", "Adelaide Oval", None, "Australia", "England", "win", "Australia"),
        match("8", "2018-12-06", "Adelaide Oval", None, "Australia", "India", "win", "India"),
        # A one-off at a place with only one Test: no home side can be told apart.
        match("9", "2019-03-01", "Somewhere", "Neutralville", "India", "Pakistan", "win", "India"),
        # Must be left out: World XI, and a Test with no result.
        match("10", "2005-10-14", "SCG", "Sydney", "Australia", "ICC World XI", "win", "Australia"),
        match("11", "2016-01-01", "Lord's", "London", "England", "Sri Lanka", "no_result"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def aliases():
    return pd.DataFrame({"cricsheet_venue": ["Lord's"], "ground": ["Lord's"]})


def test_home_team_by_place(matches):
    homes = home_team_by_place(matches)
    assert homes["London"] == "England"
    assert homes["Adelaide Oval"] == "Australia"
    assert "Neutralville" not in homes


def test_excluded_and_unfinished_tests_are_dropped(matches, aliases):
    table = build_model_table(matches, aliases)
    assert set(table["match_id"]) == {"1", "2", "3", "4", "5", "6", "7", "8", "9"}


def test_home_column(matches, aliases):
    table = build_model_table(matches, aliases).set_index("match_id")
    assert table.loc["1", "home"] == 1     # England (team1) at home
    assert table.loc["3", "home"] == -1    # England is team2 and at home
    assert table.loc["5", "home"] == 0     # neutral final at Lord's
    assert table.loc["7", "home"] == 1     # Australia at home in Adelaide
    assert table.loc["9", "home"] == 0     # no home side known


def test_result_is_from_team1_point_of_view(matches, aliases):
    table = build_model_table(matches, aliases).set_index("match_id")
    assert table.loc["1", "result"] == 0   # team1 England lost
    assert table.loc["2", "result"] == 1   # draw
    assert table.loc["6", "result"] == 2   # team1 England won


def test_rows_sorted_by_date(matches, aliases):
    table = build_model_table(matches, aliases)
    assert table["date_start"].is_monotonic_increasing


def test_model_data_arrays(matches, aliases):
    table = build_model_table(matches, aliases)
    data = to_model_data(table, extra_venues=["Kia Oval"])
    assert data.years == list(range(2015, 2026))         # every year, even ones with no Tests
    assert "Kia Oval" in data.venues                      # a 2027 ground with no Tests yet
    assert len(data.team_a) == len(table) == len(data.result)
    first = table.iloc[0]
    assert data.teams[data.team_a[0]] == first["team1"]
    assert data.years[data.year_idx[0]] == first["year"]
    assert data.venues[data.venue_idx[0]] == first["venue_key"]
    assert data.decade[0] == pytest.approx((first["year"] - 2019) / 10)
    assert np.issubdtype(data.result.dtype, np.integer)