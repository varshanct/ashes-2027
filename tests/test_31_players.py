import sys

import numpy as np
import pandas as pd
import pytest

from cricket import players
from cricket.ingest import ingest_folder, save_tables
from cricket.players import (
    PRIOR_OUTS,
    PRIOR_WICKETS,
    batting_positions,
    count_tests,
    player_ratings,
    recency_weights,
)

NEVER = float("inf")   # a half-life so long that every Test counts the same, for hand calculations


@pytest.fixture
def tables(raw_dir):
    return ingest_folder(raw_dir)


def rate(tables, as_of, half_life=NEVER):
    return player_ratings(tables["deliveries"], tables["appearances"], tables["matches"], as_of, half_life)


def test_recency_weights_halve_every_half_life():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2024-01-01", "2022-01-01"]))
    weights = recency_weights(dates, "2026-01-01", half_life=2)
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.25], atol=0.002)


def test_batting_counts_runs_and_real_dismissals_only(tables):
    r = rate(tables, "2024-01-01")
    # E Bat: 30 + 20 + 10 + 18 = 78 runs, never out (retiring hurt is not a dismissal).
    assert r.loc["eeee0001", "runs"] == 78 and r.loc["eeee0001", "outs"] == 0
    # A Bat: 10 + 8 + 15 + 24 + 12 + 6 = 75 runs, out three times (bowled, lbw, caught).
    assert r.loc["aaaa0001", "runs"] == 75 and r.loc["aaaa0001", "outs"] == 3
    assert r.loc["aaaa0001", "bat_avg"] == 25


def test_bowling_charges_wides_and_no_balls_to_the_bowler(tables):
    r = rate(tables, "2024-01-01")
    # A Bowl: 78 runs off the bat plus a wide and a no-ball = 80, and no wickets.
    assert r.loc["aaaa0003", "conceded"] == 80 and r.loc["aaaa0003", "wickets"] == 0
    assert r.loc["eeee0003", "conceded"] == 75 and r.loc["eeee0003", "wickets"] == 3


def test_ratings_are_pulled_towards_the_average(tables):
    r = rate(tables, "2024-01-01")
    bat_mean = (78 + 75) / 3                 # both batters open, so both are top order
    assert r.loc["eeee0001", "bat_rating"] == pytest.approx((78 + PRIOR_OUTS * bat_mean) / PRIOR_OUTS)
    assert r.loc["aaaa0001", "bat_rating"] == pytest.approx((75 + PRIOR_OUTS * bat_mean) / (3 + PRIOR_OUTS))
    bowl_mean = (75 + 80) / 3
    expected = (75 + PRIOR_WICKETS * bowl_mean) / (3 + PRIOR_WICKETS)
    assert r.loc["eeee0003", "bowl_rating"] == pytest.approx(expected)


def test_only_tests_before_the_date_are_used(tables):
    r = rate(tables, "2023-07-01")           # only the first Test (16 June) has started
    assert r.loc["eeee0001", "runs"] == 30
    assert r.loc["aaaa0001", "runs"] == 18 and r.loc["aaaa0001", "outs"] == 2
    assert count_tests(tables["appearances"], tables["matches"], "2023-07-01")["eeee0001"] == 1


def test_command_rates_every_squad_player(raw_dir, tmp_path, monkeypatch, capsys):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    manual.mkdir()
    pd.DataFrame({"team": ["England", "Australia"], "player_name": ["E Bat", "New Player"],
                  "cricsheet_id": ["eeee0001", "zzzz0001"], "role": ["batter", "batter"],
                  "squad_tier": ["core", "fringe"]}).to_csv(manual / "squads_2027.csv", index=False)
    pd.DataFrame(columns=players.FC_COLUMNS).to_csv(manual / "first_class.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["players", "--processed", str(processed), "--manual", str(manual),
                                      "--figures", str(tmp_path / "figures")])
    players.main()
    table = pd.read_csv(tmp_path / "figures" / "player_ratings_2027_squads.csv")
    assert table["tests"].tolist() == [4, 0]       # the washout counts as a Test played
    average = rate(ingest_folder(raw_dir), "2026-11-10", half_life=2).attrs["bat_mean_top"]
    new = table.iloc[1]                            # a player with no Tests gets the average ratings
    assert new["bat_rating"] == pytest.approx(average, abs=0.01)
    assert new["balls"] == 0
    assert "England:" in capsys.readouterr().out


def test_byes_are_not_charged_to_the_bowler():
    matches = pd.DataFrame({"match_id": ["1"], "date_start": ["2025-01-01"]})
    appearances = pd.DataFrame({"match_id": ["1"], "player_name": ["B"], "player_id": ["b1"]})
    deliveries = pd.DataFrame({          # 2 off the bat, then 4 byes, then bowled
        "match_id": ["1"] * 3, "innings_no": [1] * 3, "batter_id": ["b1"] * 3, "bowler_id": ["x1"] * 3,
        "runs_bat": [2, 0, 0], "runs_extras": [0, 4, 0], "is_wide": [False] * 3, "is_noball": [False] * 3,
        "is_legal": [True] * 3, "is_wicket": [False, False, True], "wicket_kind": [None, None, "bowled"],
        "player_out": [None, None, "B"]})
    r = player_ratings(deliveries, appearances, matches, "2026-01-01")
    assert r.loc["x1", "conceded"] == 2       # the 2 off the bat count; the 4 byes do not
    assert r.loc["x1", "balls"] == 3 and r.loc["x1", "wickets"] == 1


def innings_of(batters, runs, out):
    """One innings in which each batter faces one ball in turn, scoring runs and possibly getting out."""
    n = len(batters)
    return pd.DataFrame({
        "match_id": ["1"] * n, "innings_no": [1] * n, "batter_id": batters, "bowler_id": ["x"] * n,
        "runs_bat": runs, "runs_extras": [0] * n, "is_wide": [False] * n, "is_noball": [False] * n,
        "is_legal": [True] * n, "is_wicket": out, "wicket_kind": ["bowled" if o else None for o in out],
        "player_out": [b if o else None for b, o in zip(batters, out)]})


def test_batting_position_is_the_order_each_batter_first_faced():
    d = innings_of(["a", "b", "a", "c"], [1, 1, 1, 1], [False] * 4)
    assert batting_positions(d).to_dict() == {"a": 1, "b": 2, "c": 3}


def test_tail_enders_are_pulled_towards_the_lower_order_average():
    batters = [f"p{i}" for i in range(1, 10)]                      # nine batters, in order
    runs = [40, 40, 40, 40, 40, 40, 40, 10, 10]
    d = innings_of(batters, runs, [True] * 9)
    matches = pd.DataFrame({"match_id": ["1"], "date_start": ["2025-01-01"]})
    appearances = pd.DataFrame({"match_id": ["1"] * 9, "player_name": batters, "player_id": batters})
    r = player_ratings(d, appearances, matches, "2026-01-01", half_life=float("inf"))
    assert r.attrs["bat_mean_top"] == 40 and r.attrs["bat_mean_lower"] == 10
    assert r.loc["p9", "bat_rating"] == pytest.approx(10)          # a No. 9 stays at the lower-order level
    assert r.loc["p1", "bat_rating"] == pytest.approx(40)
