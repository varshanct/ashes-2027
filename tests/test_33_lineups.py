import sys

import numpy as np
import pandas as pd
import pytest

from cricket import lineups
from cricket.lineups import add_xi_gaps, positions_in_match, rate_xi, results_by_gap, xi_ratings
from cricket.players import player_ratings


def ball(match_id, innings_no, batter, bowler, runs=0, out=False):
    return {"match_id": match_id, "innings_no": innings_no, "batter_id": batter, "bowler_id": bowler,
            "runs_bat": runs, "runs_extras": 0, "is_wide": False, "is_noball": False, "is_legal": True,
            "is_wicket": out, "wicket_kind": "bowled" if out else None, "player_out": batter if out else None}


def made_up_tests():
    """Two Tests between A and B: one in 2022 to learn from, one in 2023 to rate, with a debutant (a4)."""
    deliveries = pd.DataFrame([
        ball("m1", 1, "a1", "b3", 50), ball("m1", 1, "a1", "b3", 0, True),
        ball("m1", 1, "a2", "b3", 30), ball("m1", 1, "a2", "b3", 0, True),
        ball("m1", 1, "a3", "b3", 2), ball("m1", 1, "a3", "b3", 0, True),
        ball("m1", 2, "b1", "a3", 40), ball("m1", 2, "b1", "a3", 0, True),
        ball("m1", 2, "b2", "a3", 20), ball("m1", 2, "b2", "a3", 0, True),
        ball("m2", 1, "a4", "b3", 10), ball("m2", 1, "a1", "b3", 5),
        ball("m2", 2, "b1", "a3", 4),
    ])
    sheets = [("m1", "A", ["a1", "a2", "a3"]), ("m1", "B", ["b1", "b2", "b3"]),
              ("m2", "A", ["a1", "a2", "a3", "a4"]), ("m2", "B", ["b1", "b2", "b3"])]
    appearances = pd.DataFrame([{"match_id": m, "team": t, "player_name": p, "player_id": p}
                                for m, t, xi in sheets for p in xi])
    matches = pd.DataFrame({
        "match_id": ["m1", "m2"], "date_start": ["2022-06-01", "2023-06-01"], "venue": ["V", "V"],
        "city": ["C", "C"], "team1": ["A", "A"], "team2": ["B", "B"], "result_type": ["win", "draw"],
        "winner": ["A", None]})
    return deliveries, appearances, matches


def test_positions_in_match_are_the_order_each_batter_first_faced():
    deliveries, _, _ = made_up_tests()
    positions = positions_in_match(deliveries)
    assert positions.loc[("m1", "a1")] == 1 and positions.loc[("m1", "a3")] == 3
    assert positions.loc[("m2", "a4")] == 1 and positions.loc[("m2", "a1")] == 2


def test_rate_xi_uses_the_five_main_bowlers_and_averages_for_newcomers():
    ratings = pd.DataFrame({"bat_rating": [40.0, 30.0, 10.0, 12.0, 8.0, 9.0, 11.0],
                            "bowl_rating": [50.0, 45.0, 25.0, 27.0, 29.0, 31.0, 33.0],
                            "balls": [0, 100, 5000, 4000, 3000, 2000, 1000]},
                           index=["p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    ratings.attrs.update(bat_mean_top=35.0, bat_mean_lower=15.0, bowl_mean=30.0)
    ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "new"]
    xi = rate_xi(ids, ratings, pd.Series({"new": 2}))      # the newcomer batted at No. 2
    assert xi["bat_xi"] == pytest.approx((40 + 30 + 10 + 12 + 8 + 9 + 11 + 35) / 8)
    assert xi["bowl_xi"] == pytest.approx((25 + 27 + 29 + 31 + 33) / 5)   # p3 to p7 have bowled most
    assert xi["new_players"] == 1
    lower = rate_xi(ids, ratings, pd.Series(dtype=float))  # a newcomer who did not bat counts as lower order
    assert lower["bat_xi"] == pytest.approx((40 + 30 + 10 + 12 + 8 + 9 + 11 + 15) / 8)


def test_rate_xi_uses_who_bowled_in_this_match():
    ratings = pd.DataFrame({"bat_rating": [40.0, 30.0, 10.0, 12.0, 8.0, 9.0, 11.0],
                            "bowl_rating": [50.0, 45.0, 25.0, 27.0, 29.0, 31.0, 33.0],
                            "balls": [6000, 100, 5000, 4000, 3000, 2000, 1000]},
                           index=["p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    ratings.attrs.update(bat_mean_top=35.0, bat_mean_lower=15.0, bowl_mean=30.0)
    ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "new"]
    # p1 has bowled most in his career but not in this match; the debutant opened the bowling.
    balls = pd.Series({"new": 200, "p3": 180, "p4": 150, "p5": 120, "p6": 90, "p7": 12})
    xi = rate_xi(ids, ratings, pd.Series({"new": 9}), balls)
    assert xi["bowl_xi"] == pytest.approx((30 + 25 + 27 + 29 + 31) / 5)       # the debutant counts at 30


def test_xi_ratings_use_only_earlier_years():
    deliveries, appearances, matches = made_up_tests()
    xi = xi_ratings(deliveries, appearances, matches, first_year=2023, progress=lambda _: None)
    assert sorted(xi["team"]) == ["A", "B"] and set(xi["match_id"]) == {"m2"}
    a = xi[xi["team"] == "A"].iloc[0]
    assert a["players"] == 4 and a["new_players"] == 1               # a4 has no earlier Test
    # The rating must come from the 2022 Test alone, so it matches ratings as of 1 January 2023.
    before = player_ratings(deliveries, appearances, matches, "2023-01-01")
    debutant = before.attrs["bat_mean_top"]                            # a4 opened, so top-order average
    expected = (before.loc[["a1", "a2", "a3"], "bat_rating"].sum() + debutant) / 4
    assert a["bat_xi"] == pytest.approx(expected)


def test_gaps_are_from_team1_point_of_view():
    table = pd.DataFrame({"match_id": ["m"], "team1": ["A"], "team2": ["B"], "result": [2]})
    xi = pd.DataFrame({"match_id": ["m", "m"], "team": ["A", "B"], "bat_xi": [30.0, 25.0],
                       "bowl_xi": [28.0, 33.0]})
    out = add_xi_gaps(table, xi)
    assert out.loc[0, "bat_gap"] == 5.0          # A bats 5 runs per dismissal better
    assert out.loc[0, "bowl_gap"] == 5.0         # A concedes 5 fewer runs per wicket: also better


def test_results_by_gap_groups_from_weakest_to_strongest():
    table = pd.DataFrame({"bat_gap": np.arange(10.0), "result": [0, 0, 1, 0, 1, 2, 1, 2, 2, 2]})
    summary = results_by_gap(table, "bat_gap", groups=2)
    assert summary["tests"].tolist() == [5, 5]
    assert summary["team1_won"].tolist() == [0.0, 0.8]
    assert summary["team1_lost"].tolist() == [0.6, 0.0]


def test_command_saves_the_xi_ratings(tmp_path, monkeypatch, capsys):
    deliveries, appearances, matches = made_up_tests()
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    processed.mkdir()
    manual.mkdir()
    deliveries.to_csv(processed / "deliveries.csv", index=False)
    appearances.to_csv(processed / "appearances.csv", index=False)
    matches.to_csv(processed / "matches.csv", index=False)
    aliases = pd.DataFrame({"cricsheet_venue": ["V"], "ground": ["V"]})
    aliases.to_csv(manual / "venue_aliases.csv", index=False)
    monkeypatch.setattr(lineups, "FIRST_YEAR", 2023)
    monkeypatch.setattr(lineups, "results_by_gap", lambda table, column: table[[column]])
    monkeypatch.setattr(sys, "argv", ["lineups", "--processed", str(processed), "--manual", str(manual)])
    lineups.main()
    saved = pd.read_csv(processed / "xi_ratings.csv")
    assert len(saved) == 2
    assert "Tests with both XIs rated: 1" in capsys.readouterr().out
