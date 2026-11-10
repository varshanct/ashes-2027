import sys

import pandas as pd
import pytest

from cricket import players
from cricket.ingest import ingest_folder, save_tables
from cricket.players import FC_BAT_SHARE, PRIOR_OUTS, PRIOR_WICKETS, check_first_class, with_first_class

SQUADS = pd.DataFrame({"team": ["England", "England", "Australia"],
                       "player_name": ["New Bat", "Old Hand", "New Bowler"],
                       "cricsheet_id": ["n1", "o1", "b1"], "role": ["batter", "batter", "bowler"],
                       "squad_tier": ["core", "core", "fringe"]})


def first_class(rows):
    """A first-class file from (id, name, matches, bat_avg, wickets, bowl_avg) rows."""
    table = pd.DataFrame(rows, columns=["cricsheet_id", "player_name", "fc_matches", "fc_bat_avg",
                                        "fc_wickets", "fc_bowl_avg"])
    return table.assign(stats_as_of="2026-10-01", source_url="https://example.com", last_updated="2026-11-10")


GOOD = first_class([("n1", "New Bat", 60, 40.0, None, None), ("b1", "New Bowler", 50, 15.0, 150, 25.0)])


def test_a_correct_file_has_no_problems():
    assert check_first_class(GOOD, SQUADS) == []


def test_problems_are_reported():
    bad = first_class([("x9", "Stranger", 60, 40.0, None, None),          # not in the squads
                       ("n1", "New Bat", 60, -3.0, None, None),           # impossible batting average
                       ("b1", "New Bowler", 50, 15.0, 0, 25.0),           # bowling average with no wickets
                       ("b1", "New Bowler", 50, 15.0, 150, 25.0)])        # listed twice
    problems = check_first_class(bad, SQUADS)
    assert len(problems) == 4
    assert any("Stranger" in p and "not in squads" in p for p in problems)
    assert any("positive fc_bat_avg" in p for p in problems)
    assert any("no wickets" in p for p in problems)
    assert any("more than once" in p for p in problems)


def test_missing_columns_are_reported():
    assert "missing columns" in check_first_class(GOOD.drop(columns="stats_as_of"), SQUADS)[0]


def test_only_players_with_few_tests_are_re_rated():
    table = pd.DataFrame({"team": SQUADS["team"], "player_name": SQUADS["player_name"],
                          "tests": [0, 50, 3], "bat_rating": [30.0, 45.0, 14.0],
                          "bowl_rating": [30.0, 30.0, 32.0]})
    ratings = pd.DataFrame({"w_runs": [0.0, 900.0, 20.0], "w_outs": [0.0, 20.0, 2.0],
                            "w_conceded": [0.0, 0.0, 100.0], "w_wickets": [0.0, 0.0, 2.0]},
                           index=pd.Index(["n1", "o1", "b1"], name="player_id"))
    fc = pd.concat([GOOD, first_class([("o1", "Old Hand", 200, 55.0, None, None)])], ignore_index=True)
    out = with_first_class(table, ratings, SQUADS, fc)
    # New Bat has no Test record: he starts from 85% of his first-class average of 40.
    assert out.loc[0, "bat_rating"] == pytest.approx(FC_BAT_SHARE * 40)
    assert out.loc[0, "bowl_rating"] == 30.0                     # no first-class bowling: unchanged
    # Old Hand has 50 Tests, so his first-class record is ignored.
    assert out.loc[1, "bat_rating"] == 45.0 and not out.loc[1, "first_class_used"]
    # New Bowler: his 2 Test wickets for 100 are added to 10 wickets at his first-class average of 25.
    assert out.loc[2, "bowl_rating"] == pytest.approx((100 + PRIOR_WICKETS * 25) / (2 + PRIOR_WICKETS))
    assert out.loc[2, "bat_rating"] == pytest.approx((20 + PRIOR_OUTS * FC_BAT_SHARE * 15) / (2 + PRIOR_OUTS))
    assert out.loc[2, "bowl_rating_tests_only"] == 32.0         # the Test-only rating is kept alongside


def test_command_stops_on_a_bad_first_class_file(raw_dir, tmp_path, monkeypatch, capsys):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    manual.mkdir()
    SQUADS.to_csv(manual / "squads_2027.csv", index=False)
    first_class([("x9", "Stranger", 60, 40.0, None, None)]).to_csv(manual / "first_class.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["players", "--processed", str(processed), "--manual", str(manual),
                                      "--figures", str(tmp_path / "figures")])
    with pytest.raises(SystemExit):
        players.main()
    assert "Stranger" in capsys.readouterr().out