import sys

import numpy as np
import pandas as pd
import pytest

from cricket import squad_forecast
from cricket.ingest import ingest_folder, save_tables
from cricket.manual_data import AVAIL_COLS
from cricket.players import FC_COLUMNS
from cricket.predict import forecast_tests
from cricket.squad_forecast import (availability_draws, need_for, pick_xi, rate_xi, role_averages,
                                    simulate_xis, xi_edge)
from test_22_predict import FIXTURES, fake_post
from test_26_forecast_command import GROUNDS, fake_sampler

CORE = [("b1", "batter", "none"), ("b2", "batter", "none"), ("b3", "batter", "none"),
        ("b4", "batter", "none"), ("b5", "batter", "none"), ("ar", "allrounder", "pace"),
        ("wk", "keeper", "none"), ("sp", "bowler", "spin"), ("p1", "bowler", "pace"),
        ("p2", "bowler", "pace"), ("p3", "bowler", "pace")]
FRINGE = [("rb", "batter", "none"), ("rp1", "bowler", "pace"), ("rp2", "bowler", "pace"),
          ("rs", "bowler", "spin")]


def squad(team="England"):
    """A balanced XI batting 1 to 11, then four reserves."""
    rows = [{"team": team, "player_name": pid, "cricsheet_id": pid, "role": role, "bowling_type": bowl,
             "squad_tier": "core", "expected_batting_position": pos}
            for pos, (pid, role, bowl) in enumerate(CORE, start=1)]
    rows += [{"team": team, "player_name": pid, "cricsheet_id": pid, "role": role, "bowling_type": bowl,
              "squad_tier": "fringe", "expected_batting_position": None} for pid, role, bowl in FRINGE]
    return pd.DataFrame(rows)


COVER = pd.DataFrame({"team": "England", "need": ["batter", "pace", "pace", "spin", "keeper"],
                      "cover_order": [1, 1, 2, 1, 1], "cricsheet_id": ["rb", "rp1", "rp2", "rs", "b3"]})


def test_need_for_each_kind_of_player():
    players = squad().set_index("cricsheet_id")
    assert need_for(players.loc["wk"]) == "keeper"
    assert need_for(players.loc["b1"]) == "batter"
    assert need_for(players.loc["ar"]) == "batter"        # an allrounder at No. 6 is replaced by a batter
    assert need_for(players.loc["sp"]) == "spin"
    assert need_for(players.loc["p1"]) == "pace"
    low = players.loc["ar"].copy()
    low["expected_batting_position"] = 8
    assert need_for(low) == "pace"                         # lower down, by a bowler of his type


def test_pick_xi_follows_the_cover_plan():
    team = squad()
    assert pick_xi(team, COVER, frozenset()) == [pid for pid, _, _ in CORE]
    assert "rp1" in pick_xi(team, COVER, frozenset({"p1"}))
    assert "rp2" in pick_xi(team, COVER, frozenset({"p1", "rp1"}))           # first cover unavailable
    keeper_out = pick_xi(team, COVER, frozenset({"wk"}))
    assert "rb" in keeper_out and "b3" in keeper_out                         # b3 keeps, a batter comes in
    for missing in [{"p1"}, {"wk"}, {"p1", "p2", "p3"}]:
        xi = pick_xi(team, COVER, frozenset(missing))
        assert len(xi) == 11 and len(set(xi)) == 11 and not set(xi) & missing


def test_pick_xi_falls_back_when_the_cover_plan_runs_out():
    team = squad()
    xi = pick_xi(team, COVER, frozenset({"p1", "p2", "p3", "rp2"}))
    assert {"rp1", "rb", "rs"} <= set(xi)                    # pace cover, then anyone available
    nobody = frozenset({"p1", "rb", "rp1", "rp2", "rs"})
    assert "replacement pace" in pick_xi(team, COVER, nobody)


def test_rate_xi_picks_the_attack_by_role():
    # x1 is a part-timer with the most Test balls; x6 is a specialist bowler with none.
    rated = pd.DataFrame({"bat_rating": [40.0, 30.0, 10.0, 12.0, 8.0, 9.0],
                          "bowl_rating": [44.0, 25.0, 27.0, 29.0, 31.0, 21.0],
                          "balls": [6000, 5000, 4000, 3000, 2000, 0],
                          "attack": [0, 2, 2, 2, 1, 2]},
                         index=["x1", "x2", "x3", "x4", "x5", "x6"])
    averages = role_averages({"bat_mean_top": 35.0, "bat_mean_lower": 15.0, "bowl_mean": 30.0})
    bat, bowl = rate_xi(["x1", "x2", "x3", "x4", "x5", "x6", "replacement batter"], rated, averages)
    assert bat == pytest.approx((40 + 30 + 10 + 12 + 8 + 9 + 35) / 7)
    assert bowl == pytest.approx((25 + 27 + 29 + 21 + 31) / 5)              # x6 counts, x1 does not
    _, with_replacement = rate_xi(["x1", "x2", "x3", "x4", "replacement pace"], rated, averages)
    assert with_replacement == pytest.approx((25 + 27 + 29 + 30 + 44) / 5)  # five needed, so x1 makes it


def test_availability_matches_the_chances_and_carries_over_between_tests():
    team = squad()
    chances = [0.5, 0.6, 0.7, 0.8, 0.9]
    availability = pd.DataFrame([{"cricsheet_id": "p1", **dict(zip(AVAIL_COLS, chances))}])
    available = availability_draws(team, availability, 20000, np.random.default_rng(1))
    assert available.shape == (20000, 5, len(team))
    p1 = list(team["cricsheet_id"]).index("p1")
    np.testing.assert_allclose(available[:, :, p1].mean(axis=0), chances, atol=0.015)
    assert available[:, :, list(team["cricsheet_id"]).index("b1")].all()     # not listed: always available
    missed_first = ~available[:, 0, p1]
    assert (~available[missed_first, 1, p1]).mean() > (~available[:, 1, p1]).mean() + 0.05
    without = availability_draws(team, availability, 100, np.random.default_rng(1), without=["b1"])
    assert not without[:, :, list(team["cricsheet_id"]).index("b1")].any()


def test_simulate_xis_with_everyone_fit_picks_the_expected_xi_every_time():
    australia = squad("Australia").assign(cricsheet_id=lambda d: "a" + d["cricsheet_id"])
    both = pd.concat([squad("England"), australia], ignore_index=True)
    cover = pd.concat([COVER, COVER.assign(team="Australia", cricsheet_id="a" + COVER["cricsheet_id"])])
    rated = pd.DataFrame({"bat_rating": 30.0, "bowl_rating": 30.0, "balls": 100, "attack": 0},
                         index=both["cricsheet_id"])
    rated.loc["b1", "bat_rating"] = 52.0                    # England's opener is 22 runs better
    averages = role_averages({"bat_mean_top": 35.0, "bat_mean_lower": 15.0, "bowl_mean": 30.0})
    available = np.ones((4, 5, len(both)), dtype=bool)
    xis, played = simulate_xis(both, cover, rated, averages, available)
    assert xis["England"]["bat"].shape == (4, 5)
    np.testing.assert_allclose(xis["England"]["bat"] - xis["Australia"]["bat"], 2.0)
    assert played["b1"] == 5 and played["ab1"] == 5 and "rb" not in played


def test_xi_edge_adds_to_the_home_side():
    post = {"xi_bat_effect": np.array([0.3, 0.3]), "xi_bowl_effect": np.array([0.2, 0.2])}
    xis = {"England": {"bat": np.full((2, 5), 40.0), "bowl": np.full((2, 5), 28.0)},
           "Australia": {"bat": np.full((2, 5), 30.0), "bowl": np.full((2, 5), 33.0)}}
    np.testing.assert_allclose(xi_edge(post, xis), 0.3 * 1.0 + 0.2 * 0.5)


def test_forecast_tests_with_an_edge():
    post = fake_post()
    plain = forecast_tests(post, FIXTURES, "England", "Australia", 2027)
    zero = forecast_tests(post, FIXTURES, "England", "Australia", 2027, edge=np.zeros((400, 3)))
    np.testing.assert_allclose(plain, zero)
    boost = forecast_tests(post, FIXTURES, "England", "Australia", 2027, edge=np.full((400, 3), 0.5))
    assert (boost[:, :, 0].mean(axis=0) > plain[:, :, 0].mean(axis=0)).all()


def xi_sampler(data, draws, tune, chains, seed):
    samples, divergences = fake_sampler(data, draws, tune, chains, seed)
    samples["xi_bat_effect"] = np.full((chains, draws), 0.3)
    samples["xi_bowl_effect"] = np.full((chains, draws), 0.2)
    return samples, divergences


@pytest.fixture
def project(raw_dir, tmp_path):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    pd.DataFrame(columns=["match_id", "team", "bat_xi", "bowl_xi"]).to_csv(processed / "xi_ratings.csv",
                                                                           index=False)
    manual.mkdir()
    pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]}).to_csv(
        manual / "venue_aliases.csv", index=False)
    pd.DataFrame({"test_no": [1, 2, 3, 4, 5], "venue": GROUNDS}).to_csv(manual / "fixtures_2027.csv",
                                                                         index=False)
    england, australia = squad("England"), squad("Australia")
    australia["cricsheet_id"] = "a" + australia["cricsheet_id"]
    australia["player_name"] = "Aus " + australia["player_name"]
    pd.concat([england, australia]).to_csv(manual / "squads_2027.csv", index=False)
    pd.concat([COVER, COVER.assign(team="Australia", cricsheet_id="a" + COVER["cricsheet_id"])]).to_csv(
        manual / "cover_plan.csv", index=False)
    pd.DataFrame([{"cricsheet_id": "p1", "player_name": "p1", **dict.fromkeys(AVAIL_COLS, 0.6)}]).to_csv(
        manual / "availability.csv", index=False)
    pd.DataFrame(columns=FC_COLUMNS).to_csv(manual / "first_class.csv", index=False)
    return tmp_path


def run(monkeypatch, project, *extra):
    monkeypatch.setattr(squad_forecast, "sample_posterior", xi_sampler)
    argv = ["squad_forecast", "--draws", "30", "--chains", "2", "--processed", str(project / "processed"),
            "--manual", str(project / "manual"), "--figures", str(project / "figures"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    squad_forecast.main()


def test_command_saves_every_file(monkeypatch, project, capsys):
    run(monkeypatch, project, "--without", "Aus p1,Aus p2")
    assert sorted(p.name for p in (project / "figures").iterdir()) == [
        "squad_forecast.png", "squad_forecast_players.csv", "squad_forecast_series.csv",
        "squad_forecast_tests.csv"]
    series = pd.read_csv(project / "figures" / "squad_forecast_series.csv")
    assert len(series) == 4 and series["forecast"].iloc[-1].endswith("without Aus p1 and Aus p2")
    total = series[["England win series", "Drawn series", "Australia win series"]].sum(axis=1)
    np.testing.assert_allclose(total, 1.0, atol=0.002)          # saved to 3 decimal places
    assert "Average Tests played" in capsys.readouterr().out


def test_command_stops_on_an_unknown_name(monkeypatch, project):
    with pytest.raises(SystemExit, match="Not in squads_2027.csv"):
        run(monkeypatch, project, "--without", "Nobody")


def test_command_asks_for_the_xi_ratings_first(monkeypatch, project):
    (project / "processed" / "xi_ratings.csv").unlink()
    with pytest.raises(SystemExit, match="cricket.lineups"):
        run(monkeypatch, project)