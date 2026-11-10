import pandas as pd
import pytest

from cricket.grounds import ground_profiles, team_record_at_grounds
from cricket.ingest import ingest_folder

ORDER = ["Trent Bridge", "Lord's", "Edgbaston", "Utilita Bowl", "Kia Oval"]
ALIASES = pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]})


@pytest.fixture
def tables(raw_dir):
    return ingest_folder(raw_dir)


@pytest.fixture
def edgbaston(tables):
    p = ground_profiles(tables["matches"], tables["innings"], tables["deliveries"], ALIASES, ORDER)
    return p.set_index("ground").loc["Edgbaston"]


def test_one_row_per_ground_in_order(tables):
    p = ground_profiles(tables["matches"], tables["innings"], tables["deliveries"], ALIASES, ORDER)
    assert p["ground"].tolist() == ORDER


def test_washout_is_not_counted(edgbaston):
    assert edgbaston["tests"] == 3  # 3001, 3002, 3003; 3004 had no result


def test_draw_rate(edgbaston):
    assert edgbaston["draw_rate"] == pytest.approx(1 / 3)


def test_bat_first_and_toss(edgbaston):
    # England batted first and won the toss in all three; they won only 3001.
    assert edgbaston["bat_first_win_rate"] == pytest.approx(1 / 3)
    assert edgbaston["toss_winner_win_rate"] == pytest.approx(1 / 3)


def test_first_innings_and_overs(edgbaston):
    assert edgbaston["avg_first_innings_runs"] == pytest.approx((30 + 20 + 18) / 3)
    assert edgbaston["avg_overs_bowled"] == pytest.approx((2.0 + 2.5 + 1.0) / 3)


def test_how_batters_were_out(edgbaston):
    # Bowled and lbw in 3001, caught in 3002; the retired-hurt batter is not a wicket.
    assert edgbaston["wickets"] == 3
    assert edgbaston["share_bowled"] == pytest.approx(1 / 3)
    assert edgbaston["share_lbw"] == pytest.approx(1 / 3)
    assert edgbaston["share_caught"] == pytest.approx(1 / 3)


def test_ground_with_no_tests(tables):
    p = ground_profiles(tables["matches"], tables["innings"], tables["deliveries"], ALIASES, ORDER)
    lords = p.set_index("ground").loc["Lord's"]
    assert lords["tests"] == 0
    assert pd.isna(lords["draw_rate"])


def test_since_filter(tables):
    p = ground_profiles(tables["matches"], tables["innings"], tables["deliveries"], ALIASES, ORDER,
                        since="2023-07-01")
    assert p.set_index("ground").loc["Edgbaston", "tests"] == 2  # 3002 and 3003


def test_team_record(tables):
    r = team_record_at_grounds(tables["matches"], ALIASES, "England", ORDER).set_index("ground")
    assert r.loc["Edgbaston"].tolist() == [3, 1, 1, 1]  # tests, wins, draws, losses
    assert r.loc["Lord's", "tests"] == 0


def test_team_record_against_one_opponent(tables):
    r = team_record_at_grounds(tables["matches"], ALIASES, "Australia", ORDER, opponent="India")
    assert r["tests"].sum() == 0