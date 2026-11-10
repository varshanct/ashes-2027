import pytest

from cricket.ingest import parse_match


@pytest.fixture
def innings_win(raw_dir):
    return parse_match(raw_dir / "3001.json")


@pytest.fixture
def chase(raw_dir):
    return parse_match(raw_dir / "3002.json")


@pytest.fixture
def draw(raw_dir):
    return parse_match(raw_dir / "3003.json")


def test_dates(innings_win):
    m = innings_win["matches"][0]
    assert m["date_start"] == "2023-06-16"
    assert m["date_end"] == "2023-06-18"
    assert m["days_played"] == 3


def test_result_is_read(innings_win):
    m = innings_win["matches"][0]
    assert m["winner"] == "England"
    assert m["margin_innings"] == 1


def test_declaration(innings_win):
    assert innings_win["innings"][0]["declared"] is True
    assert innings_win["innings"][1]["declared"] is False


def test_follow_on_detected(innings_win):
    assert innings_win["innings"][2]["follow_on"] is True
    assert innings_win["innings"][2]["batting_team"] == "Australia"


def test_normal_third_innings_is_not_follow_on(chase):
    assert chase["innings"][2]["follow_on"] is False


def test_forfeited_innings(draw):
    third = draw["innings"][2]
    assert third["forfeited"] is True
    assert third["runs"] == 0


def test_penalty_runs_added(chase):
    third = chase["innings"][2]
    assert third["penalty_runs"] == 5
    assert third["runs"] == 17  # 12 off the balls + 5 penalty


def test_retired_hurt_is_not_a_wicket(chase):
    assert chase["innings"][2]["wickets"] == 0


def test_real_wickets_counted(chase):
    assert chase["innings"][3]["wickets"] == 1


def test_legal_balls_exclude_wides_and_noballs(chase):
    assert chase["innings"][2]["legal_balls"] == 3


def test_ball_numbers_skip_illegal_deliveries(chase):
    numbers = [d["ball_in_over"] for d in chase["deliveries"] if d["innings_no"] == 3]
    assert numbers == [1, 1, 2, 2, 3]


def test_overs_bowled(innings_win):
    # 5 + 4 + 3 legal balls = 12 balls = 2.0 overs
    assert innings_win["matches"][0]["overs_bowled"] == 2.0


def test_runs_add_up(chase):
    for inn in chase["innings"]:
        from_balls = sum(d["runs_total"] for d in chase["deliveries"] if d["innings_no"] == inn["innings_no"])
        assert from_balls + inn["penalty_runs"] == inn["runs"]


def test_player_ids_filled(chase):
    assert all(d["batter_id"] for d in chase["deliveries"])
    assert all(a["player_id"] for a in chase["appearances"])