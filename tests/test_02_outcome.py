from cricket.ingest import parse_outcome


def test_innings_victory():
    r = parse_outcome({"winner": "England", "by": {"innings": 1, "runs": 12}})
    assert r["winner"] == "England"
    assert r["margin_innings"] == 1
    assert r["margin_runs"] == 12
    assert r["result_type"] == "win"


def test_win_by_wickets():
    r = parse_outcome({"winner": "Australia", "by": {"wickets": 9}})
    assert r["winner"] == "Australia"
    assert r["margin_wkts"] == 9
    assert r["margin_innings"] is None


def test_draw():
    r = parse_outcome({"result": "draw"})
    assert r["result_type"] == "draw"
    assert r["winner"] is None


def test_no_result():
    assert parse_outcome({"result": "no result"})["result_type"] == "no_result"