from cricket.checks import margin_mismatches
from cricket.ingest import ingest_folder


def test_correct_margins_pass(raw_dir):
    t = ingest_folder(raw_dir)
    assert margin_mismatches(t["matches"], t["innings"]).empty


def test_innings_win_margin_is_checked(raw_dir):
    t = ingest_folder(raw_dir)
    inns = t["innings"]
    inns.loc[(inns["match_id"] == "3001") & (inns["innings_no"] == 2), "runs"] = 15  # was 10
    bad = margin_mismatches(t["matches"], inns)
    assert bad["match_id"].tolist() == ["3001"]
    assert bad.iloc[0]["kind"] == "runs"
    assert bad.iloc[0]["from_totals"] == 7  # 30 - (15 + 8)


def test_wickets_margin_is_checked(raw_dir):
    t = ingest_folder(raw_dir)
    inns = t["innings"]
    inns.loc[(inns["match_id"] == "3002") & (inns["innings_no"] == 4), "wickets"] = 3  # was 1
    bad = margin_mismatches(t["matches"], inns)
    assert bad["match_id"].tolist() == ["3002"]
    assert bad.iloc[0]["kind"] == "wickets"
    assert bad.iloc[0]["from_totals"] == 7


def test_draws_are_not_checked(raw_dir):
    t = ingest_folder(raw_dir)
    inns = t["innings"]
    inns.loc[inns["match_id"] == "3003", "runs"] = 999
    assert margin_mismatches(t["matches"], inns).empty
