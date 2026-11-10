from cricket.checks import fourth_innings_target_mismatches
from cricket.ingest import ingest_folder


def test_target_recorded(raw_dir):
    inns = ingest_folder(raw_dir)["innings"]
    fourth = inns[(inns["match_id"] == "3002") & (inns["innings_no"] == 4)].iloc[0]
    assert fourth["target_runs"] == 23  # 20 + 17 - 15 + 1


def test_consistent_targets_pass(raw_dir):
    inns = ingest_folder(raw_dir)["innings"]
    assert fourth_innings_target_mismatches(inns).empty


def test_wrong_total_is_caught(raw_dir):
    inns = ingest_folder(raw_dir)["innings"]
    third = (inns["match_id"] == "3002") & (inns["innings_no"] == 3)
    inns.loc[third, "runs"] = 12  # pretend we forgot the 5 penalty runs
    bad = fourth_innings_target_mismatches(inns)
    assert bad["match_id"].tolist() == ["3002"]
    assert bad.iloc[0]["expected_target"] == 18