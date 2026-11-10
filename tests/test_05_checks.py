from cricket.ingest import ingest_folder


def test_target_recorded(raw_dir):
    inns = ingest_folder(raw_dir)["innings"]
    fourth = inns[(inns["match_id"] == "3002") & (inns["innings_no"] == 4)].iloc[0]
    assert fourth["target_runs"] == 23  # 20 + 17 - 15 + 1
