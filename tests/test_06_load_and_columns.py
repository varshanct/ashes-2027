from cricket.manual_data import check_columns, load_all


def save_all(tables, folder):
    for name, df in tables.items():
        df.to_csv(folder / name, index=False)


def test_load_all_reads_every_file(manual, tmp_path):
    save_all(manual, tmp_path)
    loaded = load_all(tmp_path)
    assert set(loaded) == set(manual)
    assert len(loaded["squads_2027.csv"]) == 24


def test_ids_with_leading_zeros_are_kept(manual, tmp_path):
    # Some Cricsheet IDs are all digits. Read as numbers, "00123456" would become 123456.
    ids = [f"00{n:06d}" for n in range(24)]
    manual["squads_2027.csv"]["cricsheet_id"] = ids
    save_all(manual, tmp_path)
    assert load_all(tmp_path)["squads_2027.csv"]["cricsheet_id"].tolist() == ids


def test_complete_files_have_all_columns(manual):
    assert check_columns(manual) == []


def test_missing_column_is_reported(manual):
    manual["squads_2027.csv"] = manual["squads_2027.csv"].drop(columns=["role"])
    assert check_columns(manual) == ["squads_2027.csv: missing columns ['role']"]