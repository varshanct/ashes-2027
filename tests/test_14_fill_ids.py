import pandas as pd

from cricket.fill_ids import fill_file, fill_ids


def squads():
    return pd.DataFrame({"player_name": ["Joe Root", "Pat Cummins", "No Id Yet", "Twin", "Twin"],
                         "cricsheet_id": ["1a2b3c4d", "5e6f7a8b", None, "aaaa1111", "bbbb2222"]})


def table(names, ids):
    return pd.DataFrame({"cricsheet_id": ids, "player_name": names, "status": "fit"})


def test_placeholder_is_filled():
    filled, messages = fill_ids(table(["Joe Root"], ["PUT_ID"]), squads())
    assert filled.loc[0, "cricsheet_id"] == "1a2b3c4d"
    assert messages == ["filled   Joe Root: 1a2b3c4d"]


def test_existing_id_is_left_alone():
    filled, messages = fill_ids(table(["Joe Root"], ["99999999"]), squads())
    assert filled.loc[0, "cricsheet_id"] == "99999999"
    assert messages == []


def test_unknown_name_is_reported_not_guessed():
    filled, messages = fill_ids(table(["Jo Root"], ["PUT_ID"]), squads())
    assert filled.loc[0, "cricsheet_id"] == "PUT_ID"
    assert "NOT FILLED Jo Root" in messages[0]


def test_squad_player_without_id_is_reported():
    filled, messages = fill_ids(table(["No Id Yet"], ["PUT_ID"]), squads())
    assert filled.loc[0, "cricsheet_id"] == "PUT_ID"
    assert "no squad player" in messages[0]


def test_two_players_with_same_name_are_reported():
    filled, messages = fill_ids(table(["Twin"], ["PUT_ID"]), squads())
    assert filled.loc[0, "cricsheet_id"] == "PUT_ID"
    assert "more than one" in messages[0]


def test_fill_file_saves_the_result(tmp_path):
    path = tmp_path / "availability.csv"
    table(["Pat Cummins", "Joe Root"], ["PUT_ID", "PUT_ID"]).to_csv(path, index=False)
    fill_file(path, squads())
    saved = pd.read_csv(path, dtype={"cricsheet_id": str})
    assert saved["cricsheet_id"].tolist() == ["5e6f7a8b", "1a2b3c4d"]