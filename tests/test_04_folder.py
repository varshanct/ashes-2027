import pandas as pd
import pytest

from cricket.ingest import ingest_folder, save_tables


@pytest.fixture
def tables(raw_dir):
    return ingest_folder(raw_dir)


def test_only_mens_tests_are_kept(tables):
    assert set(tables["matches"]["match_id"]) == {"3001", "3002", "3003", "3004"}


def test_sorted_by_start_date(tables):
    assert tables["matches"]["date_start"].is_monotonic_increasing


def test_no_duplicate_matches(tables):
    assert tables["matches"]["match_id"].is_unique


def test_winner_is_always_a_participant(tables):
    decided = tables["matches"].dropna(subset=["winner"])
    ok = (decided["winner"] == decided["team1"]) | (decided["winner"] == decided["team2"])
    assert ok.all()


def test_washout(tables):
    washout = tables["matches"].set_index("match_id").loc["3004"]
    assert washout["result_type"] == "no_result"
    assert washout["overs_bowled"] == 0


def test_save_writes_four_csvs(tables, tmp_path):
    save_tables(tables, tmp_path)
    for name in ["matches", "innings", "deliveries", "appearances"]:
        assert (tmp_path / f"{name}.csv").exists()
    assert len(pd.read_csv(tmp_path / "matches.csv")) == 4