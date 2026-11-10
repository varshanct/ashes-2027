import pandas as pd

from cricket.find_ids import capped_candidates, register_candidates, surname
from cricket.ingest import ingest_folder


def test_surname():
    assert surname("Steve Smith") == "Smith"
    assert surname(" Joe Root ") == "Root"
    assert surname("Ben Duckett") == "Duckett"


def test_capped_candidates_filters_by_team(raw_dir):
    t = ingest_folder(raw_dir)
    found = capped_candidates("Bat", "England", t["appearances"], t["matches"])
    assert set(found["player_name"]) == {"E Bat", "E Bat2"}
    assert set(found["player_id"]) == {"eeee0001", "eeee0002"}


def test_capped_candidates_counts_tests_and_last_date(raw_dir):
    t = ingest_folder(raw_dir)
    found = capped_candidates("Bowl", "Australia", t["appearances"], t["matches"])
    assert found["player_id"].tolist() == ["aaaa0003"]
    assert found.loc[0, "tests"] == 4
    assert str(found.loc[0, "last_test"].date()) == "2023-08-10"


def test_search_ignores_capitals(raw_dir):
    t = ingest_folder(raw_dir)
    assert len(capped_candidates("bowl", "England", t["appearances"], t["matches"])) == 1


def test_no_capped_match_gives_empty_table(raw_dir):
    t = ingest_folder(raw_dir)
    found = capped_candidates("Peake", "Australia", t["appearances"], t["matches"])
    assert found.empty
    assert list(found.columns) == ["player_id", "player_name", "tests", "last_test"]


def test_register_candidates():
    people = pd.DataFrame({"identifier": ["1a2b3c4d", "9z8y7x6w", "00aa11bb"],
                           "name": ["O Peake", "J Peaker", "SPD Smith"]})
    found = register_candidates("Peake", people)
    assert found["identifier"].tolist() == ["1a2b3c4d", "9z8y7x6w"]