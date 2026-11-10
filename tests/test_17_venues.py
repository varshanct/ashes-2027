import pandas as pd
import pytest

from cricket.grounds import attach_ground, check_aliases, venue_candidates
from cricket.ingest import ingest_folder

VENUES = pd.DataFrame({"venue": ["Trent Bridge", "Lord's", "Edgbaston", "Utilita Bowl", "Kia Oval"]})


@pytest.fixture
def matches(raw_dir):
    return ingest_folder(raw_dir)["matches"]


def full_aliases():
    return pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham", "TB", "LD", "UB", "KO"],
                         "ground": ["Edgbaston", "Trent Bridge", "Lord's", "Utilita Bowl", "Kia Oval"]})


def test_venue_candidates_finds_and_counts(matches):
    found = venue_candidates(matches, "edgbaston")
    assert found["venue"].tolist() == ["Edgbaston, Birmingham"]
    assert found.loc[0, "city"] == "Birmingham"
    assert found.loc[0, "tests"] == 4
    assert found.loc[0, "first"] == 2023 and found.loc[0, "last"] == 2023


def test_venue_candidates_no_match(matches):
    assert venue_candidates(matches, "Headingley").empty


def test_attach_ground(matches):
    m = attach_ground(matches, full_aliases())
    assert (m["ground"] == "Edgbaston").all()


def test_attach_ground_leaves_other_venues_empty(matches):
    aliases = pd.DataFrame({"cricsheet_venue": ["Lord's, London"], "ground": ["Lord's"]})
    assert attach_ground(matches, aliases)["ground"].isna().all()


def test_aliases_report_missing_grounds_and_unknown_names(matches):
    aliases = pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham", "Lords"], "ground": ["Edgbaston", "Lord's"]})
    problems = check_aliases(aliases, VENUES, matches)
    assert "venue_aliases.csv: Trent Bridge has no Cricsheet venue name yet" in problems
    assert "venue_aliases.csv: 'Lords' does not appear as a venue in matches.csv" in problems


def test_aliases_report_unknown_ground(matches):
    aliases = pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Headingley"]})
    assert "venue_aliases.csv: 'Headingley' is not a ground in venues_2027.csv" in check_aliases(aliases, VENUES, matches)


def test_aliases_report_duplicates(matches):
    aliases = pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"] * 2, "ground": ["Edgbaston"] * 2})
    assert "venue_aliases.csv: 'Edgbaston, Birmingham' is listed more than once" in check_aliases(aliases, VENUES, matches)