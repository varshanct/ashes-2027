import sys

import pandas as pd
import pytest

from cricket import grounds
from cricket.ingest import ingest_folder, save_tables


@pytest.fixture
def project(raw_dir, tmp_path):
    processed, manual = tmp_path / "processed", tmp_path / "manual"
    save_tables(ingest_folder(raw_dir), processed)
    manual.mkdir()
    pd.DataFrame({"venue": ["Edgbaston"], "city": ["Birmingham"], "source_url": ["x"]}).to_csv(
        manual / "venues_2027.csv", index=False)
    pd.DataFrame({"cricsheet_venue": ["Edgbaston, Birmingham"], "ground": ["Edgbaston"]}).to_csv(
        manual / "venue_aliases.csv", index=False)
    return tmp_path


def run(monkeypatch, project, *extra):
    argv = ["grounds", "--processed", str(project / "processed"), "--manual", str(project / "manual"),
            "--figures", str(project / "figures"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    grounds.main()


def test_main_saves_table_and_three_charts(monkeypatch, project):
    run(monkeypatch, project)
    names = sorted(p.name for p in (project / "figures").iterdir())
    assert names == ["ashes_at_grounds.png", "england_at_grounds.png",
                     "ground_profiles.csv", "ground_profiles.png"]


def test_main_since_uses_its_own_file_names(monkeypatch, project):
    run(monkeypatch, project, "--since", "2023-07-01")
    assert (project / "figures" / "ground_profiles_since_2023.png").exists()


def test_main_stops_on_bad_aliases(monkeypatch, project):
    pd.DataFrame({"cricsheet_venue": ["Nowhere"], "ground": ["Edgbaston"]}).to_csv(
        project / "manual" / "venue_aliases.csv", index=False)
    with pytest.raises(SystemExit):
        run(monkeypatch, project)
    assert not (project / "figures").exists()


def test_find_lists_candidates(monkeypatch, project, capsys):
    run(monkeypatch, project, "--find")
    assert "Edgbaston, Birmingham" in capsys.readouterr().out
