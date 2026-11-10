from cricket.manual_data import validate


def test_complete_files_pass_every_check(manual):
    assert validate(manual) == []


def test_missing_column_stops_other_checks(manual):
    manual["weather_climate.csv"] = manual["weather_climate.csv"].drop(columns=["p_rain_day"])
    assert validate(manual) == ["weather_climate.csv: missing columns ['p_rain_day']"]


def test_problems_from_different_files_are_all_reported(manual):
    manual["squads_2027.csv"].loc[0, "team"] = "India"
    manual["weather_climate.csv"].loc[0, "p_rain_day"] = 35
    found = validate(manual)
    assert any("unknown team 'India'" in p for p in found)
    assert any("write 0.35, not 35" in p for p in found)