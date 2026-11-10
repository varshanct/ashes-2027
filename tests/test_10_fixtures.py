from cricket.manual_data import check_fixtures, months_needed


def problems(manual):
    return check_fixtures(manual["fixtures_2027.csv"], manual["venues_2027.csv"])


def test_valid_schedule_passes(manual):
    assert problems(manual) == []


def test_test_lasting_six_days(manual):
    manual["fixtures_2027.csv"].loc[0, "date_end"] = "2027-06-23"
    assert problems(manual) == ["fixtures_2027.csv: Test 1 lasts 6 days; a Test is scheduled for 5"]


def test_tests_out_of_order(manual):
    fx = manual["fixtures_2027.csv"]
    fx.loc[0, ["date_start", "date_end"]] = ["2027-08-10", "2027-08-14"]
    assert "fixtures_2027.csv: Tests are not in date order" in problems(manual)


def test_unknown_venue(manual):
    manual["fixtures_2027.csv"].loc[0, "venue"] = "Headingley"
    assert "fixtures_2027.csv: venue 'Headingley' is not in venues_2027.csv" in problems(manual)


def test_wrong_home_team(manual):
    manual["fixtures_2027.csv"].loc[0, ["home_team", "away_team"]] = ["Australia", "England"]
    assert any("home_team England" in p for p in problems(manual))


def test_months_needed_spans_month_ends(manual):
    needed = months_needed(manual["fixtures_2027.csv"])
    assert ("Lord's", 6) in needed and ("Lord's", 7) in needed
    assert ("Kia Oval", 7) in needed and ("Kia Oval", 8) in needed
    assert len(needed) == 7