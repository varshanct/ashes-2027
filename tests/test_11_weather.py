from cricket.manual_data import check_weather


def problems(manual):
    return check_weather(manual["weather_climate.csv"], manual["fixtures_2027.csv"])


def test_valid_weather_passes(manual):
    assert problems(manual) == []


def test_rain_written_as_percentage(manual):
    manual["weather_climate.csv"].loc[0, "p_rain_day"] = 35
    assert problems(manual) == [
        "weather_climate.csv: p_rain_day 35.0 for Trent Bridge in month 6 must be 0-1 (write 0.35, not 35)"]


def test_blank_value_gives_a_to_do_item(manual):
    manual["weather_climate.csv"].loc[1, "p_rain_day"] = None
    assert problems(manual) == ["weather_climate.csv: fill in p_rain_day for Lord's in month 6"]


def test_missing_month_row(manual):
    manual["weather_climate.csv"] = manual["weather_climate.csv"].iloc[1:]
    assert problems(manual) == ["weather_climate.csv: no row for Trent Bridge in month 6"]
