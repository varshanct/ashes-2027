import pytest

from cricket.manual_data import check_availability


def problems(manual):
    return check_availability(manual["availability.csv"], manual["squads_2027.csv"])


def test_valid_availability_passes(manual):
    assert problems(manual) == []


@pytest.mark.parametrize("value", [-0.1, 1.5, None, "85%"])
def test_probability_must_be_0_to_1(manual, value):
    av = manual["availability.csv"]
    av["p_available_t3"] = av["p_available_t3"].astype(object)
    av.loc[0, "p_available_t3"] = value
    assert len([p for p in problems(manual) if "p_available_t3" in p]) == 1


def test_retired_player_must_be_zero(manual):
    manual["availability.csv"].loc[0, "status"] = "retired_test"
    assert any("is retired_test" in p for p in problems(manual))


def test_player_must_be_in_a_squad(manual):
    manual["availability.csv"].loc[0, "cricsheet_id"] = "zzzz9999"
    assert any("is not in squads_2027.csv" in p for p in problems(manual))


def test_unknown_status(manual):
    manual["availability.csv"].loc[0, "status"] = "crocked"
    assert any("unknown values ['crocked']" in p for p in problems(manual))


def test_bad_return_date(manual):
    manual["availability.csv"].loc[0, "expected_return"] = "01/06/2027"
    assert any("'01/06/2027' in expected_return" in p for p in problems(manual))