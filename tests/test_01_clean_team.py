from cricket.ingest import clean_team


def test_alias_is_replaced():
    assert clean_team("U.A.E.") == "United Arab Emirates"


def test_spaces_are_removed():
    assert clean_team(" England ") == "England"


def test_normal_name_is_unchanged():
    assert clean_team("Australia") == "Australia"