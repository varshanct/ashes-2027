from cricket.manual_data import check_xi


def england(sq):
    return sq["team"] == "England"


def test_balanced_xis_pass(manual):
    assert check_xi(manual["squads_2027.csv"]) == []


def test_missing_batting_position(manual):
    sq = manual["squads_2027.csv"]
    sq.loc[england(sq) & (sq["expected_batting_position"] == 7), "expected_batting_position"] = None
    assert check_xi(sq) == [
        "squads_2027.csv: England expected XI is missing batting positions [7]",
        "squads_2027.csv: England expected XI has no wicketkeeper",
    ]


def test_two_players_in_one_position(manual):
    sq = manual["squads_2027.csv"]
    sq.loc[england(sq) & (sq["player_name"] == "England Reserve Seamer"), "expected_batting_position"] = 3
    assert check_xi(sq) == ["squads_2027.csv: England has more than one player at batting positions [3]"]


def test_no_wicketkeeper(manual):
    sq = manual["squads_2027.csv"]
    sq.loc[england(sq) & (sq["role"] == "keeper"), "role"] = "batter"
    assert check_xi(sq) == ["squads_2027.csv: England expected XI has no wicketkeeper"]


def test_too_few_bowling_options(manual):
    sq = manual["squads_2027.csv"]
    sq.loc[england(sq) & sq["expected_batting_position"].isin([9, 10]), "role"] = "batter"
    problems = check_xi(sq)
    assert len(problems) == 1
    assert "England expected XI has 3 bowling options" in problems[0]


def test_fringe_players_need_no_position(manual):
    sq = manual["squads_2027.csv"]
    assert sq.loc[sq["squad_tier"] == "fringe", "expected_batting_position"].isna().all()
    assert check_xi(sq) == []