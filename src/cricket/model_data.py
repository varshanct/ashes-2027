"""Turn the match table into the numbers the Bayesian model needs."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXCLUDED_TEAMS = {"ICC World XI"}  # a one-off composite side, not a national team
HOME_SHARE = 0.8      # a team counts as the home side of a place if it plays in at least 80% of Tests there
TREND_BASE_YEAR = 2019


def home_team_by_place(matches):
    """For each place (the city, or the venue if no city is recorded), the team that plays there at home, if any."""
    place = matches["city"].fillna(matches["venue"])
    homes = {}
    for name, group in matches.groupby(place):
        counts = pd.concat([group["team1"], group["team2"]]).value_counts()
        top, n_top = counts.index[0], counts.iloc[0]
        second = counts.iloc[1] if len(counts) > 1 else 0
        if n_top >= HOME_SHARE * len(group) and n_top > second:
            homes[name] = top
    return homes


def runs_per_wicket(innings):
    """Runs scored per wicket lost across all innings: the exchange rate between runs and wickets."""
    return innings["runs"].sum() / innings["wickets"].sum()


def margin_in_wickets(matches, rpw):
    """Every win's margin in wickets: the wickets in hand, or the runs margin divided by runs per wicket.

    An innings win adds 10 wickets to its runs margin: the winner never needed its second innings.
    """
    innings_bonus = np.where(matches["margin_innings"].fillna(0) > 0, 10, 0)
    return (matches["margin_runs"] / rpw + innings_bonus).fillna(matches["margin_wkts"])


def build_model_table(matches, aliases, rpw=None):
    """One row per finished Test: who played, where, when, who was at home, and the result.

    With runs per wicket given, also the size of each win: log(1 + margin in wickets), empty for draws.
    """
    m = matches[matches["result_type"].isin(["win", "draw"])]
    m = m[~m["team1"].isin(EXCLUDED_TEAMS) & ~m["team2"].isin(EXCLUDED_TEAMS)].copy()
    home = m["city"].fillna(m["venue"]).map(home_team_by_place(m))
    m["home"] = np.where(home == m["team1"], 1, np.where(home == m["team2"], -1, 0))
    m["year"] = pd.to_datetime(m["date_start"]).dt.year
    lookup = dict(zip(aliases["cricsheet_venue"], aliases["ground"]))
    m["venue_key"] = m["venue"].map(lookup).fillna(m["venue"])
    # Result from team1's point of view: 0 = team1 lost, 1 = draw, 2 = team1 won.
    m["result"] = np.where(m["result_type"] == "draw", 1, np.where(m["winner"] == m["team1"], 2, 0))
    columns = ["match_id", "date_start", "year", "team1", "team2", "home", "venue_key", "result"]
    if rpw is not None:
        m["margin"] = np.log1p(margin_in_wickets(m, rpw))
        columns.append("margin")
    return m.sort_values(["date_start", "match_id"])[columns].reset_index(drop=True)


@dataclass
class ModelData:
    """Everything the model needs, as plain arrays, plus the labels that go with them."""
    teams: list
    years: list
    venues: list
    team_a: np.ndarray
    team_b: np.ndarray
    year_idx: np.ndarray
    venue_idx: np.ndarray
    home: np.ndarray
    decade: np.ndarray
    result: np.ndarray
    # Wins with a recorded margin: which Tests, the size of the win, and +1 if team1 won or -1 if team2 won.
    margin_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    margin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    margin_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))


def to_model_data(table, extra_venues=()):
    """Number every team, year and venue, and turn each column into an array of those numbers."""
    teams = sorted(set(table["team1"]) | set(table["team2"]))
    years = list(range(int(table["year"].min()), int(table["year"].max()) + 1))
    venues = sorted(set(table["venue_key"]) | set(extra_venues))
    team_number = {t: i for i, t in enumerate(teams)}
    venue_number = {v: i for i, v in enumerate(venues)}
    margins = {}
    if "margin" in table.columns:
        won = table["margin"].notna().to_numpy() & (table["result"] != 1).to_numpy()
        margins = {"margin_idx": np.flatnonzero(won),
                   "margin": table["margin"].to_numpy()[won],
                   "margin_sign": np.where(table["result"].to_numpy()[won] == 2, 1.0, -1.0)}
    return ModelData(
        teams=teams,
        years=years,
        venues=venues,
        team_a=table["team1"].map(team_number).to_numpy(),
        team_b=table["team2"].map(team_number).to_numpy(),
        year_idx=(table["year"] - years[0]).to_numpy(),
        venue_idx=table["venue_key"].map(venue_number).to_numpy(),
        home=table["home"].to_numpy(),
        decade=((table["year"] - TREND_BASE_YEAR) / 10).to_numpy(),
        result=table["result"].to_numpy(),
        **margins,
    )