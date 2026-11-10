"""Turn posterior draws into probabilities for each Test and for the whole series."""

import itertools

import numpy as np
import pandas as pd


def sigmoid(x):
    """The S-shaped curve that turns any number into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def result_probs(eta, width):
    """Chances of [team A loses, draw, team A wins] for a strength gap eta and a draw band of a given width."""
    eta, half = np.asarray(eta, dtype=float), np.asarray(width, dtype=float) / 2
    lose = sigmoid(-half - eta)
    win = 1 - sigmoid(half - eta)
    return np.stack([lose, 1 - lose - win, win], axis=-1)


def strength_in_year(post, year, rng):
    """Every team's strength in a future year: step the random walk forward from the last fitted year."""
    last = post["years"][-1]
    current = post["strength"][:, -1, :]
    steps = year - last
    if steps <= 0:
        return post["strength"][:, post["years"].index(year), :]
    noise = rng.standard_normal(current.shape) * post["sigma_drift"][:, None] * np.sqrt(steps)
    future = current + noise
    return future - future.mean(axis=1, keepdims=True)


def venue_draws(post, venue, rng):
    """Draws of a venue's effect on the draw band; a venue the model has never seen gets a fresh one."""
    if venue in post["venues"]:
        return post["venue_effect"][:, post["venues"].index(venue)]
    return rng.standard_normal(len(post["sigma_venue"])) * post["sigma_venue"]


def forecast_tests(post, fixtures, home_team, away_team, year, trend_base_year=2019, seed=2027):
    """For each fixture, the chance of a home win, a draw and an away win, for every posterior draw."""
    rng = np.random.default_rng(seed)
    strength = strength_in_year(post, year, rng)
    h, a = post["teams"].index(home_team), post["teams"].index(away_team)
    eta = strength[:, h] - strength[:, a] + post["home_advantage"]
    decade = (year - trend_base_year) / 10
    per_test = []
    for venue in fixtures["venue"]:
        log_width = post["draw_base"] + venue_draws(post, venue, rng) + post["draw_trend"] * decade
        lose, draw, win = np.moveaxis(result_probs(eta, np.exp(log_width)), -1, 0)
        per_test.append(np.stack([win, draw, lose], axis=-1))   # [home win, draw, away win]
    return np.stack(per_test, axis=1)                             # shape: (draws, tests, 3)


def summarise_tests(draw_probs, fixtures):
    """Average the per-draw chances into one row per Test."""
    mean = draw_probs.mean(axis=0)
    return pd.DataFrame({"test_no": fixtures["test_no"].to_numpy(), "venue": fixtures["venue"].to_numpy(),
                         "p_home": mean[:, 0], "p_draw": mean[:, 1], "p_away": mean[:, 2]})


def series_scorelines(draw_probs):
    """Chance of every series scoreline, averaging over posterior draws (Tests share the same uncertainty)."""
    n_tests = draw_probs.shape[1]
    outcomes = np.array(list(itertools.product(range(3), repeat=n_tests)))
    chances = draw_probs[:, np.arange(n_tests), outcomes].prod(axis=-1).mean(axis=0)
    home_wins = (outcomes == 0).sum(axis=1)
    away_wins = (outcomes == 2).sum(axis=1)
    table = pd.DataFrame({"home_wins": home_wins, "away_wins": away_wins, "probability": chances})
    table = table.groupby(["home_wins", "away_wins"], as_index=False)["probability"].sum()
    return table.sort_values("probability", ascending=False).reset_index(drop=True)


def series_summary(scorelines):
    """Headline series chances from the scoreline table."""
    p = scorelines
    return {
        "home_win_series": p.loc[p["home_wins"] > p["away_wins"], "probability"].sum(),
        "drawn_series": p.loc[p["home_wins"] == p["away_wins"], "probability"].sum(),
        "away_win_series": p.loc[p["home_wins"] < p["away_wins"], "probability"].sum(),
    }