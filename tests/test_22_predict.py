import numpy as np
import pandas as pd
import pytest

from cricket.predict import (
    forecast_tests,
    result_probs,
    series_scorelines,
    series_summary,
    strength_in_year,
    summarise_tests,
)


def fake_post(n=400, seed=0):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 0.1, size=(n, 3, 2))
    strength[:, :, 0] += 0.2       # team 0 (England) a little stronger
    strength -= strength.mean(axis=2, keepdims=True)
    return {"teams": ["England", "Australia"], "years": [2024, 2025, 2026], "venues": ["Lord's", "Edgbaston"],
            "strength": strength, "sigma_drift": np.full(n, 0.1), "sigma_venue": np.full(n, 0.3),
            "home_advantage": np.full(n, 0.3), "draw_base": np.full(n, -0.2), "draw_trend": np.full(n, -0.3),
            "venue_effect": np.tile([0.4, -0.4], (n, 1))}


FIXTURES = pd.DataFrame({"test_no": [1, 2, 3], "venue": ["Lord's", "Edgbaston", "Utilita Bowl"]})


def test_result_probs_add_to_one_and_respond_to_strength():
    p = result_probs(np.array([-2.0, 0.0, 2.0]), np.array([0.8, 0.8, 0.8]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > p[0, 2] and p[2, 2] > p[2, 0]    # weaker side loses more, stronger wins more
    assert p[1, 0] == pytest.approx(p[1, 2])            # level sides: win and loss equally likely


def test_wider_band_means_more_draws():
    narrow, wide = result_probs(0.0, 0.5), result_probs(0.0, 2.0)
    assert wide[1] > narrow[1]


def test_draw_probability_for_level_sides():
    # For level sides the draw chance is tanh(width / 4).
    assert result_probs(0.0, 0.8)[1] == pytest.approx(np.tanh(0.2))


def test_strength_in_future_year_is_centred_and_spreads_out():
    post = fake_post()
    rng = np.random.default_rng(1)
    future = strength_in_year(post, 2028, rng)
    np.testing.assert_allclose(future.sum(axis=1), 0.0, atol=1e-12)
    assert future[:, 0].std() > post["strength"][:, -1, 0].std()


def test_strength_in_fitted_year_is_unchanged():
    post = fake_post()
    np.testing.assert_array_equal(strength_in_year(post, 2025, np.random.default_rng(1)), post["strength"][:, 1, :])


def test_forecast_shapes_and_venue_effects():
    probs = forecast_tests(fake_post(), FIXTURES, "England", "Australia", 2027)
    assert probs.shape == (400, 3, 3)
    np.testing.assert_allclose(probs.sum(axis=2), 1.0)
    summary = summarise_tests(probs, FIXTURES)
    lords, edgbaston = summary.loc[0], summary.loc[1]
    assert lords["p_draw"] > edgbaston["p_draw"]          # Lord's has the larger venue effect
    assert summary["p_home"].gt(summary["p_away"]).all()  # stronger side at home


def test_unseen_venue_still_gets_a_forecast():
    summary = summarise_tests(forecast_tests(fake_post(), FIXTURES, "England", "Australia", 2027), FIXTURES)
    assert 0 < summary.loc[2, "p_draw"] < 1


def test_series_scorelines_add_to_one():
    probs = forecast_tests(fake_post(), FIXTURES, "England", "Australia", 2027)
    table = series_scorelines(probs)
    assert table["probability"].sum() == pytest.approx(1.0)
    assert (table["home_wins"] + table["away_wins"]).le(3).all()


def test_series_summary_matches_hand_calculation():
    # One draw, two Tests, each certain to be a home win then an away win: series drawn 1-1.
    probs = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    summary = series_summary(series_scorelines(probs))
    assert summary == {"home_win_series": 0.0, "drawn_series": 1.0, "away_win_series": 0.0}


def test_shared_uncertainty_differs_from_independent_tests():
    # Two equally likely worlds: home side strong, or away side strong. Tests are linked through the world.
    world_a = [[0.9, 0.0, 0.1]] * 2
    world_b = [[0.1, 0.0, 0.9]] * 2
    shared = series_summary(series_scorelines(np.array([world_a, world_b])))
    independent = series_summary(series_scorelines(np.array([[[0.5, 0.0, 0.5]] * 2])))
    assert shared["drawn_series"] == pytest.approx(0.18)       # 0.9*0.1*2 averaged over both worlds
    assert independent["drawn_series"] == pytest.approx(0.5)
