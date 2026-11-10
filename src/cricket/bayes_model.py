"""The Bayesian team-strength model, written in PyMC and fitted with NUTS (Hamiltonian Monte Carlo)."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PARAMETERS = ["strength", "sigma_start", "sigma_drift", "home_advantage",
              "draw_base", "draw_trend", "sigma_venue", "venue_effect"]


def build_model(data):
    """Team strengths that drift year to year, home advantage, and a draw band that depends on venue and era."""
    coords = {"team": data.teams, "year": data.years, "venue": data.venues}
    with pm.Model(coords=coords) as model:
        # Team strength: a starting level, then a small random step every year (a random walk).
        sigma_start = pm.HalfNormal("sigma_start", 1.0)
        sigma_drift = pm.HalfNormal("sigma_drift", 0.3)
        start_z = pm.Normal("start_z", 0.0, 1.0, dims="team")
        step_z = pm.Normal("step_z", 0.0, 1.0, dims=("year", "team"))
        raw = sigma_start * start_z + sigma_drift * pt.cumsum(step_z, axis=0)
        strength = pm.Deterministic("strength", raw - raw.mean(axis=1, keepdims=True), dims=("year", "team"))

        # Home advantage, the same for every team.
        home_advantage = pm.Normal("home_advantage", 0.0, 1.0)

        # Draw band: a base width, a trend over time, and a pooled effect for each venue.
        draw_base = pm.Normal("draw_base", 0.0, 1.5)
        draw_trend = pm.Normal("draw_trend", 0.0, 1.0)
        sigma_venue = pm.HalfNormal("sigma_venue", 0.5)
        venue_z = pm.Normal("venue_z", 0.0, 1.0, dims="venue")
        venue_effect = pm.Deterministic("venue_effect", sigma_venue * venue_z, dims="venue")

        # Each Test: strength gap plus home advantage, inside a draw band of the right width.
        eta = (strength[data.year_idx, data.team_a] - strength[data.year_idx, data.team_b]
               + home_advantage * data.home)
        log_width = draw_base + venue_effect[data.venue_idx] + draw_trend * data.decade
        half = pt.exp(log_width) / 2
        pm.OrderedLogistic("result", eta=eta, cutpoints=pt.stack([-half, half], axis=-1), observed=data.result)
    return model


def fit(model, draws=1000, tune=1000, chains=4, seed=2027, cores=None, progressbar=True):
    """Run NUTS and return each parameter's draws as plain arrays shaped (chains, draws, ...)."""
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=seed,
                          target_accept=0.95, progressbar=progressbar)
    samples = {name: np.asarray(idata.posterior[name].values) for name in PARAMETERS}
    divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    return samples, divergences