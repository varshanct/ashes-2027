"""Checks that the sampler explored the posterior properly, written from first principles."""

import numpy as np
import pandas as pd


def split_chains(x):
    """Cut every chain in half, so a chain that drifts from start to end shows up as disagreement."""
    x = np.asarray(x, dtype=float)
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)


def split_rhat(x):
    """Split R-hat for draws shaped (chains, draws): about 1.00 when the chains agree, above 1.01 when they don't."""
    s = split_chains(x)
    n = s.shape[1]
    within = s.var(axis=1, ddof=1).mean()
    between = n * s.mean(axis=1).var(ddof=1)
    pooled = (n - 1) / n * within + between / n
    return float(np.sqrt(pooled / within))


def effective_sample_size(x):
    """How many independent draws the correlated draws are worth (Geyer's initial positive sequence)."""
    s = split_chains(x)
    m, n = s.shape
    centred = s - s.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    autocov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    within = (autocov[:, 0] * n / (n - 1)).mean()
    pooled = (n - 1) / n * within + s.mean(axis=1).var(ddof=1)
    rho = 1 - (within - autocov.mean(axis=0)) / pooled
    rho[0] = 1.0
    total, previous = 0.0, np.inf
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        pair = min(pair, previous)      # keep the sequence from increasing
        total += pair
        previous = pair
    tau = max(-1 + 2 * total, 1 / np.log10(m * n))
    return float(m * n / tau)


def diagnose(draws):
    """One row per parameter: worst R-hat and smallest effective sample size across its elements."""
    rows = []
    for name, values in draws.items():
        flat = values.reshape(values.shape[0], values.shape[1], -1)
        rhats = [split_rhat(flat[:, :, i]) for i in range(flat.shape[2])]
        esses = [effective_sample_size(flat[:, :, i]) for i in range(flat.shape[2])]
        rows.append({"parameter": name, "elements": flat.shape[2],
                     "max_rhat": max(rhats), "min_ess": min(esses)})
    return pd.DataFrame(rows)
