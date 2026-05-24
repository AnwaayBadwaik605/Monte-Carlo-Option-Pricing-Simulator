"""
gbm.py — Geometric Brownian Motion Simulator

Simulates stock price paths using the exact discrete solution:
    S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*eps)
where eps ~ N(0,1)
"""

import numpy as np


def simulate_gbm(S0: float, r: float, sigma: float, T: float,
                 M: int, N: int, seed: int = None) -> np.ndarray:
    """
    Simulate N stock price paths using GBM over M time steps.

    Parameters
    ----------
    S0    : float  — Initial stock price
    r     : float  — Risk-free rate (annualized)
    sigma : float  — Volatility (annualized)
    T     : float  — Time to expiry in years
    M     : int    — Number of time steps per path
    N     : int    — Number of simulation paths
    seed  : int    — Random seed for reproducibility (optional)

    Returns
    -------
    paths : np.ndarray of shape (M+1, N)
        Each column is one simulated price path.
        Row 0 = S0, Row M = terminal price.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / M

    # Pre-compute the deterministic and stochastic components
    drift     = (r - 0.5 * sigma ** 2) * dt          # Itô-corrected drift per step
    diffusion = sigma * np.sqrt(dt)                   # Volatility scaling per step

    # Draw all random shocks at once: shape (M, N)
    Z = np.random.standard_normal((M, N))

    # Log-returns at each step: shape (M, N)
    log_returns = drift + diffusion * Z

    # Build paths: prepend a row of zeros, then cumsum, then exponentiate
    # Shape becomes (M+1, N)
    log_paths = np.vstack([np.zeros(N), np.cumsum(log_returns, axis=0)])

    paths = S0 * np.exp(log_paths)

    return paths


def terminal_prices(paths: np.ndarray) -> np.ndarray:
    """Return the final stock price from each simulated path."""
    return paths[-1, :]


def get_time_grid(T: float, M: int) -> np.ndarray:
    """Return evenly spaced time points from 0 to T."""
    return np.linspace(0, T, M + 1)