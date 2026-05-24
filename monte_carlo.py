"""
monte_carlo.py — Monte Carlo Option Pricer

Uses simulated GBM paths to estimate option prices:
    Price = e^(-rT) * (1/N) * sum(payoff_i)

Payoffs:
    Call: max(S_T - K, 0)
    Put:  max(K - S_T, 0)
"""

import numpy as np
from gbm import simulate_gbm, terminal_prices


def payoff(S_T: np.ndarray, K: float, option_type: str = "call") -> np.ndarray:
    """
    Compute option payoff at expiry for each simulated terminal price.

    Parameters
    ----------
    S_T         : np.ndarray — Terminal stock prices (shape: N,)
    K           : float      — Strike price
    option_type : str        — 'call' or 'put'

    Returns
    -------
    np.ndarray of payoffs (shape: N,)
    """
    option_type = option_type.lower()
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def mc_price(S0: float, K: float, r: float, sigma: float, T: float,
             M: int, N: int, option_type: str = "call",
             seed: int = 42) -> dict:
    """
    Price a European option using Monte Carlo simulation.

    Parameters
    ----------
    S0          : float — Initial stock price
    K           : float — Strike price
    r           : float — Risk-free rate (annualized)
    sigma       : float — Volatility (annualized)
    T           : float — Time to expiry in years
    M           : int   — Time steps per path
    N           : int   — Number of simulation paths
    option_type : str   — 'call' or 'put'
    seed        : int   — Random seed for reproducibility

    Returns
    -------
    dict with keys:
        price          — Estimated option price
        std_error      — Standard error of the estimate
        conf_interval  — 95% confidence interval (lower, upper)
        payoffs        — Raw payoff array (for plotting)
        paths          — Simulated price paths (for plotting)
    """
    # 1. Simulate paths
    paths = simulate_gbm(S0, r, sigma, T, M, N, seed=seed)

    # 2. Extract terminal prices
    S_T = terminal_prices(paths)

    # 3. Compute payoffs
    payoffs = payoff(S_T, K, option_type)

    # 4. Discount expected payoff to present value
    discount_factor = np.exp(-r * T)
    price = discount_factor * np.mean(payoffs)

    # 5. Standard error: sigma_payoff / sqrt(N), then discounted
    std_payoff = np.std(payoffs, ddof=1)
    std_error  = discount_factor * std_payoff / np.sqrt(N)

    # 6. 95% confidence interval (±1.96 standard errors)
    conf_interval = (price - 1.96 * std_error, price + 1.96 * std_error)

    return {
        "price":         float(price),
        "std_error":     float(std_error),
        "conf_interval": (float(conf_interval[0]), float(conf_interval[1])),
        "payoffs":       payoffs,
        "paths":         paths,
        "S_T":           S_T,
    }


def convergence_study(S0: float, K: float, r: float, sigma: float, T: float,
                      M: int, option_type: str = "call",
                      sim_counts: list = None, seed: int = 42) -> dict:
    """
    Run MC pricing at increasing N values to study convergence.

    Parameters
    ----------
    sim_counts : list of ints — N values to test (default: powers of 10 from 100 to 100000)

    Returns
    -------
    dict with keys: sim_counts, prices, std_errors
    """
    if sim_counts is None:
        sim_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]

    prices     = []
    std_errors = []

    for N in sim_counts:
        result = mc_price(S0, K, r, sigma, T, M, N, option_type, seed=seed)
        prices.append(result["price"])
        std_errors.append(result["std_error"])

    return {
        "sim_counts": sim_counts,
        "prices":     prices,
        "std_errors": std_errors,
    }