"""
black_scholes.py — Analytical Black-Scholes Option Pricer

Computes exact European option prices using the Black-Scholes formula:
    Call: C = S0*N(d1) - K*e^(-rT)*N(d2)
    Put:  P = K*e^(-rT)*N(-d2) - S0*N(-d1)

Where:
    d1 = [ln(S0/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float):
    """Compute d1 and d2 for the Black-Scholes formula."""
    log_moneyness = np.log(S0 / K)                    # ln(S0/K)
    total_vol     = sigma * np.sqrt(T)                 # sigma * sqrt(T)

    d1 = (log_moneyness + (r + 0.5 * sigma ** 2) * T) / total_vol
    d2 = d1 - total_vol

    return d1, d2


def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes price for a European Call option.

    Parameters
    ----------
    S0    : Current stock price
    K     : Strike price
    r     : Risk-free rate (annualized)
    sigma : Volatility (annualized)
    T     : Time to expiry in years

    Returns
    -------
    float : Call option price
    """
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return float(price)


def bs_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes price for a European Put option.

    Parameters
    ----------
    S0    : Current stock price
    K     : Strike price
    r     : Risk-free rate (annualized)
    sigma : Volatility (annualized)
    T     : Time to expiry in years

    Returns
    -------
    float : Put option price
    """
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return float(price)


def bs_price(S0: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """
    Unified Black-Scholes pricer for call or put.

    Parameters
    ----------
    option_type : str — 'call' or 'put'
    """
    option_type = option_type.lower()
    if option_type == "call":
        return bs_call(S0, K, r, sigma, T)
    elif option_type == "put":
        return bs_put(S0, K, r, sigma, T)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_greeks(S0: float, K: float, r: float, sigma: float, T: float,
              option_type: str = "call") -> dict:
    """
    Compute the main Greeks for a European option.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
    """
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    sqrt_T  = np.sqrt(T)
    exp_rT  = np.exp(-r * T)
    pdf_d1  = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sigma * sqrt_T)
    vega  = S0 * pdf_d1 * sqrt_T / 100          # per 1% change in vol

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (-(S0 * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * exp_rT * norm.cdf(d2)) / 365
        rho   = K * T * exp_rT * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S0 * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * exp_rT * norm.cdf(-d2)) / 365
        rho   = -K * T * exp_rT * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "vega":  round(vega,  6),
        "theta": round(theta, 6),
        "rho":   round(rho,   6),
    }