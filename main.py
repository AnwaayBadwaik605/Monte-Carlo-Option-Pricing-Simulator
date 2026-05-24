"""
main.py — Monte Carlo Option Pricing — Entry Point

Runs the full pipeline:
  1. Set parameters
  2. Run Monte Carlo simulation
  3. Compute Black-Scholes analytical price
  4. Compute Greeks
  5. Run convergence study
  6. Render full dashboard
"""

from black_scholes import bs_price, bs_greeks
from monte_carlo import mc_price, convergence_study
from visualizer import full_dashboard

# ── Parameters ───────────────────────────────────────────────────────────────
PARAMS = {
    "S0":    100.0,   # Current stock price ($)
    "K":     105.0,   # Strike price ($)
    "r":     0.05,    # Risk-free rate (5% per annum)
    "sigma": 0.20,    # Volatility (20% per annum)
    "T":     2.0,     # Time to expiry (2 years)
    "N":     100_000,  # Number of Monte Carlo simulations
    "M":     252,     # Time steps (trading days in a year)
}

OPTION_TYPE = "call"   # "call" or "put"
SEED        = 42       # For reproducibility
SAVE_PATH   = "dashboard.png"

# ── Run pipeline ─────────────────────────────────────────────────────────────
def main():
    S0    = PARAMS["S0"]
    K     = PARAMS["K"]
    r     = PARAMS["r"]
    sigma = PARAMS["sigma"]
    T     = PARAMS["T"]
    N     = PARAMS["N"]
    M     = PARAMS["M"]

    print("=" * 55)
    print("  Monte Carlo Option Pricing")
    print("=" * 55)
    print(f"  Option Type : {OPTION_TYPE.upper()}")
    print(f"  S0={S0}  K={K}  r={r*100:.1f}%  σ={sigma*100:.1f}%  T={T}yr")
    print(f"  Simulations : {N:,}   Time steps : {M}")
    print("=" * 55)

    # 1. Monte Carlo price
    print("\n[1/4] Running Monte Carlo simulation...")
    mc_result = mc_price(S0, K, r, sigma, T, M, N,
                         option_type=OPTION_TYPE, seed=SEED)
    print(f"      MC Price   = ${mc_result['price']:.5f}")
    print(f"      Std Error  = {mc_result['std_error']:.5f}")
    ci = mc_result["conf_interval"]
    print(f"      95% CI     = [${ci[0]:.4f}, ${ci[1]:.4f}]")

    # 2. Black-Scholes price
    print("\n[2/4] Computing Black-Scholes price...")
    bs = bs_price(S0, K, r, sigma, T, option_type=OPTION_TYPE)
    print(f"      BS Price   = ${bs:.5f}")
    print(f"      |Error|    = {abs(mc_result['price'] - bs):.5f}")

    # 3. Greeks
    print("\n[3/4] Computing Greeks...")
    greeks = bs_greeks(S0, K, r, sigma, T, option_type=OPTION_TYPE)
    for name, val in greeks.items():
        print(f"      {name.capitalize():6s} = {val:+.6f}")

    # 4. Convergence study
    print("\n[4/4] Running convergence study...")
    convergence = convergence_study(
        S0, K, r, sigma, T, M,
        option_type=OPTION_TYPE,
        sim_counts=[100, 500, 1000, 5000, 10000, 50000, 100000],
        seed=SEED
    )
    print("      Done.")

    # 5. Dashboard
    print("\n[→] Rendering dashboard...")
    full_dashboard(
        mc_result   = mc_result,
        bs_price_val= bs,
        greeks      = greeks,
        convergence = convergence,
        params      = PARAMS,
        option_type = OPTION_TYPE,
        save_path   = SAVE_PATH,
    )

    print("\n[✓] All done!")


if __name__ == "__main__":
    main()