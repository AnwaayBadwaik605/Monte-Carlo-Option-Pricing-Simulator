"""
visualizer.py — Matplotlib Dashboard for Monte Carlo Option Pricing

Produces a full multi-panel figure with:
  1. Simulated GBM price paths
  2. Terminal price distribution with payoff region
  3. Payoff distribution
  4. Convergence of MC price vs Black-Scholes
  5. Greeks summary bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Color palette ────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#e3b341"
MUTED      = "#8b949e"
WHITE      = "#e6edf3"
GRID_COLOR = "#21262d"


def _style():
    """Apply global dark theme styling."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   WHITE,
        "axes.titlecolor":   WHITE,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "text.color":        WHITE,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_price_paths(ax, paths, time_grid, K, S0, n_display=200):
    """Panel 1: Simulated GBM price paths."""
    N = paths.shape[1]
    display = min(n_display, N)

    # Background paths
    ax.plot(time_grid, paths[:, :display], color=ACCENT, alpha=0.04, linewidth=0.5)

    # Highlight a few paths
    for i in range(min(10, display)):
        ax.plot(time_grid, paths[:, i], color=ACCENT, alpha=0.5, linewidth=0.7)

    # Mean path
    mean_path = paths.mean(axis=1)
    ax.plot(time_grid, mean_path, color=YELLOW, linewidth=2, label="Mean path", zorder=5)

    # Strike line
    ax.axhline(K, color=RED, linewidth=1.2, linestyle="--", label=f"Strike K={K}", zorder=4)
    ax.axhline(S0, color=MUTED, linewidth=0.8, linestyle=":", label=f"S₀={S0}", zorder=4)

    ax.set_title("GBM Simulated Price Paths", fontsize=11, pad=10)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price ($)")
    ax.legend(fontsize=8, loc="upper left", facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    ax.grid(True)


def plot_terminal_distribution(ax, S_T, K, option_type="call"):
    """Panel 2: Terminal price histogram with in/out of money shading."""
    bins = min(80, len(S_T) // 50)

    # Split into ITM and OTM
    if option_type == "call":
        itm_mask = S_T > K
        itm_label = f"In the money (S_T > K={K})"
        otm_label = f"Out of money (S_T ≤ K={K})"
    else:
        itm_mask = S_T < K
        itm_label = f"In the money (S_T < K={K})"
        otm_label = f"Out of money (S_T ≥ K={K})"

    ax.hist(S_T[itm_mask],  bins=bins, color=GREEN, alpha=0.7, label=itm_label)
    ax.hist(S_T[~itm_mask], bins=bins, color=RED,   alpha=0.7, label=otm_label)
    ax.axvline(K, color=YELLOW, linewidth=1.5, linestyle="--", label=f"Strike K={K}")
    ax.axvline(np.mean(S_T), color=ACCENT, linewidth=1.2, linestyle="-.",
               label=f"Mean S_T = {np.mean(S_T):.2f}")

    ax.set_title("Terminal Price Distribution", fontsize=11, pad=10)
    ax.set_xlabel("Terminal Stock Price ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    ax.grid(True)


def plot_payoff_distribution(ax, payoffs, price, option_type="call"):
    """Panel 3: Payoff distribution."""
    nonzero = payoffs[payoffs > 0]
    pct_itm = 100 * len(nonzero) / len(payoffs)

    bins = min(60, len(nonzero) // 20) if len(nonzero) > 0 else 10

    ax.hist(payoffs[payoffs == 0], bins=3, color=RED, alpha=0.8,
            label=f"Zero payoff ({100 - pct_itm:.1f}%)")
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=bins, color=GREEN, alpha=0.8,
                label=f"Positive payoff ({pct_itm:.1f}%)")

    ax.axvline(price, color=YELLOW, linewidth=2, linestyle="--",
               label=f"MC Price = {price:.4f}")

    ax.set_title(f"Payoff Distribution ({option_type.capitalize()})", fontsize=11, pad=10)
    ax.set_xlabel("Payoff at Expiry ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    ax.grid(True)


def plot_convergence(ax, sim_counts, mc_prices, std_errors, bs_price):
    """Panel 4: MC price convergence toward Black-Scholes price."""
    sim_arr = np.array(sim_counts)
    mc_arr  = np.array(mc_prices)
    se_arr  = np.array(std_errors)

    # Confidence band
    ax.fill_between(sim_arr,
                    mc_arr - 1.96 * se_arr,
                    mc_arr + 1.96 * se_arr,
                    color=ACCENT, alpha=0.15, label="95% CI")

    ax.plot(sim_arr, mc_arr, color=ACCENT, linewidth=2,
            marker="o", markersize=4, label="MC Price")
    ax.axhline(bs_price, color=YELLOW, linewidth=1.5, linestyle="--",
               label=f"BS Price = {bs_price:.4f}")

    ax.set_xscale("log")
    ax.set_title("Convergence: MC Price → Black-Scholes", fontsize=11, pad=10)
    ax.set_xlabel("Number of Simulations (log scale)")
    ax.set_ylabel("Option Price ($)")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    ax.grid(True)


def plot_greeks(ax, greeks: dict):
    """Panel 5: Greeks bar chart."""
    names  = list(greeks.keys())
    values = list(greeks.values())
    colors = [GREEN if v >= 0 else RED for v in values]

    bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.001 if val >= 0 else -0.001),
                f"{val:.5f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, color=WHITE)

    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Option Greeks", fontsize=11, pad=10)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y")


def plot_summary_box(ax, mc_result, bs_price_val, greeks, params):
    """Panel 6: Text summary box."""
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    mc_p  = mc_result["price"]
    se    = mc_result["std_error"]
    ci    = mc_result["conf_interval"]
    error = abs(mc_p - bs_price_val)

    lines = [
        ("── Parameters ──────────────────", WHITE),
        (f"  S₀     = ${params['S0']:.2f}", MUTED),
        (f"  K      = ${params['K']:.2f}", MUTED),
        (f"  r      = {params['r']*100:.1f}%", MUTED),
        (f"  σ      = {params['sigma']*100:.1f}%", MUTED),
        (f"  T      = {params['T']:.2f} yr", MUTED),
        (f"  N      = {params['N']:,} sims", MUTED),
        ("", WHITE),
        ("── Pricing Results ─────────────", WHITE),
        (f"  BS Price  = ${bs_price_val:.5f}", YELLOW),
        (f"  MC Price  = ${mc_p:.5f}", ACCENT),
        (f"  Std Error = {se:.5f}", MUTED),
        (f"  95% CI    = [{ci[0]:.4f}, {ci[1]:.4f}]", MUTED),
        (f"  |Error|   = {error:.5f}", GREEN if error < se * 2 else RED),
        ("", WHITE),
        ("── Greeks ──────────────────────", WHITE),
        (f"  Δ Delta  = {greeks['delta']:+.5f}", MUTED),
        (f"  Γ Gamma  = {greeks['gamma']:+.5f}", MUTED),
        (f"  ν Vega   = {greeks['vega']:+.5f}", MUTED),
        (f"  Θ Theta  = {greeks['theta']:+.5f}", MUTED),
        (f"  ρ Rho    = {greeks['rho']:+.5f}", MUTED),
    ]

    y = 0.97
    for text, color in lines:
        ax.text(0.05, y, text, transform=ax.transAxes,
                fontsize=8.5, color=color, va="top", fontfamily="monospace")
        y -= 0.047


def full_dashboard(mc_result: dict, bs_price_val: float, greeks: dict,
                   convergence: dict, params: dict,
                   option_type: str = "call",
                   save_path: str = "dashboard.png"):
    """
    Render the full dashboard and save to file.

    Parameters
    ----------
    mc_result    : dict from monte_carlo.mc_price()
    bs_price_val : float from black_scholes.bs_price()
    greeks       : dict from black_scholes.bs_greeks()
    convergence  : dict from monte_carlo.convergence_study()
    params       : dict of S0, K, r, sigma, T, N
    option_type  : 'call' or 'put'
    save_path    : output file path
    """
    _style()

    fig = plt.figure(figsize=(20, 12), facecolor=DARK_BG)
    fig.suptitle(
        f"Monte Carlo Option Pricing Dashboard  ·  European {option_type.capitalize()}",
        fontsize=15, color=WHITE, y=0.98, fontfamily="monospace", fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    axes = [
        fig.add_subplot(gs[0, 0]),   # 0: price paths
        fig.add_subplot(gs[0, 1]),   # 1: terminal dist
        fig.add_subplot(gs[0, 2]),   # 2: payoff dist
        fig.add_subplot(gs[1, 0]),   # 3: convergence
        fig.add_subplot(gs[1, 1]),   # 4: greeks
        fig.add_subplot(gs[1, 2]),   # 5: summary
    ]

    from gbm import get_time_grid
    time_grid = get_time_grid(params["T"], mc_result["paths"].shape[0] - 1)

    plot_price_paths(axes[0], mc_result["paths"], time_grid,
                     params["K"], params["S0"])
    plot_terminal_distribution(axes[1], mc_result["S_T"],
                               params["K"], option_type)
    plot_payoff_distribution(axes[2], mc_result["payoffs"],
                             mc_result["price"], option_type)
    plot_convergence(axes[3], convergence["sim_counts"],
                     convergence["prices"], convergence["std_errors"],
                     bs_price_val)
    plot_greeks(axes[4], greeks)
    plot_summary_box(axes[5], mc_result, bs_price_val, greeks, params)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[✓] Dashboard saved to: {save_path}")
    plt.show()