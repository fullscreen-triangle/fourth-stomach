"""
Panel generation for Paper 4: A Mathematical Foundation for Price
5 panels, each with 4 subplots (at least one 3D), white background.
figsize=(20,5), dpi=150
SIGMA = 100.0 canonical.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

SIGMA = 100.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)


# ─────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────

def make_agent(beta):
    kappa = SIGMA / (SIGMA + beta)
    sigma = SIGMA * (1 - kappa) / kappa
    return {"beta": beta, "kappa": kappa, "sigma": sigma}


def agent_floor(agent):
    kappa = agent["kappa"]
    sigma = agent["sigma"]
    return agent["beta"] * (sigma * kappa / (1 - kappa)) / SIGMA


def composite_floor(agents):
    n = len(agents)
    if n == 0:
        return SIGMA
    product = 1.0
    for a in agents:
        product *= agent_floor(a)
    return product / (SIGMA ** (n - 1))


def aggregate_floor(agents):
    return min(agent_floor(a) for a in agents)


def trading_value(tau, agents):
    return tau - aggregate_floor(agents)


def informational_value(tau, agents):
    return tau - composite_floor(agents)


def information_premium(agents):
    return aggregate_floor(agents) - composite_floor(agents)


def trading_spread(agents):
    return 2 * aggregate_floor(agents)


def informational_spread(agents):
    return 2 * composite_floor(agents)


STYLE = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "serif",
}

COLORS = ["#1f4e79", "#c55a11", "#538135", "#7030a0", "#c00000"]


def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.4, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def apply_style_3d(ax, title, xlabel, ylabel, zlabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=5)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=5)
    ax.tick_params(labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")


# ─────────────────────────────────────────────
# Panel 1: Price Cells and Cell-Value Functions
# ─────────────────────────────────────────────
def make_panel_1():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 1: Price Cells C(p,τ) and Cell-Value Decomposition",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: V^T and V^I as functions of (tau, n)
        ax = fig.add_subplot(141, projection="3d")
        taus = np.linspace(5, 60, 30)
        ns3d = np.arange(1, 12)
        TAU3d, N3d = np.meshgrid(taus, ns3d)
        beta_fixed = 15.0
        VT = np.zeros_like(TAU3d)
        VI = np.zeros_like(TAU3d)
        for i, n in enumerate(ns3d):
            agents = [make_agent(beta_fixed)] * int(n)
            agg = aggregate_floor(agents)
            comp = composite_floor(agents)
            for j, tau in enumerate(taus):
                VT[i, j] = max(0, tau - agg)
                VI[i, j] = max(0, tau - comp)
        surf_vt = ax.plot_surface(TAU3d, N3d, VT, cmap="Blues", alpha=0.6, label="V^T")
        surf_vi = ax.plot_surface(TAU3d, N3d, VI, cmap="Oranges", alpha=0.6, label="V^I")
        apply_style_3d(ax, "(a) V^T (blue) and V^I (orange) vs (τ, n)", "τ", "n agents", "Value")

        # (b) V^T and V^I vs tau for fixed ensemble
        ax2 = axes[1]
        betas = [10.0, 20.0, 35.0]
        agents = [make_agent(b) for b in betas]
        agg = aggregate_floor(agents)
        comp = composite_floor(agents)
        taus_plot = np.linspace(0, 60, 100)
        vt_vals = np.maximum(0, taus_plot - agg)
        vi_vals = np.maximum(0, taus_plot - comp)
        ax2.plot(taus_plot, vt_vals, color=COLORS[0], linewidth=2, label=f"V^T (agg={agg:.1f})")
        ax2.plot(taus_plot, vi_vals, color=COLORS[1], linewidth=2,
                  linestyle="--", label=f"V^I (comp={comp:.2f})")
        ax2.fill_between(taus_plot, vt_vals, vi_vals, alpha=0.15, color=COLORS[2],
                          label="Information premium")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) V^T and V^I vs Cell Radius τ (n=3)", "τ (cell radius)", "Value")

        # (c) Information premium vs n agents
        ax3 = axes[2]
        ns_plot = range(1, 16)
        beta_configs = {
            "Uniform β=15": [15.0] * 15,
            "Mixed β∈[5,40]": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
                                 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0],
            "Low β=[2,20]": [max(1, 20 - k) for k in range(15)],
        }
        for label, betas_cfg in beta_configs.items():
            prems = []
            for k in ns_plot:
                a = [make_agent(b) for b in betas_cfg[:k]]
                prems.append(information_premium(a))
            ax3.plot(list(ns_plot), prems, linewidth=2, label=label)
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Information Premium Π = S_flat_agg − S_flat_comp vs n",
                    "n agents", "Π (information premium)")

        # (d) Cell-value function: varying beta composition
        ax4 = axes[3]
        tau_fixed = 30.0
        beta_grid = np.linspace(1, 99, 60)
        vt_single = [max(0, tau_fixed - b) for b in beta_grid]
        vi_single = [max(0, tau_fixed - b) for b in beta_grid]
        ax4.plot(beta_grid, vt_single, color=COLORS[0], linewidth=2, label="V^T, n=1")
        # For n=3 uniform
        n3_comps = [composite_floor([make_agent(b)] * 3) for b in beta_grid]
        vi_n3 = [max(0, tau_fixed - c) for c in n3_comps]
        ax4.plot(beta_grid, vi_n3, color=COLORS[1], linewidth=2,
                  linestyle="--", label="V^I, n=3 uniform")
        ax4.axvline(tau_fixed, color=COLORS[2], linestyle=":", linewidth=1.2,
                     label=f"β = τ = {tau_fixed}")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) V^T and V^I vs Agent Floor β (τ=30)",
                    "β (agent floor)", "Value")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper4_panel_1.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 2: Dual Spreads and Two-Value Theorem
# ─────────────────────────────────────────────
def make_panel_2():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 2: Dual-Spread Theorem — Trading Spread ΔᵀΔᴵ and Information Premium Π",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Trading vs informational spread surface (beta_min vs n)
        ax = fig.add_subplot(141, projection="3d")
        beta_mins = np.linspace(1, 50, 25)
        ns3d = np.arange(1, 12)
        BM, N3d = np.meshgrid(beta_mins, ns3d)
        DT = 2 * BM  # trading spread = 2*min floor = 2*beta_min (for uniform ensemble min=beta_min)
        # Informational spread for uniform ensemble: 2*(beta_min/SIGMA)^n * SIGMA
        DI = 2 * (BM / SIGMA) ** N3d * SIGMA
        surf_dt = ax.plot_surface(BM, N3d, DT, cmap="Blues", alpha=0.6)
        surf_di = ax.plot_surface(BM, N3d, DI, cmap="Reds", alpha=0.6)
        apply_style_3d(ax, "(a) ΔT (blue) vs ΔI (red) Surface (uniform β)",
                        "β_min", "n agents", "Spread")

        # (b) ΔT and ΔI vs n for several ensembles
        ax2 = axes[1]
        ns_plot = range(1, 16)
        configs = {
            "Uniform β=20": [20.0] * 15,
            "Heterogeneous [5,40]": [5.0 + 2.5*k for k in range(15)],
            "Low floors [1,15]": [1.0 + k for k in range(15)],
        }
        for label, betas_cfg in configs.items():
            dt_vals = []
            di_vals = []
            for k in ns_plot:
                a = [make_agent(b) for b in betas_cfg[:k]]
                dt_vals.append(trading_spread(a))
                di_vals.append(informational_spread(a))
            ax2.plot(list(ns_plot), dt_vals, linewidth=2, label=f"ΔT: {label}")
            ax2.plot(list(ns_plot), di_vals, linewidth=2, linestyle="--")
        ax2.legend(fontsize=7.5)
        apply_style(ax2, "(b) Trading Spread ΔT (solid) and Info Spread ΔI (dashed) vs n",
                    "n agents", "Spread = 2·S_flat")

        # (c) Premium Π vs n
        ax3 = axes[2]
        for label, betas_cfg in configs.items():
            prems = []
            for k in ns_plot:
                a = [make_agent(b) for b in betas_cfg[:k]]
                prems.append(information_premium(a))
            ax3.plot(list(ns_plot), prems, linewidth=2, label=label)
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Information Premium Π = S_flat_agg − S_flat_comp vs n",
                    "n agents", "Π (premium)")

        # (d) Two-Value Theorem: scatter S_flat_comp vs S_flat_agg
        ax4 = axes[3]
        n_pts = 200
        betas_random = np.random.uniform(1, 80, (n_pts, 5))
        agg_pts = []
        comp_pts = []
        for row in betas_random:
            agents = [make_agent(b) for b in row]
            agg_pts.append(aggregate_floor(agents))
            comp_pts.append(composite_floor(agents))
        agg_pts = np.array(agg_pts)
        comp_pts = np.array(comp_pts)
        ax4.scatter(agg_pts, comp_pts, alpha=0.4, color=COLORS[0], s=15, label="Random ensembles (n=5)")
        diag = np.linspace(0, max(agg_pts), 100)
        ax4.plot(diag, diag, color=COLORS[1], linewidth=1.5, linestyle="--", label="S_flat_comp = S_flat_agg")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Two-Value Theorem: S_flat_comp ≤ S_flat_agg",
                    "S_flat_agg (trading floor)", "S_flat_comp (info floor)")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper4_panel_2.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 3: Equilibrium Price and Bid-Ask Structure
# ─────────────────────────────────────────────
def make_panel_3():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 3: Equilibrium Price p*, Bid-Ask Structure, and Price Discovery Rate",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Bid-ask spread as function of (n, beta_min)
        ax = fig.add_subplot(141, projection="3d")
        ns3d = np.arange(1, 16)
        beta_mins = np.linspace(1, 60, 25)
        N3d, BM3d = np.meshgrid(ns3d, beta_mins)
        SPREAD3d = 2 * BM3d  # trading spread = 2 * S_flat_agg = 2 * beta_min
        surf = ax.plot_surface(N3d, BM3d, SPREAD3d, cmap="Blues", alpha=0.85)
        apply_style_3d(ax, "(a) Trading Spread ΔT vs (n, β_min)",
                        "n agents", "β_min", "ΔT = 2·β_min")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Bid-ask bands around p* as n increases
        ax2 = axes[1]
        p_star = 100.0
        betas_seq = [50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 1.5, 0.8]
        ns_plot = range(1, len(betas_seq) + 1)
        bids = []
        asks = []
        for k in ns_plot:
            agents = [make_agent(b) for b in betas_seq[:k]]
            agg = aggregate_floor(agents)
            bids.append(p_star - agg)
            asks.append(p_star + agg)
        ax2.fill_between(list(ns_plot), bids, asks, alpha=0.2, color=COLORS[0],
                          label="Bid-ask band")
        ax2.plot(list(ns_plot), bids, color=COLORS[0], linewidth=2, label="Bid")
        ax2.plot(list(ns_plot), asks, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Ask")
        ax2.axhline(p_star, color=COLORS[2], linewidth=1.5, linestyle=":", label="p* = 100")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Bid-Ask Bands Converging to p* = 100",
                    "n agents", "Price")

        # (c) Price discovery rate: spread decay
        ax3 = axes[2]
        beta_seqs = {
            "Harmonic: β_k = 50/k": [50.0 / k for k in range(1, 16)],
            "Geometric: β_k = 50·0.8^k": [50.0 * 0.8**k for k in range(1, 16)],
            "Uniform β=10": [10.0] * 15,
        }
        ns_p = range(1, 16)
        for label, b_seq in beta_seqs.items():
            spreads = []
            for k in ns_p:
                agents = [make_agent(b) for b in b_seq[:k]]
                spreads.append(trading_spread(agents))
            ax3.plot(list(ns_p), spreads, linewidth=2, label=label)
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Price Discovery Rate: Spread vs n",
                    "n agents", "Trading spread ΔT = 2·S_flat_agg")

        # (d) Equilibrium price cell: concentric circles visualization
        ax4 = axes[3]
        theta = np.linspace(0, 2 * np.pi, 300)
        p_star_x, p_star_y = 0.0, 0.0
        radii_configs = [
            (30.0, "τ = 30 (cell radius)", COLORS[3], "--"),
            (15.0, "S_flat_agg (trading spread/2)", COLORS[0], "-"),
            (5.0, "S_flat_comp (info spread/2)", COLORS[1], ":"),
        ]
        for r, label, color, ls in radii_configs:
            ax4.plot(r * np.cos(theta), r * np.sin(theta),
                      color=color, linewidth=2, linestyle=ls, label=label)
        ax4.plot(p_star_x, p_star_y, "k*", markersize=12, label="p* (equilibrium)")
        ax4.set_aspect("equal")
        ax4.legend(fontsize=8, loc="upper right")
        apply_style(ax4, "(d) Price Cell Structure: Cell, Bid-Ask, and Info Spread",
                    "Outcome space (x)", "Outcome space (y)")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper4_panel_3.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 4: Fundamental Value and No-Arbitrage
# ─────────────────────────────────────────────
def make_panel_4():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 4: Fundamental Value Convergence and No-Arbitrage Conditions",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Composite floor decay surface (V* convergence)
        ax = fig.add_subplot(141, projection="3d")
        betas3d = np.linspace(1, 50, 30)
        ns3d = np.arange(1, 20)
        B3d, N3d = np.meshgrid(betas3d, ns3d)
        COMP3d = SIGMA * (B3d / SIGMA) ** N3d
        surf = ax.plot_surface(B3d, N3d, COMP3d, cmap="YlOrRd_r", alpha=0.85)
        apply_style_3d(ax, "(a) S_flat_comp(n) = Σ·(β/Σ)^n for Uniform Ensemble",
                        "β (floor)", "n agents", "S_flat_comp")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Fundamental value: comp floor width converging
        ax2 = axes[1]
        true_fv = 100.0
        beta_seq_e = [40.0 * (0.75**k) for k in range(20)]  # Class E: decays to 0
        beta_seq_o = [SIGMA * (1 - 1/(k+2)**2) for k in range(20)]  # Class Omega

        ns_p = range(1, 21)
        for label, b_seq in [("Class E (→ point FV)", beta_seq_e),
                               ("Class Ω (→ interval FV)", beta_seq_o)]:
            widths = []
            for k in ns_p:
                agents = [make_agent(b) for b in b_seq[:k]]
                widths.append(2 * composite_floor(agents))
            ax2.semilogy(list(ns_p), widths, linewidth=2, label=label)
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Fundamental Value Interval Width 2·S_flat_comp (log scale)",
                    "n agents", "Width of FV interval (log scale)")

        # (c) No-arbitrage bound: |p1 - p2| <= agg1 + agg2
        ax3 = axes[2]
        n_pts = 150
        beta_samples = np.random.uniform(1, 60, (n_pts, 3))
        agg_sums = []
        for row in beta_samples:
            a1 = [make_agent(row[0])]
            a2 = [make_agent(row[1]), make_agent(row[2])]
            agg_sums.append(aggregate_floor(a1) + aggregate_floor(a2))
        agg_sums = np.array(agg_sums)
        # Price differences are 0 (same cell, same p*) so all points satisfy no-arb
        ax3.hist(agg_sums, bins=25, color=COLORS[0], edgecolor="white", alpha=0.85)
        ax3.axvline(np.mean(agg_sums), color=COLORS[1], linewidth=2,
                     linestyle="--", label=f"Mean bound = {np.mean(agg_sums):.1f}")
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) No-Arbitrage Bound Distribution (|Δp| ≤ agg₁ + agg₂)",
                    "S_flat_agg₁ + S_flat_agg₂", "Count")

        # (d) Fundamental value certainty vs ensemble class
        ax4 = axes[3]
        ns_plot = range(1, 25)
        class_e_betas = [SIGMA / (k + 1) for k in range(1, 25)]
        class_o_betas = [SIGMA * (1 - 1 / (k + 1)**2) for k in range(1, 25)]

        comp_e = [composite_floor([make_agent(b) for b in class_e_betas[:k]]) for k in ns_plot]
        comp_o = [composite_floor([make_agent(b) for b in class_o_betas[:k]]) for k in ns_plot]

        fv = 100.0
        ax4.fill_between(list(ns_plot),
                          [fv - c for c in comp_e], [fv + c for c in comp_e],
                          alpha=0.3, color=COLORS[0], label="Class E uncertainty")
        ax4.fill_between(list(ns_plot),
                          [fv - c for c in comp_o], [fv + c for c in comp_o],
                          alpha=0.3, color=COLORS[1], label="Class Ω uncertainty")
        ax4.axhline(fv, color="black", linewidth=1.5, linestyle=":", label="True FV = 100")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) FV Uncertainty Interval by Universality Class",
                    "n agents", "Fundamental value range")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper4_panel_4.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 5: Transaction Conditions and Law of One Price
# ─────────────────────────────────────────────
def make_panel_5():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 5: Transaction Conditions, Law of One Price, and Market Integration",
            fontsize=13, fontweight="bold", y=1.02
        )

        tau_fixed = 30.0

        # (a) 3D: Transaction feasibility surface
        ax = fig.add_subplot(141, projection="3d")
        beta_b_grid = np.linspace(0.5, 60, 40)
        beta_s_grid = np.linspace(0.5, 60, 40)
        BB, BS = np.meshgrid(beta_b_grid, beta_s_grid)
        # Transaction iff beta_B + beta_S <= 2*tau
        FEASIBLE = (BB + BS <= 2 * tau_fixed).astype(float)
        surf = ax.plot_surface(BB, BS, FEASIBLE, cmap="RdYlGn", alpha=0.85)
        apply_style_3d(ax, f"(a) Transaction Feasibility (τ={tau_fixed})",
                        "β_buyer", "β_seller", "Feasible (1=Yes)")

        # (b) Transaction boundary: beta_B + beta_S = 2*tau
        ax2 = axes[1]
        beta_b_range = np.linspace(0, 2 * tau_fixed, 200)
        beta_s_boundary = 2 * tau_fixed - beta_b_range
        ax2.plot(beta_b_range, beta_s_boundary, color=COLORS[0], linewidth=2,
                  label=f"β_B + β_S = 2τ = {2*tau_fixed:.0f}")
        ax2.fill_between(beta_b_range, 0, beta_s_boundary, alpha=0.2, color=COLORS[2],
                          label="Transaction region")
        ax2.fill_between(beta_b_range, beta_s_boundary,
                          np.full_like(beta_b_range, 2 * tau_fixed),
                          alpha=0.1, color=COLORS[1], label="No-trade region")
        ax2.set_xlim(0, 2 * tau_fixed)
        ax2.set_ylim(0, 2 * tau_fixed)
        ax2.legend(fontsize=8)
        apply_style(ax2, f"(b) Transaction Boundary β_B + β_S = 2τ (τ={tau_fixed})",
                    "β_buyer (floor)", "β_seller (floor)")

        # (c) Law of One Price: multiple ensembles, same equilibrium center
        ax3 = axes[2]
        p_star = 100.0
        ensemble_configs = {
            "E₁: [5,15,30]": [5.0, 15.0, 30.0],
            "E₂: [10,25,40]": [10.0, 25.0, 40.0],
            "E₃: [2,8,20,50]": [2.0, 8.0, 20.0, 50.0],
            "E₄: [20,35]": [20.0, 35.0],
        }
        y_pos = range(len(ensemble_configs))
        centers = []
        half_spreads_T = []
        half_spreads_I = []
        labels = []
        for label, betas in ensemble_configs.items():
            agents = [make_agent(b) for b in betas]
            centers.append(p_star)
            half_spreads_T.append(aggregate_floor(agents))
            half_spreads_I.append(composite_floor(agents))
            labels.append(label)
        for i, (c, ht, hi, lbl) in enumerate(zip(centers, half_spreads_T, half_spreads_I, labels)):
            ax3.barh(i, 2 * ht, left=c - ht, height=0.3, color=COLORS[0], alpha=0.6,
                      label="Trading spread" if i == 0 else "")
            ax3.barh(i, 2 * hi, left=c - hi, height=0.15, color=COLORS[1], alpha=0.8,
                      label="Info spread" if i == 0 else "")
        ax3.axvline(p_star, color="black", linewidth=1.5, linestyle="--", label="p* = 100")
        ax3.set_yticks(list(y_pos))
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Law of One Price: Same p* Across Ensembles",
                    "Price", "Ensemble")

        # (d) Buyer's bid and seller's ask dynamics
        ax4 = axes[3]
        beta_range = np.linspace(0.5, tau_fixed - 0.1, 100)
        p_s = 100.0
        buyer_bids = p_s + (tau_fixed - beta_range)
        seller_asks = p_s - (tau_fixed - beta_range)
        ax4.plot(beta_range, buyer_bids, color=COLORS[0], linewidth=2, label="Buyer bid = p* + (τ−β_B)")
        ax4.plot(beta_range, seller_asks, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Seller ask = p* − (τ−β_S)")
        ax4.axhline(p_s, color="black", linewidth=1, linestyle=":", label=f"p* = {p_s}")
        ax4.fill_between(beta_range, seller_asks, buyer_bids, alpha=0.15, color=COLORS[2],
                          label="Transaction zone")
        ax4.legend(fontsize=8)
        apply_style(ax4, f"(d) Bid-Ask as Functions of Floor β (τ={tau_fixed}, p*=100)",
                    "β (agent floor)", "Price")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper4_panel_5.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating Paper 4 panels...")
    make_panel_1()
    make_panel_2()
    make_panel_3()
    make_panel_4()
    make_panel_5()
    print("All Paper 4 panels generated.")
