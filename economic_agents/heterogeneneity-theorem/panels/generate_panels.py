"""
Panel generation for Paper 3: The Heterogeneity Theorem for Market Information
5 panels, each with 4 subplots (at least one 3D), white background.
figsize=(20,5), dpi=150
SIGMA = 100.0 canonical.
"""

import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

SIGMA = 100.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)
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


def log_floor(agent):
    return math.log(SIGMA / agent_floor(agent))


def cv_floors(agents):
    floors = [agent_floor(a) for a in agents]
    mean = np.mean(floors)
    return np.std(floors) / mean if mean > 0 else 0.0


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
# Panel 1: Heterogeneity Dominance (Schur-Concavity)
# ─────────────────────────────────────────────
def make_panel_1():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 1: Heterogeneity Dominance and Schur-Concavity of Composite Floor",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Composite floor as function of (beta1, beta2) for n=2
        ax = fig.add_subplot(141, projection="3d")
        b1_vals = np.linspace(1, 50, 40)
        b2_vals = np.linspace(1, 50, 40)
        B1, B2 = np.meshgrid(b1_vals, b2_vals)
        COMP = (B1 * B2) / SIGMA
        surf = ax.plot_surface(B1, B2, COMP, cmap="Blues", alpha=0.85, linewidth=0)
        apply_style_3d(ax, "(a) Composite Floor Surface (n=2)", "β₁", "β₂", "S_flat_comp")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Composite floor vs CV for fixed mean
        ax2 = axes[1]
        mean_floor = 20.0
        cvs = np.linspace(0.0, 0.9, 50)
        n = 4
        comp_vals = []
        for cv in cvs:
            # Generate floors with given mean and CV using lognormal
            sigma_ln = math.sqrt(math.log(1 + cv ** 2))
            mu_ln = math.log(mean_floor) - sigma_ln ** 2 / 2
            floors = np.exp(np.random.normal(mu_ln, sigma_ln, n))
            floors = floors * (mean_floor / floors.mean())
            agents = [make_agent(float(f)) for f in floors]
            comp_vals.append(composite_floor(agents))
        ax2.plot(cvs, comp_vals, color=COLORS[0], linewidth=2)
        ax2.axhline(composite_floor([make_agent(mean_floor)] * n), color=COLORS[1],
                    linestyle="--", label=f"Homogeneous (CV=0)")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Composite Floor vs CV (n=4, β̄=20)", "CV(floors)", "S_flat_comp")

        # (c) Schur-Ostrowski criterion verification
        ax3 = axes[2]
        betas_range = np.linspace(1, 99, 60)
        beta_j = 30.0
        # d/d_beta_i log(f) = 1/beta_i, so (beta_i - beta_j)(1/beta_i - 1/beta_j) <= 0
        criterion = (betas_range - beta_j) * (1.0 / betas_range - 1.0 / beta_j)
        ax3.plot(betas_range, criterion, color=COLORS[2], linewidth=2)
        ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax3.fill_between(betas_range, criterion, 0, where=(criterion <= 0),
                          alpha=0.2, color=COLORS[2])
        apply_style(ax3, "(c) Schur-Ostrowski Criterion (βⱼ=30)",
                    "βᵢ", "(βᵢ−βⱼ)(1/βᵢ−1/βⱼ)")

        # (d) Dominance: homogeneous vs heterogeneous ensembles
        ax4 = axes[3]
        means = np.linspace(5, 60, 30)
        n = 5
        spread = 15.0
        comp_homo = []
        comp_het = []
        for m in means:
            agents_h = [make_agent(m)] * n
            betas_het = [max(0.1, m - spread), max(0.1, m - spread/2),
                          m, min(99, m + spread/2), min(99, m + spread)]
            agents_e = [make_agent(b) for b in betas_het]
            comp_homo.append(composite_floor(agents_h))
            comp_het.append(composite_floor(agents_e))
        ax4.plot(means, comp_homo, color=COLORS[0], linewidth=2, label="Homogeneous")
        ax4.plot(means, comp_het, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Heterogeneous")
        ax4.fill_between(means, comp_het, comp_homo, alpha=0.15, color=COLORS[2],
                          label="Efficiency gain")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Homogeneous vs Heterogeneous (n=5)", "Mean floor β̄", "S_flat_comp")

        # remove the unused first axis slot
        axes[0].remove()

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper3_panel_1.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 2: Borel-Cantelli Classification & Universality Classes
# ─────────────────────────────────────────────
def make_panel_2():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 2: Borel-Cantelli Classification and Universality Classes (E vs Ω)",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Composite floor surface for class E vs class Omega sequences
        ax = fig.add_subplot(141, projection="3d")
        ns = np.arange(1, 31)
        # Class E: beta_k = SIGMA/(k+1) -> log-floor = log(k+1), sum diverges
        comp_e = []
        comp_o = []
        for n in ns:
            agents_e = [make_agent(SIGMA / (k + 1)) for k in range(1, n + 1)]
            agents_o = [make_agent(SIGMA * (1 - 1 / (k + 1) ** 2)) for k in range(1, n + 1)]
            comp_e.append(composite_floor(agents_e))
            comp_o.append(composite_floor(agents_o))
        comp_e = np.array(comp_e)
        comp_o = np.array(comp_o)
        ax.plot(ns, comp_e, np.zeros(len(ns)), color=COLORS[0], linewidth=2, label="Class E")
        ax.plot(ns, comp_o, np.ones(len(ns)), color=COLORS[1], linewidth=2, label="Class Ω")
        ax.set_xlabel("n", fontsize=8)
        ax.set_ylabel("Class", fontsize=8)
        ax.set_zlabel("S_flat_comp", fontsize=8)
        ax.set_title("(a) E vs Ω: Composite Floor", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # (b) Log-floor cumulative sum
        ax2 = axes[1]
        ns_plot = np.arange(1, 51)
        cum_e = np.cumsum([math.log(SIGMA / (SIGMA / (k + 1))) for k in range(1, 51)])
        cum_o = np.cumsum([math.log(SIGMA / (SIGMA * (1 - 1/(k+1)**2))) for k in range(1, 51)])
        ax2.plot(ns_plot, cum_e, color=COLORS[0], linewidth=2, label="Class E (diverges)")
        ax2.plot(ns_plot, cum_o, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Class Ω (converges)")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Cumulative Log-Floor Σℓᵢ", "n agents", "Σᵢ ℓᵢ")

        # (c) Composite floor decay comparison
        ax3 = axes[2]
        ax3.semilogy(ns, comp_e + 1e-300, color=COLORS[0], linewidth=2, label="Class E")
        ax3.semilogy(ns, comp_o, color=COLORS[1], linewidth=2,
                     linestyle="--", label="Class Ω")
        ax3.axhline(SIGMA / math.e, color=COLORS[2], linestyle=":", linewidth=1.2,
                     label=f"β_c = Σ/e ≈ {SIGMA/math.e:.1f}")
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Composite Floor Decay (log scale)", "n agents", "S_flat_comp")

        # (d) Spectral radius by class
        ax4 = axes[3]
        rho_e = [SIGMA / (SIGMA * (1 / (k + 1))) / SIGMA for k in range(1, 31)]
        # Actually rho = min floor / SIGMA
        rho_vals_e = []
        rho_vals_o = []
        for n in range(1, 31):
            agents_e = [make_agent(SIGMA / (k + 1)) for k in range(1, n + 1)]
            agents_o = [make_agent(SIGMA * (1 - 1 / (k + 1) ** 2)) for k in range(1, n + 1)]
            rho_vals_e.append(aggregate_floor(agents_e) / SIGMA)
            rho_vals_o.append(aggregate_floor(agents_o) / SIGMA)
        ax4.plot(range(1, 31), rho_vals_e, color=COLORS[0], linewidth=2, label="Class E")
        ax4.plot(range(1, 31), rho_vals_o, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Class Ω")
        ax4.axhline(1.0 / math.e, color=COLORS[2], linestyle=":", linewidth=1.2,
                     label="ρ_c = 1/e")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Spectral Radius ρ_E by Class", "n agents", "ρ_E = S_flat_agg/Σ")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper3_panel_2.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 3: Spectral Theory and Optimal Recruitment
# ─────────────────────────────────────────────
def make_panel_3():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 3: Spectral Gap Theory and Optimal Recruitment (Rearrangement Inequality)",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D spectral gap surface: delta = f(n, beta_min)
        ax = fig.add_subplot(141, projection="3d")
        ns3d = np.arange(1, 16)
        beta_mins = np.linspace(1, 60, 20)
        N3, B3 = np.meshgrid(ns3d, beta_mins)
        DELTA = 1.0 - B3 / SIGMA
        surf = ax.plot_surface(N3, B3, DELTA, cmap="Greens", alpha=0.85)
        apply_style_3d(ax, "(a) Spectral Gap δ = 1 − β_min/Σ",
                        "n agents", "β_min", "δ (spectral gap)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Cumulative composite floor: ascending vs descending vs random
        ax2 = axes[1]
        betas = [5.0, 10.0, 20.0, 35.0, 60.0]
        agents_all = [make_agent(b) for b in sorted(betas)]
        agents_desc = [make_agent(b) for b in sorted(betas, reverse=True)]
        agents_rand = [make_agent(b) for b in [20.0, 5.0, 60.0, 10.0, 35.0]]

        def cum_comp(agents_ordered):
            vals = []
            for k in range(1, len(agents_ordered) + 1):
                vals.append(composite_floor(agents_ordered[:k]))
            return np.cumsum(vals)

        ks = range(1, len(betas) + 1)
        ax2.plot(ks, cum_comp(agents_all), color=COLORS[0], linewidth=2,
                  marker="o", markersize=6, label="Ascending (optimal)")
        ax2.plot(ks, cum_comp(agents_desc), color=COLORS[1], linewidth=2,
                  marker="s", markersize=6, linestyle="--", label="Descending")
        ax2.plot(ks, cum_comp(agents_rand), color=COLORS[2], linewidth=2,
                  marker="^", markersize=6, linestyle=":", label="Random")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Cumulative Composite Floor by Recruitment Order",
                    "Agents recruited (k)", "Σ S_flat_comp(k)")

        # (c) Rearrangement inequality: all permutations
        ax3 = axes[2]
        from itertools import permutations
        betas_small = [5.0, 15.0, 40.0, 70.0]
        perms = list(permutations(betas_small))
        perm_totals = []
        for perm in perms:
            agents_p = [make_agent(b) for b in perm]
            total = sum(composite_floor(agents_p[:k]) for k in range(1, len(perm) + 1))
            perm_totals.append(total)
        ax3.hist(perm_totals, bins=10, color=COLORS[3], edgecolor="white", alpha=0.85)
        ax3.axvline(min(perm_totals), color=COLORS[0], linewidth=2,
                     linestyle="--", label=f"Min (ascending): {min(perm_totals):.1f}")
        ax3.axvline(max(perm_totals), color=COLORS[1], linewidth=2,
                     linestyle="--", label=f"Max (descending): {max(perm_totals):.1f}")
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) All Permutations: Cumulative Comp Floor",
                    "Total cumulative S_flat_comp", "Count")

        # (d) Spectral gap vs n for several beta compositions
        ax4 = axes[3]
        configs = {
            "All β=10": [10.0] * 10,
            "β∈[5,50]": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            "β∈[1,90]": [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 90.0],
        }
        for label, betas_cfg in configs.items():
            gaps = []
            for k in range(1, len(betas_cfg) + 1):
                agents_k = [make_agent(b) for b in betas_cfg[:k]]
                gaps.append(1.0 - aggregate_floor(agents_k) / SIGMA)
            ax4.plot(range(1, len(betas_cfg) + 1), gaps, linewidth=2, label=label)
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Spectral Gap δ vs Ensemble Size", "n agents", "δ = 1 − ρ_E")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper3_panel_3.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 4: Phase Transition and Floor-Variance Bound
# ─────────────────────────────────────────────
def make_panel_4():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 4: Phase Transition at β_c = Σ/e and Floor-Variance Efficiency Bound",
            fontsize=13, fontweight="bold", y=1.02
        )

        beta_c = SIGMA / math.e

        # (a) 3D: Log-floor surface ℓ(beta) = log(SIGMA/beta) and 1-nat threshold
        ax = fig.add_subplot(141, projection="3d")
        betas3d = np.linspace(1, 99, 60)
        ns3d = np.arange(1, 20)
        B3d, N3d = np.meshgrid(betas3d, ns3d)
        ELL = np.log(SIGMA / B3d)
        CUMELL = N3d * ELL
        surf = ax.plot_surface(B3d, N3d, CUMELL, cmap="RdYlGn_r", alpha=0.8)
        ax.plot([beta_c, beta_c], [1, 20], [0, 20 * 1.0], color="red",
                 linewidth=2, label="β_c = Σ/e")
        apply_style_3d(ax, "(a) Cumulative Log-Floor n·ℓ(β)", "β", "n", "n·ℓ(β)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) ℓ(beta) vs beta: critical point
        ax2 = axes[1]
        betas_plot = np.linspace(0.5, 99.5, 200)
        ell_vals = np.log(SIGMA / betas_plot)
        ax2.plot(betas_plot, ell_vals, color=COLORS[0], linewidth=2)
        ax2.axvline(beta_c, color=COLORS[1], linestyle="--", linewidth=1.5,
                     label=f"β_c = Σ/e ≈ {beta_c:.2f}")
        ax2.axhline(1.0, color=COLORS[2], linestyle=":", linewidth=1.2,
                     label="1 nat threshold")
        ax2.fill_betweenx([0, 5], 0, beta_c, alpha=0.1, color=COLORS[0], label="Class E region")
        ax2.fill_betweenx([0, 5], beta_c, 100, alpha=0.1, color=COLORS[1], label="Class Ω region")
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 5)
        ax2.legend(fontsize=7.5)
        apply_style(ax2, "(b) Log-Floor ℓ(β) = log(Σ/β) and Phase Transition",
                    "β (agent floor)", "ℓ(β) = log(Σ/β) [nats]")

        # (c) Floor-Variance Bound: actual vs bound
        ax3 = axes[2]
        n_bound = 5
        cvs = np.linspace(0, 0.8, 40)
        mean_f = 25.0
        actual_vals = []
        bound_vals = []
        for cv in cvs:
            if cv == 0:
                agents = [make_agent(mean_f)] * n_bound
            else:
                sigma_ln = math.sqrt(math.log(1 + cv**2))
                mu_ln = math.log(mean_f) - sigma_ln**2 / 2
                floors = np.exp(np.random.normal(mu_ln, sigma_ln, n_bound))
                floors = np.clip(floors * (mean_f / floors.mean()), 0.1, 99.9)
                agents = [make_agent(float(f)) for f in floors]
            actual_vals.append(composite_floor(agents))
            bound = ((mean_f / SIGMA) ** n_bound) * SIGMA * math.exp(-n_bound * cv**2 / 2)
            bound_vals.append(bound)
        ax3.plot(cvs, actual_vals, color=COLORS[0], linewidth=2, label="S_flat_comp (actual)")
        ax3.plot(cvs, bound_vals, color=COLORS[1], linewidth=2,
                  linestyle="--", label="Variance bound")
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Floor-Variance Efficiency Bound (n=5, β̄=25)",
                    "CV(floors)", "S_flat_comp")

        # (d) Phase transition: composite floor at n agents all at beta_c vs below vs above
        ax4 = axes[3]
        ns_plot = np.arange(1, 31)
        betas_cases = {
            f"β = β_c = {beta_c:.1f}": beta_c,
            "β = 10 (Class E)": 10.0,
            "β = 80 (Class Ω)": 80.0,
        }
        for label, b in betas_cases.items():
            comps = [composite_floor([make_agent(b)] * int(n)) for n in ns_plot]
            expected = [SIGMA * (b / SIGMA) ** n for n in ns_plot]
            ax4.semilogy(ns_plot, expected, linewidth=2, label=label)
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Composite Floor Decay by Phase (log scale)",
                    "n agents", "S_flat_comp (log scale)")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper3_panel_4.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 5: Main Heterogeneity Theorem & Information Aggregation
# ─────────────────────────────────────────────
def make_panel_5():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 5: Main Heterogeneity Theorem — Efficiency Gain from Diversity",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Composite floor as fn of (CV, n) for fixed mean
        ax = fig.add_subplot(141, projection="3d")
        cvs3d = np.linspace(0.0, 0.9, 25)
        ns3d_vals = np.arange(2, 12)
        CV3d, N3d = np.meshgrid(cvs3d, ns3d_vals)
        mean_f = 20.0
        COMP3d = np.zeros_like(CV3d)
        for i, n in enumerate(ns3d_vals):
            for j, cv in enumerate(cvs3d):
                bound = ((mean_f / SIGMA) ** n) * SIGMA * math.exp(-n * cv**2 / 2)
                COMP3d[i, j] = bound
        surf = ax.plot_surface(CV3d, N3d, COMP3d, cmap="viridis", alpha=0.85)
        apply_style_3d(ax, "(a) Composite Floor: f(CV, n)", "CV", "n agents", "S_flat_comp bound")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Efficiency gain: % reduction vs CV for multiple n
        ax2 = axes[1]
        cvs = np.linspace(0.01, 0.9, 50)
        mean_f = 20.0
        for n, color in zip([2, 4, 6, 8], COLORS[:4]):
            gains = []
            for cv in cvs:
                homo = (mean_f / SIGMA) ** n * SIGMA
                het = homo * math.exp(-n * cv**2 / 2)
                gains.append(100 * (homo - het) / homo)
            ax2.plot(cvs, gains, color=color, linewidth=2, label=f"n={n}")
        ax2.legend(fontsize=8)
        apply_style(ax2, "(b) Efficiency Gain from Heterogeneity (β̄=20)",
                    "CV(floors)", "% Reduction in S_flat_comp")

        # (c) Information aggregation rate: sum of log-floors
        ax3 = axes[2]
        # Several ensembles with different log-floor rates
        configs = {
            "Harmonic (Class E)": [SIGMA / (k + 1) for k in range(1, 25)],
            "Geometric (Class E)": [SIGMA * 0.7**k for k in range(1, 25)],
            "Near-SIGMA (Class Ω)": [SIGMA * (1 - 1/(k+1)**2) for k in range(1, 25)],
        }
        for label, betas_seq in configs.items():
            ns_seq = range(1, len(betas_seq) + 1)
            cum_ells = np.cumsum([math.log(SIGMA / b) for b in betas_seq])
            ax3.plot(list(ns_seq), cum_ells, linewidth=2, label=label)
        ax3.legend(fontsize=8)
        apply_style(ax3, "(c) Information Aggregation Rate Σᵢℓᵢ",
                    "n agents", "Cumulative log-floor Σℓᵢ [nats]")

        # (d) Stability of efficiency class under perturbation
        ax4 = axes[3]
        n_base = 20
        class_e_base = [SIGMA / (k + 2) for k in range(n_base)]
        base_sum = sum(math.log(SIGMA / b) for b in class_e_base)
        perturb_betas = np.linspace(1, 99, 40)
        perturbed_sums = []
        for p_b in perturb_betas:
            new_betas = list(class_e_base)
            new_betas[0] = p_b
            perturbed_sums.append(sum(math.log(SIGMA / b) for b in new_betas))
        ax4.plot(perturb_betas, perturbed_sums, color=COLORS[0], linewidth=2)
        ax4.axhline(base_sum, color=COLORS[1], linestyle="--",
                     label=f"Original Σℓᵢ = {base_sum:.2f}")
        ax4.axvline(SIGMA / math.e, color=COLORS[2], linestyle=":",
                     label=f"β_c = Σ/e ≈ {SIGMA/math.e:.1f}")
        ax4.legend(fontsize=8)
        apply_style(ax4, "(d) Stability: Σℓᵢ Under Single Agent Perturbation",
                    "Perturbed agent floor β", "New Σℓᵢ [nats]")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper3_panel_5.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating Paper 3 panels...")
    make_panel_1()
    make_panel_2()
    make_panel_3()
    make_panel_4()
    make_panel_5()
    print("All Paper 3 panels generated.")
