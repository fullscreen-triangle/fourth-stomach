"""
Generate 5 publication panels for:
  Market Equilibrium as Purpose Fixed-Point:
  A Mathematical Theory of Coordination among Bounded Economic Agents

Each panel: 4 charts in a row, white background, minimal text.
At least one 3D chart per panel.
All charts are data-driven from the mathematical framework.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path

OUT = Path(__file__).parent
RNG = np.random.default_rng(2025)
SIGMA = 100.0

# ── Style ─────────────────────────────────────────────────────────────────────

def fig_panel(n=4, ratio=5):
    fig = plt.figure(figsize=(4 * ratio, ratio), facecolor="white")
    fig.patch.set_facecolor("white")
    return fig

def ax2d(fig, pos):
    ax = fig.add_subplot(pos)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return ax

def ax3d(fig, pos):
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=6, pad=1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.85")
    ax.yaxis.pane.set_edgecolor("0.85")
    ax.zaxis.pane.set_edgecolor("0.85")
    ax.grid(True, lw=0.3, alpha=0.4)
    return ax

def tight(fig, name):
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")

CMAP = cm.viridis
CMAP2 = cm.plasma
CMAP3 = cm.coolwarm
LCOLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Ensemble Algebra and Catalytic Composition
# ─────────────────────────────────────────────────────────────────────────────

def panel1():
    fig = fig_panel()

    # ── 1a: 3D — Composite floor f₁⊠f₂ = f₁·f₂/Σ over (f₁, f₂) ─────────────
    ax = ax3d(fig, 141)
    F1, F2 = np.meshgrid(np.linspace(0.1, 10.0, 60), np.linspace(0.1, 10.0, 60))
    CF = F1 * F2 / SIGMA
    ax.plot_surface(F1, F2, CF, cmap=CMAP, alpha=0.88, linewidth=0)
    ax.set_xlabel("f₁", fontsize=7, labelpad=1)
    ax.set_ylabel("f₂", fontsize=7, labelpad=1)
    ax.set_zlabel("f₁⊠f₂", fontsize=7, labelpad=1)
    ax.view_init(elev=28, azim=-50)

    # ── 1b: Composite floor vs n_agents (log scale) ───────────────────────────
    ax2 = ax2d(fig, 142)
    n_max = 12
    for trial in range(8):
        floors_t = RNG.uniform(0.5, 6.0, n_max)
        cf_n = [float(np.prod(floors_t[:n])) / (SIGMA ** (n - 1))
                for n in range(1, n_max + 1)]
        ax2.semilogy(range(1, n_max + 1), cf_n,
                     color=LCOLORS[trial % 5], lw=1.2, alpha=0.65)
    ax2.set_xlabel("n  (agents)", fontsize=8)
    ax2.set_ylabel("composite floor", fontsize=8)
    ax2.set_xticks(range(1, n_max + 1))

    # ── 1c: Heatmap — S(E, x; C) for 3-agent ensemble ────────────────────────
    ax3 = ax2d(fig, 143)
    xs = np.linspace(-4, 4, 180)
    XX, YY = np.meshgrid(xs, xs)
    betas3 = [0.4, 1.0, 1.8]
    cell_r3 = 1.0
    S_E = np.full(XX.shape, np.inf)
    for b in betas3:
        d3 = np.maximum(0.0, np.sqrt(XX**2 + YY**2) - cell_r3)
        S_E = np.minimum(S_E, np.maximum(b, d3))
    im3 = ax3.imshow(S_E, extent=[-4, 4, -4, 4], origin="lower",
                     cmap=CMAP, aspect="equal")
    theta = np.linspace(0, 2 * np.pi, 300)
    ax3.plot(cell_r3 * np.cos(theta), cell_r3 * np.sin(theta),
             "w-", lw=1.0, alpha=0.7)
    fig.colorbar(im3, ax=ax3, pad=0.02, fraction=0.046,
                 label="S(E, x; C)", shrink=0.85).ax.tick_params(labelsize=6)
    ax3.set_xticks([]); ax3.set_yticks([])

    # ── 1d: ⊠ algebra — f₁⊠f₂ vs f₂⊠f₁ (commutativity) ─────────────────────
    ax4 = ax2d(fig, 144)
    n1d = 400
    f1s = RNG.uniform(0.1, 10.0, n1d)
    f2s = RNG.uniform(0.1, 10.0, n1d)
    lhs = f1s * f2s / SIGMA
    rhs = f2s * f1s / SIGMA
    ax4.scatter(lhs, rhs, s=5, c=f1s, cmap=CMAP2, alpha=0.55)
    lo4, hi4 = lhs.min(), lhs.max()
    ax4.plot([lo4, hi4], [lo4, hi4], "k--", lw=0.8, alpha=0.5)
    ax4.set_xlabel("f₁ ⊠ f₂", fontsize=8)
    ax4.set_ylabel("f₂ ⊠ f₁", fontsize=8)

    tight(fig, "paper2_panel_1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Belief Incompatibility and Common-Cell Convergence
# ─────────────────────────────────────────────────────────────────────────────

def panel2():
    fig = fig_panel()

    # ── 2a: 3D — Projection samples from 3 disjoint-K agents ─────────────────
    ax = ax3d(fig, 141)
    x0 = np.array([0.0, 0.0])
    colors_2a = [LCOLORS[0], LCOLORS[1], LCOLORS[2]]
    betas_2a = [0.5, 1.0, 1.5]
    for i, (b, col) in enumerate(zip(betas_2a, colors_2a)):
        n_proj = 120
        dirs = RNG.standard_normal((n_proj, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = b * RNG.uniform(0, 1, n_proj) ** 0.5
        pts = x0 + radii[:, None] * dirs
        # Z-coordinate = beta (layered for visibility)
        z_off = i * 0.4
        ax.scatter(pts[:, 0], pts[:, 1],
                   np.full(n_proj, z_off), s=4, color=col, alpha=0.6)
        ax.plot_surface(
            *np.meshgrid(np.linspace(-b, b, 30), np.linspace(-b, b, 30)),
            np.full((30, 30), z_off),
            alpha=0.06, color=col)
    ax.set_xlabel("x₁", fontsize=7, labelpad=1)
    ax.set_ylabel("x₂", fontsize=7, labelpad=1)
    ax.set_zlabel("agent", fontsize=7, labelpad=1)
    ax.set_zticks([0.0, 0.4, 0.8])
    ax.set_zticklabels(["A₁", "A₂", "A₃"], fontsize=6)
    ax.view_init(elev=28, azim=35)

    # ── 2b: Scatter — S₁ vs S₂ for 800 states (incompatibility) ─────────────
    ax2 = ax2d(fig, 142)
    x_pts2 = RNG.uniform(-4, 4, (800, 2))
    cell_r2 = 1.5
    b1, b2 = 0.4, 1.8
    d2 = np.maximum(0.0, np.linalg.norm(x_pts2, axis=1) - cell_r2)
    s1 = np.maximum(b1, d2)
    s2 = np.maximum(b2, d2)
    ax2.scatter(s1, s2, s=5, c=d2, cmap=CMAP, alpha=0.5)
    ax2.plot([0, 6], [0, 6], "k--", lw=0.7, alpha=0.4)
    ax2.set_xlabel("S(A₁, x; C)", fontsize=8)
    ax2.set_ylabel("S(A₂, x; C)", fontsize=8)

    # ── 2c: CCC — ensemble S achieves cell for n = 2, 5, 10, 20 ──────────────
    ax3 = ax2d(fig, 143)
    cell_r3 = 2.0
    n_sizes = [2, 5, 10, 20]
    for i, n in enumerate(n_sizes):
        betas_n = RNG.uniform(0.1, 1.5, n)
        # For x in cell: S = min_i beta_i (all < tau)
        # Distribution of S over in-cell samples
        x_in = RNG.standard_normal((200, 2))
        x_in = x_in / np.linalg.norm(x_in, axis=1, keepdims=True)
        x_in *= cell_r3 * RNG.uniform(0, 1, 200)[:, None] ** 0.5
        d_in = np.maximum(0.0, np.linalg.norm(x_in, axis=1) - cell_r3)
        s_vals = np.full(200, np.inf)
        for b in betas_n:
            s_vals = np.minimum(s_vals, np.maximum(b, d_in))
        ax3.plot(np.sort(s_vals), np.linspace(0, 1, 200),
                 color=LCOLORS[i], lw=1.4, label=f"n={n}")
    ax3.axvline(cell_r3, color="0.6", lw=0.7, ls=":")
    ax3.set_xlabel("S(E, x; C)", fontsize=8)
    ax3.set_ylabel("CDF", fontsize=8)
    ax3.legend(fontsize=6, handlelength=1.0, framealpha=0.3)

    # ── 2d: Ensemble S field — 5-agent heatmap (different β arrangement) ──────
    ax4 = ax2d(fig, 144)
    xs4 = np.linspace(-4, 4, 180)
    XX4, YY4 = np.meshgrid(xs4, xs4)
    betas_4 = [0.3, 0.6, 1.0, 1.5, 2.2]
    cell_r4 = 1.2
    S4 = np.full(XX4.shape, np.inf)
    for b in betas_4:
        d4 = np.maximum(0.0, np.sqrt(XX4**2 + YY4**2) - cell_r4)
        S4 = np.minimum(S4, np.maximum(b, d4))
    im4 = ax4.imshow(S4, extent=[-4, 4, -4, 4], origin="lower",
                     cmap=CMAP2, aspect="equal")
    theta4 = np.linspace(0, 2 * np.pi, 300)
    ax4.plot(cell_r4 * np.cos(theta4), cell_r4 * np.sin(theta4),
             "w-", lw=1.0, alpha=0.7)
    fig.colorbar(im4, ax=ax4, pad=0.02, fraction=0.046,
                 label="S(E)", shrink=0.85).ax.tick_params(labelsize=6)
    ax4.set_xticks([]); ax4.set_yticks([])

    tight(fig, "paper2_panel_2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — Reachability and Market Depth
# ─────────────────────────────────────────────────────────────────────────────

def panel3():
    fig = fig_panel()

    # ── 3a: 3D — Lower bound surface over (τ, n) ─────────────────────────────
    ax = ax3d(fig, 141)
    TAU, N = np.meshgrid(np.linspace(0.5, 4.0, 40), np.arange(1, 16))
    avg_floor = 1.2          # representative per-agent floor
    sample_r = 1.0 + 5.0    # r + 5
    r_cell = 1.0
    comp_floor_grid = (avg_floor ** N) / (SIGMA ** (N - 1))
    reach_r_grid = np.maximum(0.0, r_cell + TAU - comp_floor_grid)
    lb_grid = (reach_r_grid / sample_r) ** 2
    ax.plot_surface(N.astype(float), TAU, lb_grid,
                    cmap=CMAP, alpha=0.88, linewidth=0)
    ax.set_xlabel("n", fontsize=7, labelpad=1)
    ax.set_ylabel("τ", fontsize=7, labelpad=1)
    ax.set_zlabel("reach lb", fontsize=7, labelpad=1)
    ax.view_init(elev=32, azim=-45)

    # ── 3b: Reachability lower bound vs n for 4 τ values ─────────────────────
    ax2 = ax2d(fig, 142)
    ns_b = np.arange(1, 16)
    avg_f_b = 2.0
    r_b = 1.0
    sample_rb = r_b + 5.0
    for i, tau_v in enumerate([0.5, 1.0, 2.0, 3.5]):
        cf_b = [(avg_f_b ** n) / (SIGMA ** (n - 1)) for n in ns_b]
        lb_b = [max(0.0, (r_b + tau_v - cf) / sample_rb) ** 2 for cf in cf_b]
        ax2.plot(ns_b, lb_b, color=LCOLORS[i], lw=1.4,
                 marker="o", ms=3, label=f"τ={tau_v}")
    ax2.set_xlabel("n  (agents)", fontsize=8)
    ax2.set_ylabel("reach lb", fontsize=8)
    ax2.legend(fontsize=6, handlelength=1.0, framealpha=0.3)

    # ── 3c: Composite floor decay — measured vs lower bound ───────────────────
    ax3 = ax2d(fig, 143)
    n3_max = 14
    for trial in range(10):
        floors_t3 = RNG.uniform(0.3, 3.0, n3_max)
        cf_t = [float(np.prod(floors_t3[:n])) / (SIGMA ** (n - 1))
                for n in range(1, n3_max + 1)]
        ax3.semilogy(range(1, n3_max + 1), cf_t,
                     color=LCOLORS[trial % 5], lw=1.0, alpha=0.6)
    ax3.set_xlabel("n  (agents)", fontsize=8)
    ax3.set_ylabel("composite floor", fontsize=8)

    # ── 3d: Reachability fraction contours in (n, floor) space ───────────────
    ax4 = ax2d(fig, 144)
    ns_d = np.linspace(1, 15, 60)
    avg_floors_d = np.linspace(0.1, 5.0, 60)
    NN, FF = np.meshgrid(ns_d, avg_floors_d)
    r_d, tau_d, sample_rd = 1.0, 2.0, 6.0
    CF_D = (FF ** NN) / (SIGMA ** (NN - 1))
    reach_rd = np.maximum(0.0, r_d + tau_d - CF_D)
    lb_D = (reach_rd / sample_rd) ** 2
    cp = ax4.contourf(NN, FF, lb_D, levels=20, cmap=CMAP)
    ax4.contour(NN, FF, lb_D, levels=6, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cp, ax=ax4, pad=0.02, fraction=0.046,
                 label="reach lb", shrink=0.85).ax.tick_params(labelsize=6)
    ax4.set_xlabel("n", fontsize=8)
    ax4.set_ylabel("avg floor", fontsize=8)

    tight(fig, "paper2_panel_3.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Purpose Existence and ω-Limit Convergence
# ─────────────────────────────────────────────────────────────────────────────

def panel4():
    fig = fig_panel()

    # ── 4a: 3D — Purpose functional surface Φ_E(τ, S_flat) ──────────────────
    ax = ax3d(fig, 141)
    TAU4, SF4 = np.meshgrid(np.linspace(0.1, 4.0, 60), np.linspace(0.01, 2.0, 60))
    # Phi_E(C) = S_flat(E) when x is inside C; purpose exists when tau > S_flat
    # Purpose margin = tau - S_flat (positive = purpose exists)
    margin = TAU4 - SF4
    margin_pos = np.where(margin > 0, margin, np.nan)
    ax.plot_surface(TAU4, SF4, margin_pos, cmap=CMAP2, alpha=0.87, linewidth=0)
    ax.set_xlabel("τ", fontsize=7, labelpad=1)
    ax.set_ylabel("S̄(E)", fontsize=7, labelpad=1)
    ax.set_zlabel("τ − S̄", fontsize=7, labelpad=1)
    ax.view_init(elev=28, azim=-40)

    # ── 4b: ω-limit — gradient flow trajectories to cell ─────────────────────
    ax2 = ax2d(fig, 142)
    cell_c4 = np.array([0.0, 0.0])
    cell_r4 = 1.0
    n_traj = 20
    t_steps = 120
    step_sz = 0.12
    for _ in range(n_traj):
        x = RNG.uniform(-4, 4, 2)
        xs_t = [x[0]]; ys_t = [x[1]]
        for __ in range(t_steps):
            dx = cell_c4 - x
            n = np.linalg.norm(dx)
            if n < 1e-8:
                break
            x = x + step_sz * dx / n
            xs_t.append(x[0]); ys_t.append(x[1])
        ax2.plot(xs_t, ys_t, lw=0.8, color=LCOLORS[_ % 5], alpha=0.55)
    theta4b = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(cell_r4 * np.cos(theta4b), cell_r4 * np.sin(theta4b),
             "k-", lw=1.2)
    ax2.set_aspect("equal")
    ax2.set_xticks([]); ax2.set_yticks([])

    # ── 4c: Bid-ask spread = τ − S̄(E) vs S̄(E) for varying ensemble size ────
    ax3 = ax2d(fig, 143)
    tau_c = 2.0
    n_sizes_c = list(range(1, 16))
    for trial in range(8):
        base_f = float(RNG.uniform(0.5, 3.0))
        spreads = []
        for n in n_sizes_c:
            floors_c = [base_f + float(RNG.uniform(-0.2, 0.2)) for _ in range(n)]
            cf_c = float(np.prod(floors_c)) / (SIGMA ** (n - 1))
            spreads.append(max(0.0, tau_c - cf_c))
        ax3.plot(n_sizes_c, spreads, color=LCOLORS[trial % 5], lw=1.1, alpha=0.65)
    ax3.axhline(tau_c, color="0.5", lw=0.7, ls="--", alpha=0.5)
    ax3.set_xlabel("n  (agents)", fontsize=8)
    ax3.set_ylabel("τ − S̄(E)", fontsize=8)

    # ── 4d: Purpose existence map in (τ, S_flat) — scatter ───────────────────
    ax4 = ax2d(fig, 144)
    n_pts4 = 600
    taus4 = RNG.uniform(0.1, 4.0, n_pts4)
    sflats4 = RNG.uniform(0.01, 3.5, n_pts4)
    exists = (taus4 > sflats4).astype(float)
    ax4.scatter(taus4, sflats4, s=6, c=exists, cmap=CMAP3, alpha=0.6,
                vmin=0, vmax=1)
    t_line = np.linspace(0.01, 4.0, 200)
    ax4.plot(t_line, t_line, "k-", lw=1.0)
    ax4.set_xlabel("τ(C)", fontsize=8)
    ax4.set_ylabel("S̄(E)", fontsize=8)

    tight(fig, "paper2_panel_4.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Motivation Heterogeneity and Market Information Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def panel5():
    fig = fig_panel()

    # ── 5a: 3D — Composite floor over (n, avg_floor) ─────────────────────────
    ax = ax3d(fig, 141)
    N5, AF5 = np.meshgrid(np.arange(1, 16, dtype=float), np.linspace(0.2, 5.0, 50))
    CF5 = (AF5 ** N5) / (SIGMA ** (N5 - 1))
    CF5_clip = np.clip(CF5, 0, 20)
    ax.plot_surface(N5, AF5, CF5_clip, cmap=CMAP, alpha=0.87, linewidth=0)
    ax.set_xlabel("n", fontsize=7, labelpad=1)
    ax.set_ylabel("f̄", fontsize=7, labelpad=1)
    ax.set_zlabel("composite floor", fontsize=7, labelpad=1)
    ax.view_init(elev=32, azim=-50)

    # ── 5b: Borel-Cantelli — composite floor vs n for 6 agent configurations ──
    ax2 = ax2d(fig, 142)
    n5b_max = 50
    ns5b = np.arange(1, n5b_max + 1)
    # Config 1: homogeneous q=0.9
    q_homo = 0.9
    cf_homo = [SIGMA * q_homo ** n for n in ns5b]
    ax2.semilogy(ns5b, cf_homo, color="black", lw=1.5, ls="--",
                 label="homogeneous")
    # Config 2-6: heterogeneous
    for trial in range(5):
        qs = RNG.uniform(0.6, 0.98, n5b_max)
        cf_het = [SIGMA * float(np.prod(qs[:n])) for n in ns5b]
        ax2.semilogy(ns5b, cf_het, color=LCOLORS[trial], lw=1.0, alpha=0.65)
    ax2.set_xlabel("n  (agents)", fontsize=8)
    ax2.set_ylabel("composite floor", fontsize=8)
    ax2.legend(fontsize=6, handlelength=1.2, framealpha=0.3)

    # ── 5c: Heterogeneous vs homogeneous composite floor — scatter ────────────
    ax3 = ax2d(fig, 143)
    n_trials5c = 300
    n_ens = 15
    homo_vals, het_vals = [], []
    for _ in range(n_trials5c):
        q_h = float(RNG.uniform(0.7, 0.98))
        cf_h = SIGMA * q_h ** n_ens
        qs_het = RNG.uniform(0.5, 0.99, n_ens)
        cf_het2 = SIGMA * float(np.prod(qs_het))
        homo_vals.append(cf_h)
        het_vals.append(cf_het2)
    homo_vals = np.array(homo_vals)
    het_vals = np.array(het_vals)
    colors5c = np.log1p(homo_vals / (het_vals + 1e-30))
    ax3.scatter(homo_vals, het_vals, s=6, c=colors5c, cmap=CMAP2, alpha=0.6)
    lo5c = min(homo_vals.min(), het_vals.min())
    hi5c = max(homo_vals.max(), het_vals.max())
    ax3.plot([lo5c, hi5c], [lo5c, hi5c], "k--", lw=0.8, alpha=0.5)
    ax3.set_xlabel("homogeneous composite floor", fontsize=7)
    ax3.set_ylabel("heterogeneous composite floor", fontsize=7)

    # ── 5d: EMH — bid-ask spread vs ensemble size for multiple τ values ───────
    ax4 = ax2d(fig, 144)
    ns5d = np.arange(1, 20)
    for i, tau_v in enumerate([0.5, 1.0, 2.0, 3.5]):
        spreads5d = []
        floors5d = RNG.uniform(0.3, 2.0, 20)
        for n in ns5d:
            cf_5d = float(np.prod(floors5d[:n])) / (SIGMA ** (n - 1))
            spreads5d.append(max(0.0, tau_v - cf_5d))
        ax4.plot(ns5d, spreads5d, color=LCOLORS[i], lw=1.4,
                 label=f"τ={tau_v}")
    ax4.set_xlabel("n  (agents)", fontsize=8)
    ax4.set_ylabel("bid-ask spread", fontsize=8)
    ax4.legend(fontsize=6, handlelength=1.0, framealpha=0.3)

    tight(fig, "paper2_panel_5.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Paper 2 panels...")
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print("Done.")
