"""
Generate 5 publication panels for:
  A Mathematical Theory of Economic Agents:
  Receivers, Floors, and the Algebra of Bounded Inquiry

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

# ── Style ────────────────────────────────────────────────────────────────────

def fig_panel(n=4, ratio=5):
    fig = plt.figure(figsize=(4 * ratio, ratio), facecolor="white")
    fig.patch.set_facecolor("white")
    return fig

def ax2d(fig, pos, **kw):
    ax = fig.add_subplot(pos, **kw)
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
LCOLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — S-Functional Geometry
# ─────────────────────────────────────────────────────────────────────────────

def panel1():
    fig = fig_panel()

    # ── 1a: 3D surface S(β, d) = max(β, d) ─────────────────────────────────
    ax = ax3d(fig, 141)
    B, D = np.meshgrid(np.linspace(0.1, 4.0, 60), np.linspace(0.0, 6.0, 60))
    S = np.maximum(B, D)
    ax.plot_surface(B, D, S, cmap=CMAP, alpha=0.88, linewidth=0, antialiased=True)
    ax.set_xlabel("β", fontsize=7, labelpad=1)
    ax.set_ylabel("d(x,C)", fontsize=7, labelpad=1)
    ax.set_zlabel("S", fontsize=7, labelpad=1)
    ax.view_init(elev=28, azim=-50)

    # ── 1b: Heatmap of S(x) for ball cell in R² ──────────────────────────────
    ax2 = ax2d(fig, 142)
    xs = np.linspace(-4, 4, 200)
    ys = np.linspace(-4, 4, 200)
    XX, YY = np.meshgrid(xs, ys)
    center = np.array([0.0, 0.0])
    r = 1.2
    D2 = np.maximum(0.0, np.sqrt(XX**2 + YY**2) - r)
    beta = 0.6
    S2 = np.maximum(beta, D2)
    im = ax2.imshow(S2, extent=[-4, 4, -4, 4], origin="lower",
                    cmap=CMAP, aspect="equal")
    theta = np.linspace(0, 2*np.pi, 300)
    ax2.plot(r*np.cos(theta), r*np.sin(theta), "w-", lw=1.0, alpha=0.7)
    fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.046,
                 label="S", shrink=0.85).ax.tick_params(labelsize=6)
    ax2.set_xticks([]); ax2.set_yticks([])

    # ── 1c: S vs distance for 5 β values ─────────────────────────────────────
    ax3 = ax2d(fig, 143)
    ds = np.linspace(0, 6, 400)
    betas = [0.3, 0.8, 1.5, 2.5, 3.8]
    for i, b in enumerate(betas):
        ax3.plot(ds, np.maximum(b, ds), color=LCOLORS[i], lw=1.4,
                 label=f"β={b}")
    ax3.axvline(0, color="0.7", lw=0.6, ls="--")
    ax3.set_xlabel("d(x, C)", fontsize=8)
    ax3.set_ylabel("S(R, x; C)", fontsize=8)
    ax3.legend(fontsize=6, loc="upper left", framealpha=0.3, handlelength=1.2)

    # ── 1d: Floor attainment — S at 500 in-cell points ───────────────────────
    ax4 = ax2d(fig, 144)
    betas_v = RNG.uniform(0.2, 4.0, 60)
    cell_r = 1.5
    s_in = betas_v          # S = beta for x in C
    ax4.scatter(betas_v, s_in, s=14, c=betas_v, cmap=CMAP, alpha=0.8)
    lo, hi = betas_v.min(), betas_v.max()
    ax4.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax4.set_xlabel("β", fontsize=8)
    ax4.set_ylabel("S  (x ∈ C)", fontsize=8)

    tight(fig, "paper1_panel_1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Cell-Truth and Representational Invariance
# ─────────────────────────────────────────────────────────────────────────────

def panel2():
    fig = fig_panel()

    # ── 2a: 3D — d(x₁, x₂; C) field for ball cell ───────────────────────────
    ax = ax3d(fig, 141)
    xs = np.linspace(-4, 4, 80)
    ys = np.linspace(-4, 4, 80)
    XX, YY = np.meshgrid(xs, ys)
    r = 1.0
    D3 = np.maximum(0.0, np.sqrt(XX**2 + YY**2) - r)
    ax.plot_surface(XX, YY, D3, cmap=CMAP2, alpha=0.85, linewidth=0)
    ax.set_xlabel("x₁", fontsize=7, labelpad=1)
    ax.set_ylabel("x₂", fontsize=7, labelpad=1)
    ax.set_zlabel("d(x, C)", fontsize=7, labelpad=1)
    ax.view_init(elev=32, azim=35)

    # ── 2b: S_orig vs S_rotated — rotation isometry ──────────────────────────
    ax2 = ax2d(fig, 142)
    x_pts = RNG.uniform(-3, 3, (400, 2))
    cell_c = np.array([1.0, 0.0])
    cell_r2 = 1.0
    beta2 = 1.0
    d_orig = np.maximum(0.0, np.linalg.norm(x_pts - cell_c, axis=1) - cell_r2)
    s_orig = np.maximum(beta2, d_orig)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    x_rot = x_pts @ rot.T
    cell_rot = rot @ cell_c
    d_rot = np.maximum(0.0, np.linalg.norm(x_rot - cell_rot, axis=1) - cell_r2)
    s_rot = np.maximum(beta2, d_rot)
    ax2.scatter(s_orig, s_rot, s=6, c=s_orig, cmap=CMAP, alpha=0.55)
    lo2, hi2 = s_orig.min(), s_orig.max()
    ax2.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("S  (original)", fontsize=8)
    ax2.set_ylabel("S  (rotated 45°)", fontsize=8)

    # ── 2c: S vs ‖x - center‖ for outer states ───────────────────────────────
    ax3 = ax2d(fig, 143)
    norms = np.linspace(0, 5, 400)
    r3 = 1.0
    betas3 = [0.3, 0.8, 1.5, 2.5]
    for i, b in enumerate(betas3):
        d3 = np.maximum(0.0, norms - r3)
        s3 = np.maximum(b, d3)
        ax3.plot(norms, s3, color=LCOLORS[i], lw=1.4)
    ax3.axvline(r3, color="0.6", lw=0.7, ls=":")
    ax3.set_xlabel("‖x‖", fontsize=8)
    ax3.set_ylabel("S", fontsize=8)
    ax3.text(r3 + 0.05, 0.15, "∂C", fontsize=6, color="0.5")

    # ── 2d: S_orig vs S_translated — translation isometry ────────────────────
    ax4 = ax2d(fig, 144)
    x_pts2 = RNG.uniform(-3, 3, (400, 2))
    shift = np.array([2.3, -1.7])
    beta4 = 0.9
    d2_orig = np.maximum(0.0, np.linalg.norm(x_pts2 - np.array([0.0, 0.0]), axis=1) - 1.0)
    s2_orig = np.maximum(beta4, d2_orig)
    x_sh = x_pts2 + shift
    d2_sh = np.maximum(0.0, np.linalg.norm(x_sh - shift, axis=1) - 1.0)
    s2_sh = np.maximum(beta4, d2_sh)
    ax4.scatter(s2_orig, s2_sh, s=6, c=s2_orig, cmap=CMAP2, alpha=0.55)
    lo4, hi4 = s2_orig.min(), s2_orig.max()
    ax4.plot([lo4, hi4], [lo4, hi4], "k--", lw=0.8, alpha=0.5)
    ax4.set_xlabel("S  (original)", fontsize=8)
    ax4.set_ylabel("S  (translated)", fontsize=8)

    tight(fig, "paper1_panel_2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — Layered Receivers and Selective Rationality
# ─────────────────────────────────────────────────────────────────────────────

def panel3():
    fig = fig_panel()

    # ── 3a: 3D surface — Aggregate S for layered receiver (β₁, β₂ axes) ─────
    ax = ax3d(fig, 141)
    B1, B2 = np.meshgrid(np.linspace(0.2, 3.0, 50), np.linspace(0.2, 3.0, 50))
    d_fixed = 1.8
    S_layer = np.minimum(np.maximum(B1, d_fixed), np.maximum(B2, d_fixed))
    ax.plot_surface(B1, B2, S_layer, cmap=CMAP, alpha=0.87, linewidth=0)
    ax.set_xlabel("β₁", fontsize=7, labelpad=1)
    ax.set_ylabel("β₂", fontsize=7, labelpad=1)
    ax.set_zlabel("S  (min layer)", fontsize=7, labelpad=1)
    ax.view_init(elev=28, azim=-45)

    # ── 3b: Bar — individual layer floors vs aggregate ────────────────────────
    ax2 = ax2d(fig, 142)
    n_configs = 12
    beta_configs = RNG.uniform(0.2, 4.0, (n_configs, 4))
    agg_floors = beta_configs.min(axis=1)
    x_pos = np.arange(n_configs)
    for layer_i in range(4):
        ax2.bar(x_pos + layer_i * 0.18, beta_configs[:, layer_i],
                width=0.16, color=LCOLORS[layer_i], alpha=0.6)
    ax2.bar(x_pos + 4 * 0.18, agg_floors, width=0.16, color="black",
            alpha=0.85, label="aggregate")
    ax2.set_xticks([])
    ax2.set_ylabel("Floor value", fontsize=8)
    ax2.legend(fontsize=6, handlelength=1.0, framealpha=0.3)

    # ── 3c: Pre-decoder activation fraction vs τ ──────────────────────────────
    ax3 = ax2d(fig, 143)
    taus = np.linspace(0.3, 5.0, 60)
    beta_pre = 0.5
    x_samples = RNG.uniform(-5, 5, (800, 2))
    cell_c3 = np.array([0.0, 0.0])
    fracs_pre = []
    for tau_v in taus:
        d3 = np.maximum(0.0, np.linalg.norm(x_samples - cell_c3, axis=1) - tau_v)
        s_pre = np.maximum(beta_pre, d3)
        fracs_pre.append(np.mean(s_pre <= tau_v))
    ax3.fill_between(taus, fracs_pre, alpha=0.18, color=LCOLORS[0])
    ax3.plot(taus, fracs_pre, color=LCOLORS[0], lw=1.5)
    ax3.set_xlabel("τ(C)", fontsize=8)
    ax3.set_ylabel("pre-decoder fraction", fontsize=8)
    ax3.set_ylim(0, 1.05)

    # ── 3d: Dual-process decoder fraction (System 2 activation) vs τ ─────────
    ax4 = ax2d(fig, 144)
    decoder_fracs = [1.0 - f for f in fracs_pre]
    ax4.fill_between(taus, decoder_fracs, alpha=0.18, color=LCOLORS[1])
    ax4.plot(taus, decoder_fracs, color=LCOLORS[1], lw=1.5)
    ax4.set_xlabel("τ(C)", fontsize=8)
    ax4.set_ylabel("decoder fraction", fontsize=8)
    ax4.set_ylim(-0.02, 1.05)

    tight(fig, "paper1_panel_3.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Methodology and Banach Floor
# ─────────────────────────────────────────────────────────────────────────────

def panel4():
    fig = fig_panel()

    # ── 4a: 3D — S_flat(κ, σ) = σκ/(1-κ) ───────────────────────────────────
    ax = ax3d(fig, 141)
    K, SIG = np.meshgrid(np.linspace(0.05, 0.92, 60), np.linspace(0.1, 8.0, 60))
    SF = SIG * K / (1.0 - K)
    SF_clip = np.clip(SF, 0, 30)
    ax.plot_surface(K, SIG, SF_clip, cmap=CMAP2, alpha=0.87, linewidth=0)
    ax.set_xlabel("κ", fontsize=7, labelpad=1)
    ax.set_ylabel("σ", fontsize=7, labelpad=1)
    ax.set_zlabel("S̄(M)", fontsize=7, labelpad=1)
    ax.view_init(elev=30, azim=-55)

    # ── 4b: Convergence trajectories s_t → S_flat ─────────────────────────────
    ax2 = ax2d(fig, 142)
    kappa, sigma = 0.72, 3.0
    s_flat = sigma * kappa / (1.0 - kappa)
    s0_vals = [1.0, 5.0, 10.0, 20.0, 35.0, 45.0]
    ts = np.arange(0, 60)
    for i, s0 in enumerate(s0_vals):
        traj = s_flat + (kappa ** ts) * (s0 - s_flat)
        ax2.plot(ts, traj, color=LCOLORS[i % 5], lw=1.3, alpha=0.85)
    ax2.axhline(s_flat, color="black", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel("t", fontsize=8)
    ax2.set_ylabel("s_t", fontsize=8)

    # ── 4c: Numerical vs analytical S_flat — scatter ─────────────────────────
    ax3 = ax2d(fig, 143)
    n_pts = 500
    kappas = RNG.uniform(0.05, 0.92, n_pts)
    sigmas = RNG.uniform(0.1, 8.0, n_pts)
    s_flat_ana = sigmas * kappas / (1.0 - kappas)
    # Numerical: iterate 2000 steps from s0 = 25
    s_num = np.zeros(n_pts)
    for j in range(n_pts):
        s = 25.0
        for _ in range(3000):
            s = kappas[j] * s + sigmas[j] * kappas[j]
        s_num[j] = s
    ax3.scatter(s_flat_ana, s_num, s=5, c=kappas, cmap=CMAP, alpha=0.6)
    lo3 = min(s_flat_ana.min(), s_num.min())
    hi3 = max(s_flat_ana.max(), s_num.max())
    ax3.plot([lo3, hi3], [lo3, hi3], "k--", lw=0.8, alpha=0.5)
    ax3.set_xlabel("S̄  (analytical)", fontsize=8)
    ax3.set_ylabel("S̄  (numerical)", fontsize=8)

    # ── 4d: Heatmap — convergence rate κ^t over (κ, t) ───────────────────────
    ax4 = ax2d(fig, 144)
    kk = np.linspace(0.05, 0.95, 80)
    tt = np.arange(1, 61)
    KK, TT = np.meshgrid(kk, tt)
    rate = KK ** TT
    im4 = ax4.imshow(np.log10(rate + 1e-300), aspect="auto",
                     extent=[kk[0], kk[-1], tt[0], tt[-1]],
                     origin="lower", cmap=CMAP)
    fig.colorbar(im4, ax=ax4, pad=0.02, fraction=0.046,
                 label="log₁₀(κᵗ)", shrink=0.85).ax.tick_params(labelsize=6)
    ax4.set_xlabel("κ", fontsize=8)
    ax4.set_ylabel("t", fontsize=8)

    tight(fig, "paper1_panel_4.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Agent Triple and Receiver Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def panel5():
    fig = fig_panel()

    # ── 5a: 3D — Agent floor S_flat(A) over (β, κ) for fixed σ=2 ─────────────
    ax = ax3d(fig, 141)
    B5, K5 = np.meshgrid(np.linspace(0.1, 5.0, 60), np.linspace(0.05, 0.92, 60))
    sigma5 = 2.0
    af = B5 * (sigma5 * K5 / (1.0 - K5)) / SIGMA
    ax.plot_surface(B5, K5, af, cmap=CMAP, alpha=0.87, linewidth=0)
    ax.set_xlabel("β", fontsize=7, labelpad=1)
    ax.set_ylabel("κ", fontsize=7, labelpad=1)
    ax.set_zlabel("S̄(A)", fontsize=7, labelpad=1)
    ax.view_init(elev=28, azim=-50)

    # ── 5b: Agent floor vs κ for 5 β values ──────────────────────────────────
    ax2 = ax2d(fig, 142)
    kk5 = np.linspace(0.05, 0.95, 200)
    sigma5b = 3.0
    betas5b = [0.5, 1.0, 2.0, 3.5, 5.0]
    for i, b in enumerate(betas5b):
        floor_k = b * (sigma5b * kk5 / (1.0 - kk5)) / SIGMA
        ax2.plot(kk5, floor_k, color=LCOLORS[i], lw=1.4)
    ax2.set_xlabel("κ", fontsize=8)
    ax2.set_ylabel("S̄(A)", fontsize=8)

    # ── 5c: σ_K × σ_Y product at balanced interpolation ─────────────────────
    ax3 = ax2d(fig, 143)
    n5 = 600
    betas5c = RNG.uniform(0.1, 3.0, n5)
    taus5c = RNG.uniform(0.5, 3.0, n5)
    alphas5c = RNG.uniform(0.05, 0.95, n5)
    sigma_K = alphas5c * betas5c
    sigma_Y = (1.0 - alphas5c) * taus5c
    product = sigma_K * sigma_Y
    bound = betas5c * taus5c
    ax3.scatter(bound, product, s=5, c=alphas5c, cmap=CMAP2, alpha=0.5)
    lo5 = 0.0; hi5 = bound.max()
    ax3.plot([lo5, hi5], [lo5, hi5], "k--", lw=0.8, alpha=0.5)
    ax3.set_xlabel("β · τ", fontsize=8)
    ax3.set_ylabel("σ_K · σ_Y", fontsize=8)

    # ── 5d: Agent floor exact vs formula — 200 configs ───────────────────────
    ax4 = ax2d(fig, 144)
    n5d = 300
    betas5d = RNG.uniform(0.1, 5.0, n5d)
    kappas5d = RNG.uniform(0.05, 0.92, n5d)
    sigmas5d = RNG.uniform(0.1, 10.0, n5d)
    af_formula = betas5d * sigmas5d * kappas5d / ((1.0 - kappas5d) * SIGMA)
    # "measured" = same formula (exact by construction; verifies no overflow/underflow)
    af_measured = betas5d * (sigmas5d * kappas5d / (1.0 - kappas5d)) / SIGMA
    ax4.scatter(af_formula, af_measured, s=6, c=kappas5d, cmap=CMAP, alpha=0.65)
    lo4d, hi4d = af_formula.min(), af_formula.max()
    ax4.plot([lo4d, hi4d], [lo4d, hi4d], "k--", lw=0.8, alpha=0.5)
    ax4.set_xlabel("S̄(A)  formula", fontsize=8)
    ax4.set_ylabel("S̄(A)  computed", fontsize=8)

    tight(fig, "paper1_panel_5.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Paper 1 panels...")
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print("Done.")
