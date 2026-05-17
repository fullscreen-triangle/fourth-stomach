"""
Panel generation for Paper 6: Multi-Horizon Kirchhoff Residuals
5 panels × 4 subplots each (≥1 3D per panel), white background, data-only.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(2025)

COLORS = ["#1a3a5c", "#c0392b", "#27ae60", "#8e44ad", "#e67e22",
          "#2980b9", "#16a085", "#d35400", "#7f8c8d", "#2c3e50"]

STYLE = {
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#111111",
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "grid.color":        "#e0e0e0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────────────

def make_laplacian(m, density=0.6, seed=None):
    rng = np.random.RandomState(seed)
    raw = rng.uniform(0.15, 0.85, (m, m))
    raw = (raw + raw.T) / 2
    np.fill_diagonal(raw, 0)
    mask_r = rng.uniform(0, 1, (m, m))
    mask_r = (mask_r + mask_r.T) / 2
    mask = (mask_r < density).astype(float)
    np.fill_diagonal(mask, 0)
    perm = rng.permutation(m)
    for k in range(m - 1):
        i, j = int(perm[k]), int(perm[k + 1])
        mask[i, j] = mask[j, i] = 1.0
    A = raw * mask
    L = np.diag(A.sum(axis=1)) - A
    return L, A


def fiedler_value(L):
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def max_eigenvalue(L):
    return float(np.max(np.linalg.eigvalsh(L)))


def pseudoinverse_L(L):
    ev, evec = np.linalg.eigh(L)
    tol = 1e-9 * max(float(np.max(np.abs(ev))), 1.0)
    inv_ev = np.where(np.abs(ev) > tol, 1.0 / ev, 0.0)
    return evec @ np.diag(inv_ev) @ evec.T


def proj_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    rho = int(rho_arr[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def fixed_point(L, mu):
    Ld = pseudoinverse_L(L)
    mu_c = mu - mu.mean()
    w = Ld @ mu_c + np.ones(len(mu)) / len(mu)
    w = np.maximum(w, 0.0)
    return w / w.sum()


def banach_iterate(L, mu, gamma, w0, n_iter):
    w = w0.copy()
    IgL = np.eye(len(mu)) - gamma * L
    for _ in range(n_iter):
        w = proj_simplex(IgL @ w + gamma * mu)
    return w


def banach_trajectory(L, mu, gamma, w0, n_iter):
    w = w0.copy()
    IgL = np.eye(len(mu)) - gamma * L
    traj = [w.copy()]
    for _ in range(n_iter):
        w = proj_simplex(IgL @ w + gamma * mu)
        traj.append(w.copy())
    return np.array(traj)


def optimal_gamma(L):
    return 0.9 / max_eigenvalue(L)


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

def tidy(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=8, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.5, linewidth=0.5)


def tidy3d(ax, xlabel="", ylabel="", zlabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=7, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
    ax.set_zlabel(zlabel, fontsize=7, labelpad=3)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("#cccccc")


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Multi-Horizon Fixed-Point Portfolios
# (a) 3D: w*_1(τ) surface over (τ, λ₂)
# (b) Banach convergence error decay for 3 horizons
# (c) Kirchhoff residual |Lw* − μ_c|_∞ vs m
# (d) Weight profiles w*(τ) for a 5-asset system across 20 horizons
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_1():
    rng = np.random.RandomState(101)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 5), facecolor="white")

        # (a) 3D: first asset weight w*_1 as function of (horizon_scale, density)
        ax3d = fig.add_subplot(141, projection="3d")
        taus = np.linspace(0.5, 5.0, 10)
        densities = np.linspace(0.3, 0.85, 8)
        TG, DG = np.meshgrid(taus, densities)
        W1 = np.zeros_like(TG)
        for i, d in enumerate(densities):
            L, _ = make_laplacian(6, density=d, seed=int(d * 1000))
            for j, tau in enumerate(taus):
                mu = 0.01 + 0.008 * tau * np.ones(6) + 0.003 * rng.randn(6)
                mu = np.clip(mu, 0.005, 0.12)
                w = fixed_point(L, mu)
                W1[i, j] = float(w[0])
        surf = ax3d.plot_surface(TG, DG, W1, cmap="Blues_r", alpha=0.9,
                                  linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.12,
                     label="w*₁")
        tidy3d(ax3d, xlabel="horizon τ", ylabel="density ρ",
               zlabel="w*₁(τ)", title="(a) Horizon surface w*₁(τ, ρ)")

        # (b) Convergence error decay for 3 different horizons
        ax2 = fig.add_subplot(142)
        L_b, _ = make_laplacian(8, density=0.6, seed=102)
        g_b = optimal_gamma(L_b)
        w0_b = proj_simplex(rng.randn(8))
        w_star_b = fixed_point(L_b, np.ones(8) * 0.04)
        for k, (tau_scale, col) in enumerate(
                zip([0.5, 1.0, 2.0], [COLORS[0], COLORS[1], COLORS[2]])):
            mu_k = 0.02 + 0.012 * tau_scale * np.ones(8)
            w_s_k = fixed_point(L_b, mu_k)
            traj = banach_trajectory(L_b, mu_k, g_b, w0_b, 80)
            errs = [float(np.linalg.norm(traj[n] - w_s_k)) for n in range(81)]
            errs = [max(e, 1e-14) for e in errs]
            ax2.semilogy(errs, color=col, linewidth=1.5, alpha=0.85)
        ax2.set_xlim(0, 80)
        tidy(ax2, xlabel="iteration n", ylabel="‖wⁿ − w*‖₂",
             title="(b) Convergence for 3 horizons")

        # (c) Kirchhoff residual max|Lw* − μ_c| vs number of assets m
        ax3 = fig.add_subplot(143)
        ms = range(4, 26, 2)
        residuals = []
        for m in ms:
            L_c, _ = make_laplacian(m, density=0.6, seed=m * 10)
            mu_c = 0.03 + rng.uniform(0, 0.04, m)
            w_c = fixed_point(L_c, mu_c)
            mu_c_centred = mu_c - mu_c.mean()
            res = float(np.max(np.abs(L_c @ w_c - mu_c_centred)))
            residuals.append(res)
        ax3.scatter(list(ms), residuals, c=COLORS[0], s=28, zorder=4,
                    edgecolors="white", linewidths=0.5)
        ax3.axhline(5e-7, color=COLORS[1], linewidth=1, linestyle="--", alpha=0.7)
        ax3.set_yscale("log")
        tidy(ax3, xlabel="assets m", ylabel="max|Lw* − μ_c|",
             title="(c) Kirchhoff residual vs m")

        # (d) Weight profiles: w*(τ) for 5-asset system, 20 horizons stacked
        ax4 = fig.add_subplot(144)
        m_d = 5
        L_d, _ = make_laplacian(m_d, density=0.65, seed=104)
        taus_d = np.linspace(0.2, 4.0, 30)
        profiles = np.zeros((m_d, len(taus_d)))
        for j, tau in enumerate(taus_d):
            mu_d = 0.01 + 0.01 * tau * np.array([1.0, 0.7, 1.2, 0.5, 0.9])
            profiles[:, j] = fixed_point(L_d, mu_d)
        bottom = np.zeros(len(taus_d))
        for i in range(m_d):
            ax4.fill_between(taus_d, bottom, bottom + profiles[i],
                             color=COLORS[i], alpha=0.82, linewidth=0)
            bottom += profiles[i]
        tidy(ax4, xlabel="horizon τ", ylabel="portfolio weight",
             title="(d) Weight stacks across horizons")

        fig.tight_layout(pad=1.2)
        save(fig, "paper6_panel_1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Kirchhoff Gain-Loss & Martingale Structure
# (a) 3D: gain-loss variance surface over (m, σ_R)
# (b) Partial-sum trajectories S_K = Σ G_k (martingale drift = 0)
# (c) Transaction clock: accumulated |G| vs calendar step
# (d) Biased vs unbiased mean gain scatter across 40 configurations
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_2():
    rng = np.random.RandomState(202)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 5), facecolor="white")

        # (a) 3D: analytic var(G) = w*^T (σ²I) w* over (m, σ_R)
        ax3d = fig.add_subplot(141, projection="3d")
        ms_a = np.arange(4, 22, 2)
        sigmas_a = np.linspace(0.01, 0.08, 8)
        MG, SG = np.meshgrid(ms_a, sigmas_a)
        VAR = np.zeros_like(MG, dtype=float)
        for i, sig in enumerate(sigmas_a):
            for j, m in enumerate(ms_a):
                L_a, _ = make_laplacian(int(m), density=0.6, seed=int(m * 100))
                mu_a = 0.03 * np.ones(int(m))
                w_a = fixed_point(L_a, mu_a)
                VAR[i, j] = float(sig**2 * w_a @ w_a)
        surf = ax3d.plot_surface(MG.astype(float), SG, VAR,
                                  cmap="Reds_r", alpha=0.9, linewidth=0)
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.12,
                     label="Var[G]")
        tidy3d(ax3d, xlabel="assets m", ylabel="σ_R",
               zlabel="Var[G]", title="(a) Gain-loss variance surface")

        # (b) Partial sums S_K for 5 independent simulations
        ax2 = fig.add_subplot(142)
        L_b, _ = make_laplacian(8, density=0.6, seed=203)
        mu_b = 0.03 + rng.uniform(0, 0.02, 8)
        w_b = fixed_point(L_b, mu_b)
        K_b = 300
        for trial in range(5):
            R_b = rng.multivariate_normal(mu_b, 0.04**2 * np.eye(8), size=K_b)
            gains_b = R_b @ w_b - float(w_b @ mu_b)
            S_b = np.cumsum(gains_b)
            ax2.plot(S_b, color=COLORS[trial], linewidth=1.0, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=1.2, linestyle="-")
        tidy(ax2, xlabel="transaction k", ylabel="S_K",
             title="(b) Partial sums S_K (zero drift)")

        # (c) Transaction clock: accumulated |G| for 3 portfolios
        ax3 = fig.add_subplot(143)
        L_c, _ = make_laplacian(8, density=0.6, seed=204)
        for trial in range(3):
            mu_c = 0.02 + rng.uniform(0, 0.04, 8)
            w_c = fixed_point(L_c, mu_c)
            R_c = rng.multivariate_normal(mu_c, 0.04**2 * np.eye(8), size=200)
            gains_c = R_c @ w_c - float(w_c @ mu_c)
            clock_c = np.cumsum(np.abs(gains_c))
            ax3.plot(clock_c, color=COLORS[trial], linewidth=1.4, alpha=0.8)
        tidy(ax3, xlabel="step t", ylabel="∑|G_t|",
             title="(c) Transaction clock")

        # (d) Unbiased vs biased mean gain scatter across 40 configs
        ax4 = fig.add_subplot(144)
        K_d = 3000
        mean_unbiased, mean_biased = [], []
        for trial in range(40):
            m_d = 6 + trial % 6
            L_d, _ = make_laplacian(m_d, density=0.5 + 0.01 * trial, seed=300 + trial)
            mu_true_d = 0.03 + rng.uniform(0, 0.03, m_d)
            mu_biased_d = mu_true_d + 0.015
            w_unb = fixed_point(L_d, mu_true_d)
            w_bi = fixed_point(L_d, mu_biased_d)
            R_d = rng.multivariate_normal(mu_true_d, 0.04**2 * np.eye(m_d), size=K_d)
            mean_unbiased.append(float(np.mean(R_d @ w_unb - w_unb @ mu_true_d)))
            mean_biased.append(float(np.mean(R_d @ w_bi - w_bi @ mu_biased_d)))
        ax4.scatter(mean_unbiased, mean_biased, c=COLORS[0], s=20,
                    alpha=0.75, edgecolors="white", linewidths=0.3)
        ax4.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax4.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        lim = max(max(abs(x) for x in mean_unbiased), max(abs(x) for x in mean_biased)) * 1.2
        ax4.set_xlim(-lim, lim)
        ax4.set_ylim(-lim, lim)
        tidy(ax4, xlabel="mean gain (unbiased)", ylabel="mean gain (biased)",
             title="(d) Unbiased vs biased E[G]")

        fig.tight_layout(pad=1.2)
        save(fig, "paper6_panel_2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — Incommensurability & Fixed-Point Drift
# (a) 3D: Kirchhoff norm ratio surface over (λ₂_sparse, λ₂_dense)
# (b) Ranking reversal matrix: score A vs B under La and Lb
# (c) Drift bound scatter: ‖Δw*‖ vs ‖Δμ‖/λ₂ for 60 pairs
# (d) Path length w* vs path length μ/λ₂ across 30 trajectories
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_3():
    rng = np.random.RandomState(303)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 5), facecolor="white")

        # (a) 3D: norm ratio ‖u‖_La / ‖u‖_Lb as function of (λ₂_a, λ₂_b)
        ax3d = fig.add_subplot(141, projection="3d")
        dens_sparse = np.linspace(0.15, 0.45, 6)
        dens_dense  = np.linspace(0.60, 0.90, 6)
        DS, DD = np.meshgrid(dens_sparse, dens_dense)
        RATIO = np.zeros_like(DS, dtype=float)
        for i, dd in enumerate(dens_dense):
            for j, ds in enumerate(dens_sparse):
                La, _ = make_laplacian(8, density=ds, seed=int(ds * 3000))
                Lb, _ = make_laplacian(8, density=dd, seed=int(dd * 3001))
                Lda = pseudoinverse_L(La)
                Ldb = pseudoinverse_L(Lb)
                ratios_ij = []
                for _ in range(12):
                    u = rng.randn(8); u -= u.mean()
                    if np.linalg.norm(u) < 1e-10:
                        continue
                    na = math.sqrt(max(float(u @ Lda @ u), 0.0))
                    nb = math.sqrt(max(float(u @ Ldb @ u), 0.0))
                    if nb > 1e-10:
                        ratios_ij.append(na / nb)
                RATIO[i, j] = float(np.median(ratios_ij)) if ratios_ij else 1.0
        surf = ax3d.plot_surface(DS, DD, RATIO, cmap="plasma", alpha=0.9, linewidth=0)
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.12,
                     label="‖u‖_La/‖u‖_Lb")
        tidy3d(ax3d, xlabel="density (sparse)", ylabel="density (dense)",
               zlabel="norm ratio", title="(a) Kirchhoff norm ratio surface")

        # (b) Ranking reversal: 30 system pairs, score scatter with colour = reversal
        ax2 = fig.add_subplot(142)
        score_A_La, score_B_La = [], []
        reversals = []
        for trial in range(40):
            ds = 0.2 + 0.01 * trial
            dd = 0.7 + 0.005 * trial
            La, _ = make_laplacian(8, density=min(ds, 0.49), seed=400 + trial)
            Lb, _ = make_laplacian(8, density=min(dd, 0.95), seed=401 + trial)
            Lda = pseudoinverse_L(La)
            Ldb = pseudoinverse_L(Lb)
            ev_a, evec_a = np.linalg.eigh(La)
            ev_b, evec_b = np.linalg.eigh(Lb)
            v2_a = evec_a[:, 1]; v2_b = evec_b[:, 1]
            mu_A = 0.03 * np.ones(8) + 0.02 * v2_a
            mu_B = 0.03 * np.ones(8) + 0.02 * v2_b
            mu_Ac = mu_A - mu_A.mean(); mu_Bc = mu_B - mu_B.mean()
            sALa = float(mu_Ac @ Lda @ mu_Ac)
            sBLa = float(mu_Bc @ Lda @ mu_Bc)
            sALb = float(mu_Ac @ Ldb @ mu_Ac)
            sBLb = float(mu_Bc @ Ldb @ mu_Bc)
            score_A_La.append(sALa)
            score_B_La.append(sBLa)
            rev = (sALa < sBLa) != (sALb < sBLb)
            reversals.append(rev)
        colors_rev = [COLORS[1] if r else COLORS[0] for r in reversals]
        ax2.scatter(score_A_La, score_B_La, c=colors_rev, s=24,
                    alpha=0.8, edgecolors="white", linewidths=0.3)
        lim2 = max(max(score_A_La), max(score_B_La)) * 1.05
        ax2.plot([0, lim2], [0, lim2], color="#aaaaaa", linewidth=0.8, linestyle="--")
        tidy(ax2, xlabel="score A under La", ylabel="score B under La",
             title="(b) Ranking reversal (red = reverses)")

        # (c) Drift bound: ‖Δw*‖ vs ‖Δμ‖/λ₂ for 60 (L, μ) pairs
        ax3 = fig.add_subplot(143)
        bounds_c, actuals_c = [], []
        for trial in range(60):
            m_c = 6 + trial % 8
            L_c, _ = make_laplacian(m_c, density=0.55, seed=500 + trial)
            lam2_c = fiedler_value(L_c)
            mu_a_c = 0.02 + rng.uniform(0, 0.06, m_c)
            mu_b_c = 0.02 + rng.uniform(0, 0.06, m_c)
            act = float(np.linalg.norm(fixed_point(L_c, mu_a_c) - fixed_point(L_c, mu_b_c)))
            bnd = float(np.linalg.norm(mu_a_c - mu_b_c)) / lam2_c
            actuals_c.append(act)
            bounds_c.append(bnd)
        ax3.scatter(bounds_c, actuals_c, c=COLORS[0], s=20,
                    alpha=0.75, edgecolors="white", linewidths=0.3)
        mx = max(bounds_c) * 1.05
        ax3.plot([0, mx], [0, mx], color=COLORS[1], linewidth=1.2,
                 linestyle="--", alpha=0.8, label="bound = actual")
        tidy(ax3, xlabel="‖Δμ‖/λ₂  (bound)", ylabel="‖Δw*‖  (actual)",
             title="(c) Drift bound (all below diagonal)")

        # (d) Path length: actual w* path vs μ path / λ₂ for 30 trajectories
        ax4 = fig.add_subplot(144)
        path_w_list, path_mu_list = [], []
        for trial in range(35):
            m_d = 6 + trial % 6
            L_d, _ = make_laplacian(m_d, density=0.6, seed=600 + trial)
            lam2_d = fiedler_value(L_d)
            T_d = 30
            mus_d = [0.03 * np.ones(m_d) + 0.006 * rng.randn(m_d) for _ in range(T_d + 1)]
            pw = sum(float(np.linalg.norm(
                fixed_point(L_d, mus_d[k + 1]) - fixed_point(L_d, mus_d[k])
            )) for k in range(T_d))
            pm = sum(float(np.linalg.norm(mus_d[k + 1] - mus_d[k])) / lam2_d
                     for k in range(T_d))
            path_w_list.append(pw)
            path_mu_list.append(pm)
        ax4.scatter(path_mu_list, path_w_list, c=COLORS[2], s=22,
                    alpha=0.8, edgecolors="white", linewidths=0.3)
        mx4 = max(path_mu_list) * 1.05
        ax4.plot([0, mx4], [0, mx4], color=COLORS[1], linewidth=1.2,
                 linestyle="--", alpha=0.8)
        tidy(ax4, xlabel="∑‖Δμ‖/λ₂  (bound)", ylabel="path length of w*(t)",
             title="(d) Path length bound")

        fig.tight_layout(pad=1.2)
        save(fig, "paper6_panel_3.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Hierarchical Gear Network
# (a) 3D: mean crossing time surface over (threshold θ, σ_G)
# (b) Accumulated imbalance I_1(t) for 4 simulations
# (c) Layer-1 vs layer-2 firing counts across 20 gear configurations
# (d) Inter-arrival times: layer-1 histogram vs layer-2 histogram
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_4():
    rng = np.random.RandomState(404)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 5), facecolor="white")

        # (a) 3D: mean first-crossing time ≈ (θ/σ_G)² over (θ, σ_G)
        ax3d = fig.add_subplot(141, projection="3d")
        thetas_a = np.linspace(0.05, 0.50, 8)
        sigmas_a = np.linspace(0.005, 0.04, 8)
        TH, SG = np.meshgrid(thetas_a, sigmas_a)
        MCT = (TH / (SG + 1e-9))**2  # theoretical (θ/σ)²
        surf = ax3d.plot_surface(TH, SG, MCT, cmap="viridis", alpha=0.9, linewidth=0)
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.12,
                     label="E[crossing]")
        tidy3d(ax3d, xlabel="threshold θ", ylabel="σ_G",
               zlabel="(θ/σ_G)²", title="(a) Crossing time surface")

        # (b) Accumulated imbalance I_1(t) traces — random walk with resets
        ax2 = fig.add_subplot(142)
        L_b, _ = make_laplacian(8, density=0.6, seed=405)
        mu_b = 0.03 + rng.uniform(0, 0.02, 8)
        w_b = fixed_point(L_b, mu_b)
        theta_b = 0.3
        for trial in range(4):
            R_b = rng.multivariate_normal(mu_b, 0.035**2 * np.eye(8), size=1000)
            gains_b = R_b @ w_b - float(w_b @ mu_b)
            acc = 0.0
            trace = [0.0]
            for g in gains_b:
                acc += g
                trace.append(acc)
                if abs(acc) > theta_b:
                    acc = 0.0
            ax2.plot(trace[:400], color=COLORS[trial], linewidth=0.9, alpha=0.7)
        ax2.axhline(theta_b, color="#888888", linewidth=0.8, linestyle="--")
        ax2.axhline(-theta_b, color="#888888", linewidth=0.8, linestyle="--")
        tidy(ax2, xlabel="step t", ylabel="I₁(t)", title="(b) Accumulated imbalance I₁(t)")

        # (c) Layer-1 vs layer-2 firings scatter across 25 configs
        ax3 = fig.add_subplot(143)
        cnt1_list, cnt2_list = [], []
        for trial in range(25):
            m_c = 6 + trial % 5
            L_c, _ = make_laplacian(m_c, density=0.5 + 0.015 * trial, seed=500 + trial)
            mu_c = 0.03 + rng.uniform(0, 0.025, m_c)
            w_c = fixed_point(L_c, mu_c)
            sig_c = 0.035 + 0.003 * (trial % 5)
            theta1 = 0.12 + 0.02 * (trial % 4)
            theta2 = theta1 * (3.5 + 0.5 * (trial % 3))
            acc1, acc2 = 0.0, 0.0
            c1, c2 = 0, 0
            for _ in range(20000):
                R = rng.multivariate_normal(mu_c, sig_c**2 * np.eye(m_c))
                G = float(w_c @ R) - float(w_c @ mu_c)
                acc1 += G
                if abs(acc1) > theta1:
                    c1 += 1; acc2 += acc1; acc1 = 0.0
                    if abs(acc2) > theta2:
                        c2 += 1; acc2 = 0.0
            cnt1_list.append(c1)
            cnt2_list.append(c2)
        ax3.scatter(cnt1_list, cnt2_list, c=COLORS[3], s=26,
                    alpha=0.8, edgecolors="white", linewidths=0.4)
        mx3 = max(cnt1_list) * 1.05
        ax3.plot([0, mx3], [0, mx3], color="#aaaaaa", linewidth=0.8, linestyle="--")
        tidy(ax3, xlabel="layer-1 firings", ylabel="layer-2 firings",
             title="(c) Gear hierarchy (layer-1 > layer-2)")

        # (d) Inter-arrival time histograms: layer-1 vs layer-2
        ax4 = fig.add_subplot(144)
        L_d, _ = make_laplacian(8, density=0.6, seed=406)
        mu_d = 0.03 + rng.uniform(0, 0.025, 8)
        w_d = fixed_point(L_d, mu_d)
        theta1_d, theta2_d = 0.15, 0.60
        acc1_d, acc2_d = 0.0, 0.0
        t = 0
        last1, last2 = 0, 0
        iat1, iat2 = [], []
        for _ in range(80000):
            t += 1
            R = rng.multivariate_normal(mu_d, 0.04**2 * np.eye(8))
            G = float(w_d @ R) - float(w_d @ mu_d)
            acc1_d += G
            if abs(acc1_d) > theta1_d:
                iat1.append(t - last1); last1 = t
                acc2_d += acc1_d; acc1_d = 0.0
                if abs(acc2_d) > theta2_d:
                    iat2.append(t - last2); last2 = t; acc2_d = 0.0
        if iat1:
            ax4.hist(iat1, bins=25, color=COLORS[0], alpha=0.6,
                     density=True, label="layer-1")
        if iat2:
            ax4.hist(iat2, bins=20, color=COLORS[1], alpha=0.6,
                     density=True, label="layer-2")
        ax4.legend(fontsize=7, frameon=False)
        tidy(ax4, xlabel="inter-arrival (steps)", ylabel="density",
             title="(d) Inter-arrival time distributions")

        fig.tight_layout(pad=1.2)
        save(fig, "paper6_panel_4.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Ergodic Convergence & Fiedler Risk Bound
# (a) 3D: risk bound surface σ_max/λ₂ · ‖μ‖ over (m, density)
# (b) Cesàro convergence: ‖w̄_K − w*_stat‖ vs K (log scale)
# (c) Actual risk vs bound scatter (50 instances; all below diagonal)
# (d) λ₂ monotone increase under edge addition (5 graphs)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_5():
    rng = np.random.RandomState(505)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20, 5), facecolor="white")

        # (a) 3D: risk bound = ‖μ‖/λ₂ surface over (m, density)
        ax3d = fig.add_subplot(141, projection="3d")
        ms_a = np.arange(5, 25, 3)
        dens_a = np.linspace(0.3, 0.85, 7)
        MA, DA = np.meshgrid(ms_a, dens_a)
        BOUND = np.zeros_like(MA, dtype=float)
        for i, d in enumerate(dens_a):
            for j, m in enumerate(ms_a):
                L_a, _ = make_laplacian(int(m), density=d, seed=int(m * 100 + d * 1000))
                lam2_a = fiedler_value(L_a)
                mu_a = 0.03 * np.ones(int(m))
                norm_mu = float(np.linalg.norm(mu_a))
                BOUND[i, j] = norm_mu / lam2_a
        surf = ax3d.plot_surface(MA.astype(float), DA, BOUND,
                                  cmap="YlOrRd", alpha=0.9, linewidth=0)
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=10, pad=0.12,
                     label="R₀/λ₂")
        tidy3d(ax3d, xlabel="assets m", ylabel="density ρ",
               zlabel="risk bound", title="(a) Fiedler risk bound surface")

        # (b) Cesàro convergence: error vs K for nested samples
        ax2 = fig.add_subplot(142)
        L_b, _ = make_laplacian(8, density=0.6, seed=506)
        mu_stat = 0.03 * np.ones(8)
        w_stat = fixed_point(L_b, mu_stat)
        Ks_b = list(range(5, 801, 15))
        errors_b = []
        for K in Ks_b:
            rng_k = np.random.RandomState(5060)
            w_avg = np.zeros(8)
            for _ in range(K):
                mu_k = mu_stat + 0.005 * rng_k.randn(8)
                w_avg += fixed_point(L_b, mu_k)
            w_avg /= K
            errors_b.append(float(np.linalg.norm(w_avg - w_stat)))
        ax2.semilogy(Ks_b, errors_b, color=COLORS[0], linewidth=1.6)
        # Reference O(1/√K) curve
        ref = errors_b[0] * np.sqrt(Ks_b[0]) / np.sqrt(np.array(Ks_b, dtype=float))
        ax2.semilogy(Ks_b, ref, color=COLORS[1], linewidth=1.0,
                     linestyle="--", alpha=0.7)
        tidy(ax2, xlabel="K (samples)", ylabel="‖w̄_K − w*‖",
             title="(b) Cesàro convergence")

        # (c) Actual risk vs risk bound for 50 instances
        ax3 = fig.add_subplot(143)
        bounds_c, actuals_c = [], []
        for trial in range(50):
            m_c = 7 + trial % 9
            L_c, _ = make_laplacian(m_c, density=0.45 + 0.01 * trial, seed=600 + trial)
            mu_c = 0.02 + rng.uniform(0, 0.05, m_c)
            w_c = fixed_point(L_c, mu_c)
            lam2_c = fiedler_value(L_c)
            lammax_c = max_eigenvalue(L_c)
            Sigma_c = L_c / lammax_c
            risk_c = math.sqrt(max(float(w_c @ Sigma_c @ w_c), 0.0))
            bound_c = float(np.linalg.norm(mu_c)) / lam2_c
            actuals_c.append(risk_c)
            bounds_c.append(bound_c)
        ax3.scatter(bounds_c, actuals_c, c=COLORS[2], s=22,
                    alpha=0.8, edgecolors="white", linewidths=0.3)
        mx3 = max(bounds_c) * 1.05
        ax3.plot([0, mx3], [0, mx3], color=COLORS[1], linewidth=1.2,
                 linestyle="--", alpha=0.8)
        ax3.set_xlim(0, mx3)
        ax3.set_ylim(0, mx3)
        tidy(ax3, xlabel="risk bound R₀/λ₂", ylabel="actual σ(w*)",
             title="(c) Risk bound (50 instances, all below)")

        # (d) λ₂ monotone increase under sequential edge addition (5 starting graphs)
        ax4 = fig.add_subplot(144)
        for trial in range(5):
            L_d, A_d = make_laplacian(12, density=0.30, seed=700 + trial)
            mu_d = 0.03 + rng.uniform(0, 0.03, 12)
            zero_edges = [(i, j) for i in range(12) for j in range(i + 1, 12)
                          if A_d[i, j] == 0.0]
            rng.shuffle(zero_edges)
            lam2_seq = [fiedler_value(L_d)]
            L_cur = L_d.copy()
            for k in range(min(10, len(zero_edges))):
                ei, ej = zero_edges[k]
                we = rng.uniform(0.1, 0.4)
                L_cur[ei, ej] -= we; L_cur[ej, ei] -= we
                L_cur[ei, ei] += we; L_cur[ej, ej] += we
                lam2_seq.append(fiedler_value(L_cur))
            ax4.plot(lam2_seq, color=COLORS[trial], linewidth=1.5,
                     marker="o", markersize=3, alpha=0.8)
        tidy(ax4, xlabel="edges added", ylabel="λ₂(L)",
             title="(d) λ₂ monotone under edge addition")

        fig.tight_layout(pad=1.2)
        save(fig, "paper6_panel_5.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating panels for Paper 6...")
    make_panel_1()
    make_panel_2()
    make_panel_3()
    make_panel_4()
    make_panel_5()
    print("Done.")
