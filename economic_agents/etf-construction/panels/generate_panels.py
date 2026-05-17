"""
Panel generation for Paper 5: Optimal ETF Construction via Banach Fixed-Point Theory
5 panels, each with 4 subplots (at least one 3D), white background.
figsize=(20,5), dpi=150
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

COLORS = ["#1f4e79", "#c55a11", "#538135", "#7030a0", "#c00000"]

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


# ─────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────

def make_laplacian(m, density=0.6, seed=None):
    """Random weighted asset graph Laplacian."""
    rng = np.random.RandomState(seed)
    raw = rng.uniform(0.15, 0.85, (m, m))
    raw = (raw + raw.T) / 2
    np.fill_diagonal(raw, 0)
    mask_raw = rng.uniform(0, 1, (m, m))
    mask_raw = (mask_raw + mask_raw.T) / 2
    mask = (mask_raw < density).astype(float)
    np.fill_diagonal(mask, 0)
    # Ensure connectivity: add a random spanning chain
    perm = rng.permutation(m)
    for k in range(m - 1):
        i, j = perm[k], perm[k + 1]
        mask[i, j] = mask[j, i] = 1.0
    A = raw * mask
    d = A.sum(axis=1)
    L = np.diag(d) - A
    return L, A


def fiedler_value(L):
    """Second smallest eigenvalue (algebraic connectivity)."""
    ev = np.sort(np.linalg.eigvalsh(L))
    return float(ev[1])


def max_eigenvalue(L):
    """Largest eigenvalue of L."""
    ev = np.linalg.eigvalsh(L)
    return float(np.max(ev))


def pseudoinverse_L(L):
    """Moore-Penrose pseudoinverse via eigendecomposition."""
    ev, evec = np.linalg.eigh(L)
    tol = 1e-9 * np.max(np.abs(ev))
    inv_ev = np.where(np.abs(ev) > tol, 1.0 / ev, 0.0)
    return evec @ np.diag(inv_ev) @ evec.T


def proj_simplex(v):
    """Euclidean projection onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    rho = int(rho_arr[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def banach_iterate(L, mu, gamma, w0, n_iter):
    """Banach iterates; returns array shape (n_iter+1, m)."""
    m = len(mu)
    w = w0.copy()
    traj = [w.copy()]
    IgL = np.eye(m) - gamma * L
    for _ in range(n_iter):
        w = proj_simplex(IgL @ w + gamma * mu)
        traj.append(w.copy())
    return np.array(traj)


def fixed_point_weights(L, mu):
    """Interior fixed point w* = L†mu_c + (1/m)*1.
    Derivation: Lw* = mu_c (Kirchhoff), min-norm solution L†mu_c has 1^T(L†mu_c)=0,
    so we shift by 1/m to satisfy the simplex constraint 1^T w* = 1.
    """
    Ld = pseudoinverse_L(L)
    mu_c = mu - mu.mean()         # mu_c in Im(L): 1^T mu_c = 0
    w0 = Ld @ mu_c                # 1^T w0 = 0 since w0 in Im(L)
    m = len(mu)
    w_star = w0 + np.ones(m) / m  # shift so 1^T w_star = 1
    if np.all(w_star >= -1e-9):
        w_star = np.maximum(w_star, 0.0)
        return w_star / w_star.sum()
    return proj_simplex(w_star)   # boundary fixed point


def portfolio_variance(w, Sigma):
    return float(w @ Sigma @ w)


def composition_count(n, d):
    """T(n,d) = d * (d+1)^(n-1)."""
    return d * (d + 1) ** (n - 1)


# ─────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────

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


def bary_to_cart(w):
    """Barycentric (w1,w2,w3) → Cartesian (x,y) on equilateral triangle."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, math.sqrt(3) / 2])
    return w[0] * v1 + w[1] * v2 + w[2] * v3


# ─────────────────────────────────────────────
# Panel 1: Asset Graph and Laplacian Spectrum
# ─────────────────────────────────────────────

def make_panel_1():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            "Panel 1: Asset Graph Construction, Laplacian Spectrum, and Fiedler Value",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: λ₂ surface over (m, density)
        ax = fig.add_subplot(141, projection="3d")
        m_grid = np.arange(5, 31, 5)
        dens_grid = np.linspace(0.25, 0.85, 7)
        MG, DG = np.meshgrid(m_grid, dens_grid)
        LAM2 = np.zeros_like(MG, dtype=float)
        for i, d in enumerate(dens_grid):
            for j, m in enumerate(m_grid):
                L, _ = make_laplacian(int(m), density=d, seed=int(m * 100 + d * 1000))
                LAM2[i, j] = fiedler_value(L)
        surf = ax.plot_surface(MG.astype(float), DG, LAM2,
                               cmap="Blues", alpha=0.88)
        apply_style_3d(ax, r"(a) Fiedler Value $\lambda_2$ vs $(m,\,\rho)$",
                       "m (assets)", "density ρ", r"$\lambda_2(L)$")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Eigenvalue spectrum of a 20-asset graph
        ax2 = axes[1]
        L20, _ = make_laplacian(20, density=0.55, seed=99)
        ev20 = np.sort(np.linalg.eigvalsh(L20))
        colors_ev = [COLORS[1] if i == 0 else
                     COLORS[0] if i == 1 else
                     COLORS[2] for i in range(len(ev20))]
        ax2.bar(range(len(ev20)), ev20, color=colors_ev, edgecolor="white", width=0.8)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.annotate(r"$\lambda_1=0$", xy=(0, 0), xytext=(1.5, -0.3),
                     fontsize=8, color=COLORS[1])
        ax2.annotate(r"$\lambda_2$ (Fiedler)", xy=(1, ev20[1]), xytext=(3, ev20[1] + 0.5),
                     fontsize=8, color=COLORS[0],
                     arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=0.8))
        apply_style(ax2, r"(b) Eigenvalue Spectrum of 20-Asset Laplacian",
                    "Eigenvalue index", r"$\lambda_k(L)$")

        # (c) λ₂ vs m for three graph families
        ax3 = axes[2]
        ms = np.arange(4, 31)
        # Complete graph with unit weights: λ₂ = m
        lam2_complete = ms.astype(float)
        # Path graph: λ₂ = 2(1-cos(π/m))
        lam2_path = 2 * (1 - np.cos(np.pi / ms))
        # Random graph (density=0.6): numerical
        lam2_random = [fiedler_value(make_laplacian(int(m), 0.6, seed=m)[0]) for m in ms]
        ax3.plot(ms, lam2_complete, color=COLORS[0], linewidth=2, label="Complete (λ₂ = m)")
        ax3.plot(ms, lam2_random, color=COLORS[2], linewidth=2,
                 linestyle="--", label="Random (ρ = 0.6)")
        ax3.plot(ms, lam2_path, color=COLORS[1], linewidth=2,
                 linestyle=":", label=r"Path (λ₂ = 2(1−cos π/m))")
        ax3.legend(fontsize=8)
        apply_style(ax3, r"(c) Algebraic Connectivity $\lambda_2$ vs Asset Count $m$",
                    "m (assets)", r"$\lambda_2(L)$")

        # (d) Adjacency weight heatmap
        ax4 = axes[3]
        _, A20 = make_laplacian(20, density=0.55, seed=99)
        im = ax4.imshow(A20, cmap="Blues", aspect="auto", vmin=0)
        fig.colorbar(im, ax=ax4, shrink=0.8, label="Edge weight $w_{ij}$")
        ax4.set_xticks(range(0, 20, 4))
        ax4.set_yticks(range(0, 20, 4))
        ax4.tick_params(labelsize=7)
        ax4.set_title("(d) Asset Correlation Graph $A$ (20 assets)",
                      fontsize=11, fontweight="bold", pad=8)
        ax4.set_xlabel("Asset index $j$", fontsize=9)
        ax4.set_ylabel("Asset index $i$", fontsize=9)

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper5_panel_1.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 2: Contraction Mapping and Convergence
# ─────────────────────────────────────────────

def make_panel_2():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            r"Panel 2: Contraction Factor $\kappa = 1-\gamma\lambda_2$ and Banach Iteration Convergence",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: κ landscape over (γ·λ_max, λ₂/λ_max)
        lam_max_fixed = 8.0
        gamma_norm_grid = np.linspace(0.01, 0.99, 40)   # γ * λ_max ∈ (0,1)
        ratio_grid = np.linspace(0.05, 0.95, 40)        # λ₂/λ_max ∈ (0,1)
        GN, RG = np.meshgrid(gamma_norm_grid, ratio_grid)
        lam2_g = RG * lam_max_fixed
        gamma_g = GN / lam_max_fixed
        KAPPA_surf = 1.0 - gamma_g * lam2_g
        KAPPA_surf = np.clip(KAPPA_surf, 0, 1)

        ax = fig.add_subplot(141, projection="3d")
        surf = ax.plot_surface(GN, RG, KAPPA_surf, cmap="plasma_r", alpha=0.88)
        # Optimal line
        opt_ratio = ratio_grid
        opt_gn = 2.0 / (1 + opt_ratio)  # γ·λ_max at Chebyshev optimum
        opt_kappa = (1 - opt_ratio) / (1 + opt_ratio)
        ax.plot(opt_gn, opt_ratio, opt_kappa, color="red", linewidth=2,
                label="Optimal γ")
        apply_style_3d(ax, r"(a) Contraction Factor $\kappa(\gamma,\,\lambda_2/\lambda_m)$",
                       r"$\gamma\lambda_m$", r"$\lambda_2/\lambda_m$",
                       r"$\kappa = 1-\gamma\lambda_2$")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) ||w^(n) - w*||₂ vs iteration n
        ax2 = axes[1]
        np.random.seed(7)
        m_conv = 8
        L_conv, _ = make_laplacian(m_conv, density=0.65, seed=7)
        mu_conv = np.random.uniform(0.01, 0.08, m_conv)
        lam2_c = fiedler_value(L_conv)
        lamM_c = max_eigenvalue(L_conv)
        gamma_c = 2.0 / (lam2_c + lamM_c)   # optimal Chebyshev step
        kappa_c = (lamM_c - lam2_c) / (lamM_c + lam2_c)
        w_star = fixed_point_weights(L_conv, mu_conv)
        n_iter = 60
        for trial, color in zip(range(5), COLORS):
            w0 = proj_simplex(np.random.randn(m_conv))
            traj = banach_iterate(L_conv, mu_conv, gamma_c, w0, n_iter)
            errs = np.array([np.linalg.norm(traj[i] - w_star) for i in range(n_iter + 1)])
            ax2.plot(range(n_iter + 1), errs, color=color, linewidth=1.5,
                     alpha=0.85, label=f"Init {trial+1}")
        ax2.legend(fontsize=8)
        apply_style(ax2, r"(b) $\|w^{(n)}-w^*\|_2$ vs Iteration $n$",
                    "Iteration $n$", r"$\|w^{(n)}-w^*\|_2$")

        # (c) log(||w^(n) - w*||) vs n — linear decay
        ax3 = axes[2]
        np.random.seed(7)
        for trial, color in zip(range(5), COLORS):
            w0 = proj_simplex(np.random.randn(m_conv))
            traj = banach_iterate(L_conv, mu_conv, gamma_c, w0, n_iter)
            errs = np.array([np.linalg.norm(traj[i] - w_star) for i in range(n_iter + 1)])
            errs = np.maximum(errs, 1e-15)
            ax3.plot(range(n_iter + 1), np.log10(errs), color=color,
                     linewidth=1.5, alpha=0.85)
        # Theoretical slope
        n_th = np.arange(n_iter + 1)
        log_slope = np.log10(kappa_c) * n_th + np.log10(0.8)
        ax3.plot(n_th, log_slope, "k--", linewidth=1.8,
                 label=fr"Slope $= \log_{{10}}\kappa^*={math.log10(kappa_c):.3f}$")
        ax3.legend(fontsize=8)
        apply_style(ax3, r"(c) $\log_{10}\|w^{(n)}-w^*\|_2$: Linear Decay at Rate $\kappa^n$",
                    "Iteration $n$", r"$\log_{10}$ error")

        # (d) Optimal κ* vs condition ratio λ_max/λ₂
        ax4 = axes[3]
        cond_ratios = np.linspace(1.05, 30, 200)
        kappa_star = (cond_ratios - 1) / (cond_ratios + 1)
        kappa_approx = 1 - 2 / cond_ratios   # large-cond approximation
        ax4.plot(cond_ratios, kappa_star, color=COLORS[0], linewidth=2,
                 label=r"$\kappa^* = \frac{\lambda_m-\lambda_2}{\lambda_m+\lambda_2}$")
        ax4.plot(cond_ratios, np.clip(kappa_approx, 0, 1), color=COLORS[1],
                 linewidth=2, linestyle="--",
                 label=r"Approx. $1 - 2/(\lambda_m/\lambda_2)$")
        ax4.axvline(1, color="gray", linewidth=0.8, linestyle=":")
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize=8)
        apply_style(ax4, r"(d) Optimal $\kappa^*$ vs Condition Ratio $\lambda_m/\lambda_2$",
                    r"Condition ratio $\lambda_m/\lambda_2$",
                    r"$\kappa^* = (\lambda_m-\lambda_2)/(\lambda_m+\lambda_2)$")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper5_panel_2.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 3: Fixed-Point Portfolio and Kirchhoff
# ─────────────────────────────────────────────

def make_panel_3():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            r"Panel 3: Fixed-Point Portfolio $w^* = L^\dagger\mu/(\mathbf{1}^\top L^\dagger\mu)$ "
            r"and Kirchhoff Equilibrium $Lw^* = \mu - \xi\mathbf{1}$",
            fontsize=12, fontweight="bold", y=1.02
        )

        # (a) 3D: w₁* surface over (μ₁, μ₂) for 3-asset simplex
        ax = fig.add_subplot(141, projection="3d")
        # Fix 3-asset graph
        L3 = np.array([[1.2, -0.7, -0.5],
                       [-0.7, 1.5, -0.8],
                       [-0.5, -0.8, 1.3]])
        mu1_grid = np.linspace(0.005, 0.10, 30)
        mu2_grid = np.linspace(0.005, 0.10, 30)
        M1, M2 = np.meshgrid(mu1_grid, mu2_grid)
        W1_star = np.zeros_like(M1)
        mu3_fixed = 0.04
        for i in range(len(mu1_grid)):
            for j in range(len(mu2_grid)):
                mu_ij = np.array([M1[i, j], M2[i, j], mu3_fixed])
                w = fixed_point_weights(L3, mu_ij)
                W1_star[i, j] = max(0, w[0])
        surf = ax.plot_surface(M1 * 100, M2 * 100, W1_star,
                               cmap="Blues", alpha=0.88)
        apply_style_3d(ax,
                       r"(a) $w_1^*(\mu_1,\mu_2)$ for 3-Asset Simplex",
                       r"$\mu_1$ (%)", r"$\mu_2$ (%)", r"$w_1^*$")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Convergence paths on 3-asset simplex triangle
        ax2 = axes[1]
        # Draw triangle
        tri_verts = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
        ax2.plot(tri_verts[:, 0], tri_verts[:, 1], "k-", linewidth=1.2)
        ax2.text(-0.06, -0.04, r"$e_1$", fontsize=9)
        ax2.text(1.02, -0.04, r"$e_2$", fontsize=9)
        ax2.text(0.50, math.sqrt(3) / 2 + 0.03, r"$e_3$", fontsize=9)
        mu3v = np.array([0.06, 0.03, 0.045])
        w_star3 = fixed_point_weights(L3, mu3v)
        w_star3 = np.maximum(w_star3, 0)
        w_star3 /= w_star3.sum()
        cx, cy = bary_to_cart(w_star3)
        # Banach trajectories
        lam2_3 = fiedler_value(L3)
        lamM_3 = max_eigenvalue(L3)
        g3 = 2.0 / (lam2_3 + lamM_3)
        np.random.seed(3)
        for k, color in zip(range(6), COLORS * 2):
            w0 = proj_simplex(np.random.randn(3))
            traj3 = banach_iterate(L3, mu3v, g3, w0, 80)
            cart = np.array([bary_to_cart(np.maximum(traj3[i], 0) / np.maximum(traj3[i], 0).sum())
                             for i in range(0, 81, 2)])
            ax2.plot(cart[:, 0], cart[:, 1], color=color,
                     linewidth=1.2, alpha=0.7)
            ax2.plot(cart[0, 0], cart[0, 1], "o", color=color, markersize=5)
        ax2.plot(cx, cy, "k*", markersize=14, label=r"$w^*$", zorder=5)
        ax2.legend(fontsize=9)
        ax2.set_aspect("equal")
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.08, 0.97)
        ax2.axis("off")
        ax2.set_title(r"(b) Convergence Paths on 3-Asset Simplex $\Delta_3$",
                      fontsize=11, fontweight="bold", pad=8)

        # (c) Kirchhoff scatter: (Lw*)_i vs (μ_i - ξ)
        ax3 = axes[2]
        np.random.seed(42)
        kirchhoff_lhs = []
        kirchhoff_rhs = []
        for _ in range(50):
            m_k = np.random.randint(5, 15)
            L_k, _ = make_laplacian(m_k, density=0.6, seed=np.random.randint(1000))
            mu_k = np.random.uniform(0.01, 0.09, m_k)
            w_k = fixed_point_weights(L_k, mu_k)
            w_k = np.maximum(w_k, 0)
            w_k /= w_k.sum()
            lhs = L_k @ w_k
            xi_k = mu_k.mean()
            rhs = mu_k - xi_k
            kirchhoff_lhs.extend(lhs.tolist())
            kirchhoff_rhs.extend(rhs.tolist())
        kirchhoff_lhs = np.array(kirchhoff_lhs)
        kirchhoff_rhs = np.array(kirchhoff_rhs)
        ax3.scatter(kirchhoff_rhs, kirchhoff_lhs, alpha=0.35, s=8,
                    color=COLORS[0], label="50 random instances")
        lim = max(abs(kirchhoff_rhs).max(), abs(kirchhoff_lhs).max()) * 1.05
        ax3.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5,
                 label=r"$Lw^* = \mu-\xi\mathbf{1}$")
        ax3.legend(fontsize=8)
        apply_style(ax3,
                    r"(c) Kirchhoff Verification: $(Lw^*)_i$ vs $(\mu_i-\bar\mu)$",
                    r"$\mu_i - \bar\mu$ (r.h.s.)",
                    r"$(Lw^*)_i$ (l.h.s.)")

        # (d) Portfolio weight profiles for 5 return configurations
        ax4 = axes[3]
        m_d = 8
        L_d, _ = make_laplacian(m_d, density=0.6, seed=55)
        configs = {
            "Uniform μ": np.ones(m_d) * 0.04,
            "Rising μ": np.linspace(0.01, 0.09, m_d),
            "Falling μ": np.linspace(0.09, 0.01, m_d),
            "Two spikes": np.array([0.01, 0.01, 0.09, 0.01, 0.01, 0.09, 0.01, 0.01]),
            "Random μ": np.array([0.07, 0.02, 0.05, 0.08, 0.03, 0.06, 0.01, 0.04]),
        }
        x = np.arange(m_d)
        width = 0.15
        for k, (label, mu_cfg) in enumerate(configs.items()):
            w_cfg = fixed_point_weights(L_d, mu_cfg)
            w_cfg = np.maximum(w_cfg, 0)
            w_cfg /= w_cfg.sum()
            ax4.bar(x + k * width, w_cfg, width, color=COLORS[k % len(COLORS)],
                    alpha=0.78, label=label)
        ax4.axhline(1 / m_d, color="gray", linewidth=1, linestyle=":",
                    label=f"1/m = {1/m_d:.3f}")
        ax4.legend(fontsize=7.5)
        apply_style(ax4, r"(d) Optimal Weights $w^*$ for 5 Return Configurations",
                    "Asset index", r"Portfolio weight $w_i^*$")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper5_panel_3.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 4: Risk Bound and Fiedler Value
# ─────────────────────────────────────────────

def make_panel_4():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            r"Panel 4: Risk Bound $\sigma(w^*)\leq R_0/\lambda_2$ "
            r"and the Diversification Premium",
            fontsize=13, fontweight="bold", y=1.02
        )

        # (a) 3D: Risk bound R₀/λ₂ over (R₀, λ₂)
        ax = fig.add_subplot(141, projection="3d")
        R0_vals = np.linspace(0.05, 2.0, 35)
        lam2_vals = np.linspace(0.1, 4.0, 35)
        R0G, L2G = np.meshgrid(R0_vals, lam2_vals)
        BOUND = R0G / L2G
        surf = ax.plot_surface(R0G, L2G, BOUND, cmap="YlOrRd_r", alpha=0.88)
        apply_style_3d(ax,
                       r"(a) Risk Bound $R_0/\lambda_2$ Surface",
                       r"$R_0 = \sigma_{\max}\|\mu\|_2$",
                       r"$\lambda_2(L)$",
                       r"Risk bound")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) Actual σ(w*) vs bound R₀/λ₂ scatter
        ax2 = axes[1]
        np.random.seed(17)
        actuals = []
        bounds = []
        lam2_col = []
        n_instances = 180
        for _ in range(n_instances):
            m_b = np.random.randint(6, 18)
            L_b, _ = make_laplacian(m_b, density=np.random.uniform(0.4, 0.8),
                                    seed=np.random.randint(5000))
            mu_b = np.random.uniform(0.01, 0.08, m_b)
            w_b = fixed_point_weights(L_b, mu_b)
            w_b = np.maximum(w_b, 0)
            w_b /= w_b.sum()
            # Covariance: use normalised L as proxy
            lam2_b = fiedler_value(L_b)
            lamM_b = max_eigenvalue(L_b)
            Sigma_b = L_b / (lamM_b + 1e-9)   # normalised, PSD
            actual = math.sqrt(max(0, float(w_b @ Sigma_b @ w_b)))
            sig_max = math.sqrt(max(np.linalg.eigvalsh(Sigma_b)))
            R0_b = sig_max * np.linalg.norm(mu_b)
            bound_b = R0_b / (lam2_b + 1e-9)
            actuals.append(actual)
            bounds.append(bound_b)
            lam2_col.append(lam2_b)
        actuals = np.array(actuals)
        bounds = np.array(bounds)
        sc = ax2.scatter(bounds, actuals, c=lam2_col, cmap="Blues",
                         alpha=0.55, s=18, label="Random ETF")
        diag = np.linspace(0, max(bounds) * 1.02, 100)
        ax2.plot(diag, diag, "r--", linewidth=1.5,
                 label=r"$\sigma(w^*) = R_0/\lambda_2$ (bound)")
        fig.colorbar(sc, ax=ax2, shrink=0.7, label=r"$\lambda_2$")
        ax2.legend(fontsize=8)
        apply_style(ax2,
                    r"(b) Actual $\sigma(w^*)$ vs Bound $R_0/\lambda_2$ (180 ETFs)",
                    r"Risk bound $R_0/\lambda_2$",
                    r"Actual $\sigma(w^*)$")

        # (c) σ(w*) vs λ₂ as edges added to a 12-asset graph
        ax3 = axes[2]
        np.random.seed(23)
        m_c = 12
        # Start with a minimal spanning tree (m-1 edges), add edges progressively
        rng_c = np.random.RandomState(23)
        base_A = np.zeros((m_c, m_c))
        perm_c = rng_c.permutation(m_c)
        for k in range(m_c - 1):
            i, j = int(perm_c[k]), int(perm_c[k + 1])
            w_ij = rng_c.uniform(0.2, 0.6)
            base_A[i, j] = base_A[j, i] = w_ij
        # All potential additional edges (not in spanning tree)
        potential_edges = [(i, j, rng_c.uniform(0.1, 0.7))
                           for i in range(m_c) for j in range(i + 1, m_c)
                           if base_A[i, j] == 0]
        rng_c.shuffle(potential_edges)
        mu_c = rng_c.uniform(0.02, 0.07, m_c)
        lam2_seq = []
        sigma_seq = []
        A_curr = base_A.copy()
        for step in range(len(potential_edges) + 1):
            d_curr = A_curr.sum(axis=1)
            L_curr = np.diag(d_curr) - A_curr
            lam2_s = fiedler_value(L_curr)
            lamM_s = max_eigenvalue(L_curr)
            w_s = fixed_point_weights(L_curr, mu_c)
            w_s = np.maximum(w_s, 0)
            w_s /= w_s.sum()
            Sigma_s = L_curr / (lamM_s + 1e-9)
            sig_s = math.sqrt(max(0, float(w_s @ Sigma_s @ w_s)))
            lam2_seq.append(lam2_s)
            sigma_seq.append(sig_s)
            if step < len(potential_edges):
                ei, ej, ew = potential_edges[step]
                A_curr[ei, ej] = A_curr[ej, ei] = ew
        ax3.plot(lam2_seq, sigma_seq, color=COLORS[0], linewidth=2,
                 marker="o", markersize=3.5, label=r"$\sigma(w^*)$ (actual)")
        sig_max_c = math.sqrt(max(np.linalg.eigvalsh(L_curr / (max_eigenvalue(L_curr) + 1e-9))))
        R0_c = sig_max_c * np.linalg.norm(mu_c)
        lam2_plot = np.linspace(min(lam2_seq) * 0.9, max(lam2_seq) * 1.05, 100)
        ax3.plot(lam2_plot, R0_c / lam2_plot, color=COLORS[1], linewidth=2,
                 linestyle="--", label=r"Bound $R_0/\lambda_2$")
        ax3.legend(fontsize=8)
        apply_style(ax3,
                    r"(c) Portfolio Risk $\sigma(w^*)$ as Edges Added (m=12)",
                    r"$\lambda_2(L)$ (algebraic connectivity)",
                    r"Portfolio risk $\sigma(w^*)$")

        # (d) Bound tightness (actual/bound) vs λ₂
        ax4 = axes[3]
        tightness = np.array(actuals) / (np.array(bounds) + 1e-12)
        ax4.scatter(lam2_col, tightness, alpha=0.45, s=15,
                    color=COLORS[0], label="Random ETFs")
        ax4.axhline(1.0, color="red", linewidth=1.5, linestyle="--",
                    label="Bound (ratio = 1)")
        ax4.set_ylim(0, 1.05)
        ax4.legend(fontsize=8)
        apply_style(ax4,
                    r"(d) Bound Tightness $\sigma(w^*)/(R_0/\lambda_2)$ vs $\lambda_2$",
                    r"$\lambda_2(L)$ (Fiedler value)",
                    r"Tightness ratio $\leq 1$")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper5_panel_4.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Panel 5: Composition-Inflation and Execution
# ─────────────────────────────────────────────

def make_panel_5():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white")
        fig.suptitle(
            r"Panel 5: Composition-Inflation $\mathcal{T}(n,d)=d\cdot(d+1)^{n-1}$ "
            r"and $O(1)$ Execution via Pre-Computed State Tables",
            fontsize=12, fontweight="bold", y=1.02
        )

        # (a) 3D: T(n,d) surface over (n, d)
        ax = fig.add_subplot(141, projection="3d")
        n_vals_3d = np.arange(1, 14)
        d_vals_3d = np.arange(1, 6)
        NG, DG = np.meshgrid(n_vals_3d, d_vals_3d)
        TG = np.vectorize(composition_count)(NG, DG).astype(float)
        log_TG = np.log10(np.clip(TG, 1, None))
        surf = ax.plot_surface(NG.astype(float), DG.astype(float), log_TG,
                               cmap="viridis", alpha=0.88)
        apply_style_3d(ax,
                       r"(a) $\log_{10}\mathcal{T}(n,d) = \log_{10}[d(d+1)^{n-1}]$",
                       "n (cycles)", "d (dimensions)",
                       r"$\log_{10}\mathcal{T}$")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)

        # (b) T(n,3) = 3·4^(n-1): states and memory for m=500
        ax2 = axes[1]
        n_range = np.arange(1, 18)
        T3 = np.array([composition_count(int(n), 3) for n in n_range], dtype=float)
        mem_GB = T3 * 500 * 4 / 1e9   # m=500 assets, 4 bytes per float
        ax2.semilogy(n_range, T3, color=COLORS[0], linewidth=2.5,
                     marker="o", markersize=5, label=r"$\mathcal{T}(n,3)=3\cdot4^{n-1}$")
        ax2_r = ax2.twinx()
        ax2_r.plot(n_range, mem_GB, color=COLORS[1], linewidth=2,
                   linestyle="--", marker="s", markersize=4,
                   label="Memory (m=500, GB)")
        ax2_r.axhline(1.57, color=COLORS[1], linewidth=0.8, linestyle=":", alpha=0.6)
        ax2_r.axhline(0.098, color=COLORS[2], linewidth=0.8, linestyle=":", alpha=0.6)
        ax2_r.annotate("1.57 GB (n₀=10)", xy=(10, 1.57),
                       xytext=(11.5, 3), fontsize=7.5, color=COLORS[1],
                       arrowprops=dict(arrowstyle="->", lw=0.7))
        ax2_r.annotate("98 MB (n₀=8)", xy=(8, 0.098),
                       xytext=(9.5, 0.35), fontsize=7.5, color=COLORS[2],
                       arrowprops=dict(arrowstyle="->", lw=0.7))
        ax2_r.set_ylabel("Memory footprint (GB)", fontsize=9, color=COLORS[1])
        ax2_r.tick_params(axis="y", labelsize=8, labelcolor=COLORS[1])
        ax2.set_xlabel("$n_0$ (pre-computation depth)", fontsize=9)
        ax2.set_ylabel(r"$\mathcal{T}(n_0,3)$ (state count)", fontsize=9)
        ax2.set_title(r"(b) $\mathcal{T}(n,3)=3\cdot4^{n-1}$ and Memory Footprint (m=500)",
                      fontsize=11, fontweight="bold", pad=8)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.4, linestyle="--")

        # (c) Execution speedup S vs portfolio size m
        ax3 = axes[2]
        m_range = np.logspace(1, 3, 100)
        n0_fixed = 10
        eps_fixed = 1e-6
        configs_speed = [
            (10, 1e-6, r"cond=10, $\varepsilon=10^{-6}$"),
            (50, 1e-6, r"cond=50, $\varepsilon=10^{-6}$"),
            (100, 1e-6, r"cond=100, $\varepsilon=10^{-6}$"),
            (100, 1e-8, r"cond=100, $\varepsilon=10^{-8}$"),
        ]
        for cond, eps, label in configs_speed:
            speedup = (m_range ** 2) * cond * math.log(1 / eps) / n0_fixed
            ax3.loglog(m_range, speedup, linewidth=2, label=label)
        ax3.axvline(500, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.annotate("m=500", xy=(500, 1e5), xytext=(550, 5e4),
                     fontsize=8, rotation=0)
        ax3.legend(fontsize=7.5)
        apply_style(ax3,
                    r"(c) Speedup $S = m^2\cdot\mathrm{cond}\cdot\log(1/\varepsilon)/n_0$ vs $m$",
                    "Portfolio size $m$",
                    r"Speedup factor $S$")

        # (d) State count T(n₀,3) and memory for m ∈ {50, 100, 500}
        ax4 = axes[3]
        n0_range = np.arange(1, 18)
        T3_d = np.array([composition_count(int(n), 3) for n in n0_range], dtype=float)
        for m_d, color in zip([50, 100, 500], COLORS[:3]):
            mem_d = T3_d * m_d * 4 / 1e6  # MB
            ax4.semilogy(n0_range, mem_d, color=color, linewidth=2,
                         marker="o", markersize=4, label=f"m={m_d}")
        ax4.axhline(98, color="gray", linewidth=1, linestyle="--", alpha=0.7,
                    label="L3 cache ≈ 100 MB")
        ax4.axhline(1000, color="gray", linewidth=1, linestyle=":", alpha=0.7,
                    label="RAM target 1 GB")
        ax4.legend(fontsize=8)
        apply_style(ax4,
                    r"(d) Memory $\mathcal{T}(n_0,3)\cdot m\cdot 4$ bytes vs $n_0$",
                    "$n_0$ (pre-computation depth)",
                    "Memory (MB)")

        axes[0].remove()
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "paper5_panel_5.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating Paper 5 panels...")
    make_panel_1()
    make_panel_2()
    make_panel_3()
    make_panel_4()
    make_panel_5()
    print("All Paper 5 panels generated.")
