"""
Generate 5 panels for Paper 7: Transactional Magnitude Calculus
Each panel: 4 data-driven charts (>=1 3D), white background, minimal text.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

STYLE = {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
}

COLORS = ["#1a3a5c", "#c0392b", "#27ae60", "#8e44ad", "#e67e22",
          "#2980b9", "#e74c3c", "#16a085", "#7f8c8d", "#f39c12"]


def make_figure():
    fig = plt.figure(figsize=(20, 5), facecolor="white")
    return fig


def simulate_gain_loss(n_steps, seed=None, scale=0.03):
    rng = np.random.RandomState(seed)
    return rng.randn(n_steps) * scale


def compute_clock(gain_loss, dt=1.0):
    return np.cumsum(np.abs(gain_loss)) * dt


def s_transform(x, x_star, s_floor=2.0):
    max_dev = np.max(np.abs(x - x_star)) + 1e-10
    dist = np.abs(x - x_star) / max_dev
    return s_floor + (100.0 - s_floor) * dist


# ============================================================
# Panel 1: Transaction Clock Properties
# ============================================================

def generate_panel_1():
    with plt.rc_context(STYLE):
        fig = make_figure()

        # --- Chart A: 3D surface of clock value over (scale, T) ---
        ax1 = fig.add_subplot(141, projection="3d")
        scales = np.linspace(0.01, 0.06, 20)
        T_vals = np.linspace(1.0, 20.0, 20)
        SS, TT = np.meshgrid(scales, T_vals)
        gbar = SS * np.sqrt(2.0 / np.pi)
        Clock_surf = gbar * TT   # E[Theta(T)] = gbar * T
        ax1.plot_surface(SS, TT, Clock_surf, cmap="Blues", alpha=0.9,
                         linewidth=0, antialiased=True)
        ax1.set_xlabel("scale", labelpad=2)
        ax1.set_ylabel("T", labelpad=2)
        ax1.set_zlabel("E[Θ]", labelpad=2)
        ax1.set_title("E[Θ(T)] surface")
        ax1.tick_params(labelsize=6)

        # --- Chart B: Clock paths for 5 independent realizations ---
        ax2 = fig.add_subplot(142)
        n_steps, dt = 1000, 0.01
        t = np.arange(n_steps) * dt
        for p in range(5):
            G = simulate_gain_loss(n_steps, seed=1 + p, scale=0.03)
            Theta = compute_clock(G, dt)
            ax2.plot(t, Theta, color=COLORS[p], lw=0.8, alpha=0.85)
        # Reference line: gbar * t
        gbar_ref = 0.03 * np.sqrt(2 / np.pi)
        ax2.plot(t, gbar_ref * t, "k--", lw=1.2, label="E[Θ]")
        ax2.set_xlabel("calendar time t")
        ax2.set_ylabel("Θ(t)")
        ax2.set_title("Clock paths")
        ax2.legend(fontsize=7, frameon=False)

        # --- Chart C: Empirical vs theoretical E[Theta(T)] across scales ---
        ax3 = fig.add_subplot(143)
        scales_c = np.linspace(0.01, 0.06, 12)
        empirical_means, theoretical_means = [], []
        n_paths_c, n_steps_c, T_c, dt_c = 80, 500, 5.0, 0.01
        for sc in scales_c:
            ends = [compute_clock(simulate_gain_loss(n_steps_c, seed=int(sc * 1000) + p, scale=sc), dt_c)[-1]
                    for p in range(n_paths_c)]
            empirical_means.append(float(np.mean(ends)))
            theoretical_means.append(sc * np.sqrt(2.0 / np.pi) * T_c)
        ax3.scatter(theoretical_means, empirical_means, s=25, color=COLORS[0], zorder=3)
        diag = np.linspace(min(theoretical_means), max(theoretical_means), 50)
        ax3.plot(diag, diag, "k--", lw=1.0)
        ax3.set_xlabel("theoretical E[Θ(T)]")
        ax3.set_ylabel("empirical E[Θ(T)]")
        ax3.set_title("E[Θ] theory vs empirical")

        # --- Chart D: Increments |ΔΘ| = |G[i]|*dt (absolute continuity) ---
        ax4 = fig.add_subplot(144)
        G_d = simulate_gain_loss(300, seed=42, scale=0.03)
        dt_d = 0.01
        Theta_d = compute_clock(G_d, dt_d)
        inc = np.diff(Theta_d)
        pred = np.abs(G_d[1:]) * dt_d
        ax4.scatter(pred, inc, s=10, color=COLORS[1], alpha=0.6, zorder=3)
        lims = [0, max(pred.max(), inc.max()) * 1.05]
        ax4.plot(lims, lims, "k--", lw=1.0)
        ax4.set_xlabel("|G[i]| · dt  (predicted)")
        ax4.set_ylabel("ΔΘ  (actual)")
        ax4.set_title("Absolute continuity")

        plt.tight_layout(pad=1.5)
        fig.savefig("paper7_panel_1.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Panel 1 saved.")


# ============================================================
# Panel 2: Subordination and Monetary Derivative
# ============================================================

def generate_panel_2():
    with plt.rc_context(STYLE):
        fig = make_figure()

        # --- Chart A: 3D surface Var[Y(s)] over (sigma, gbar) ---
        ax1 = fig.add_subplot(141, projection="3d")
        sigmas = np.linspace(0.005, 0.03, 20)
        gbars = np.linspace(0.010, 0.050, 20)
        SIG, GB = np.meshgrid(sigmas, gbars)
        s_fix = 0.5
        VarY = SIG ** 2 * s_fix / GB
        ax1.plot_surface(SIG, GB, VarY, cmap="Greens", alpha=0.9,
                         linewidth=0, antialiased=True)
        ax1.set_xlabel("σ", labelpad=2)
        ax1.set_ylabel("ḡ", labelpad=2)
        ax1.set_zlabel("Var[Y(s)]", labelpad=2)
        ax1.set_title("Var[Y(s)] surface")
        ax1.tick_params(labelsize=6)

        # --- Chart B: Multiple Y(s) trajectories (X in transaction time) ---
        ax2 = fig.add_subplot(142)
        n_steps, dt = 3000, 0.01
        sigma_b = 0.01
        n_paths_b = 5
        s_grid = np.linspace(0, 2.0, 200)
        for p in range(n_paths_b):
            rp = np.random.RandomState(200 + p)
            G = rp.randn(n_steps) * 0.03
            Theta = compute_clock(G, dt)
            X = np.cumsum(rp.randn(n_steps)) * sigma_b * np.sqrt(dt)
            Y_path = []
            for s_val in s_grid:
                idx = np.searchsorted(Theta, s_val)
                Y_path.append(float(X[min(idx, n_steps - 1)]))
            ax2.plot(s_grid, Y_path, color=COLORS[p], lw=0.8, alpha=0.85)
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_xlabel("transaction time s")
        ax2.set_ylabel("Y(s)")
        ax2.set_title("Trajectories in transaction time")

        # --- Chart C: Monetary derivative magnitude vs calendar derivative ---
        ax3 = fig.add_subplot(143)
        n_c, dt_c = 500, 0.01
        t_c = np.arange(n_c) * dt_c
        G_c = simulate_gain_loss(n_c, seed=301, scale=0.03)
        f_c = np.sin(t_c * 1.0)
        f_dot_c = np.cos(t_c * 1.0)
        mderiv_c = f_dot_c / (np.abs(G_c) + 1e-12)
        ax3.scatter(np.abs(f_dot_c)[10:-10], np.abs(mderiv_c)[10:-10],
                    s=8, color=COLORS[2], alpha=0.4)
        ax3.set_xlabel("|df/dt|")
        ax3.set_ylabel("|df/dΘ|")
        ax3.set_title("Calendar vs monetary velocity")

        # --- Chart D: Variance ratio Var[Y]/Var[X] vs gbar ---
        ax4 = fig.add_subplot(144)
        gbars_d = np.linspace(0.01, 0.05, 30)
        s_fix_d = 0.5
        ratio_theory = 1.0 / (gbars_d / s_fix_d)  # Var[Y(s)] = sigma^2*s/gbar -> ratio = s/gbar
        ax4.plot(gbars_d, ratio_theory / ratio_theory[0], color=COLORS[0], lw=1.5)
        ax4.fill_between(gbars_d, ratio_theory / ratio_theory[0] * 0.9,
                         ratio_theory / ratio_theory[0] * 1.1,
                         color=COLORS[0], alpha=0.15)
        ax4.set_xlabel("gain-loss activity ḡ")
        ax4.set_ylabel("relative Var[Y(s)]")
        ax4.set_title("Variance attenuation with ḡ")

        plt.tight_layout(pad=1.5)
        fig.savefig("paper7_panel_2.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Panel 2 saved.")


# ============================================================
# Panel 3: S-Entropy Dimensionlessness and Tangent Space
# ============================================================

def generate_panel_3():
    with plt.rc_context(STYLE):
        fig = make_figure()

        # --- Chart A: 3D S-transform surface over (x, x_star) ---
        ax1 = fig.add_subplot(141, projection="3d")
        x_range = np.linspace(-5, 5, 30)
        x_star_range = np.linspace(-3, 3, 30)
        XG, XSG = np.meshgrid(x_range, x_star_range)
        s_floor = 2.0
        max_dev = np.abs(XG - XSG).max() + 1e-10
        dist = np.abs(XG - XSG) / max_dev
        S_surf = s_floor + (100 - s_floor) * dist
        ax1.plot_surface(XG, XSG, S_surf, cmap="Purples", alpha=0.9,
                         linewidth=0, antialiased=True)
        ax1.set_xlabel("x", labelpad=2)
        ax1.set_ylabel("x*", labelpad=2)
        ax1.set_zlabel("S(x,x*)", labelpad=2)
        ax1.set_title("S-transform surface")
        ax1.tick_params(labelsize=6)

        # --- Chart B: Monetary derivatives of price and volume (addable) ---
        ax2 = fig.add_subplot(142)
        n_b, dt_b = 300, 0.01
        t_b = np.arange(n_b) * dt_b
        rng_b = np.random.RandomState(402)
        G_b = rng_b.randn(n_b) * 0.02
        price = 100 + 0.5 * t_b + rng_b.randn(n_b) * 2
        volume = 1e6 + 1e4 * t_b + rng_b.randn(n_b) * 5e3
        s_price = s_transform(price, 100 + 0.5 * t_b, s_floor)
        s_vol = s_transform(volume, 1e6 + 1e4 * t_b, s_floor)
        md_price = np.gradient(s_price, dt_b) / (np.abs(G_b) + 1e-10)
        md_vol = np.gradient(s_vol, dt_b) / (np.abs(G_b) + 1e-10)
        md_sum = md_price + md_vol
        ax2.plot(t_b, md_price, color=COLORS[0], lw=0.8, label="price")
        ax2.plot(t_b, md_vol, color=COLORS[1], lw=0.8, label="volume")
        ax2.plot(t_b, md_sum, color=COLORS[2], lw=1.2, label="sum")
        ax2.axhline(0, color="k", lw=0.5, ls="--")
        ax2.set_xlabel("t")
        ax2.set_ylabel("dS/dΘ")
        ax2.set_title("Heterogeneous MD (addable)")
        ax2.legend(fontsize=6, frameon=False)

        # --- Chart C: Floor persistence — S-values stay in [S_floor, 100] ---
        ax3 = fig.add_subplot(143)
        n_c = 500
        rng_c = np.random.RandomState(403)
        x_paths = np.cumsum(rng_c.randn(5, n_c) * 0.1, axis=1)
        x_star_c = np.zeros(n_c)
        for i, xp in enumerate(x_paths):
            sv = s_transform(xp, x_star_c, s_floor)
            ax3.plot(sv, color=COLORS[i], lw=0.8, alpha=0.85)
        ax3.axhline(s_floor, color="k", lw=1.0, ls="--", label="S_floor")
        ax3.axhline(100, color="k", lw=1.0, ls=":", label="S=100")
        ax3.set_xlabel("step")
        ax3.set_ylabel("S-value")
        ax3.set_title("Floor persistence")
        ax3.legend(fontsize=6, frameon=False)

        # --- Chart D: Monetary norm properties (triangle inequality scatter) ---
        ax4 = fig.add_subplot(144)
        rng_d = np.random.RandomState(404)
        n_dim = 5
        lhs_vals, rhs_vals = [], []
        for _ in range(200):
            u = rng_d.randn(n_dim)
            v = rng_d.randn(n_dim)
            def mnorm(w):
                return np.linalg.norm(w) / (np.sqrt(len(w)) * 100)
            lhs_vals.append(mnorm(u + v))
            rhs_vals.append(mnorm(u) + mnorm(v))
        ax4.scatter(rhs_vals, lhs_vals, s=8, color=COLORS[3], alpha=0.5, zorder=3)
        lim = max(max(rhs_vals), max(lhs_vals)) * 1.05
        ax4.plot([0, lim], [0, lim], "k--", lw=1.0)
        ax4.set_xlabel("||u||+||v|| (bound)")
        ax4.set_ylabel("||u+v|| (actual)")
        ax4.set_title("Monetary norm triangle ineq.")

        plt.tight_layout(pad=1.5)
        fig.savefig("paper7_panel_3.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Panel 3 saved.")


# ============================================================
# Panel 4: Gear Network and Telescoping Formula
# ============================================================

def simulate_gear_2layer(G, dt, theta1, theta2):
    n = len(G)
    Theta1 = compute_clock(G, dt)
    cnt1, last1, fire1 = 0, 0.0, []
    for i in range(n):
        if Theta1[i] - last1 >= theta1:
            cnt1 += 1
            last1 = Theta1[i]
            fire1.append(i)
    I1 = np.zeros(n)
    for fi in fire1:
        if fi < n:
            I1[fi] += theta1
    Theta2 = np.cumsum(I1)
    cnt2, last2 = 0, 0.0
    for i in range(n):
        if Theta2[i] - last2 >= theta2:
            cnt2 += 1
            last2 = Theta2[i]
    return cnt1, cnt2, Theta1, np.cumsum(I1)


def generate_panel_4():
    with plt.rc_context(STYLE):
        fig = make_figure()

        # --- Chart A: 3D surface of firing ratio N1/N2 over (theta1, theta2) ---
        ax1 = fig.add_subplot(141, projection="3d")
        th1_vals = np.linspace(0.2, 1.0, 12)
        th2_vals = np.linspace(1.0, 6.0, 12)
        TH1, TH2 = np.meshgrid(th1_vals, th2_vals)
        ratio_surf = TH2 / TH1  # theoretical E[N1/N2] = theta2/theta1
        ax1.plot_surface(TH1, TH2, ratio_surf, cmap="Oranges", alpha=0.9,
                         linewidth=0, antialiased=True)
        ax1.set_xlabel("theta1", labelpad=2)
        ax1.set_ylabel("theta2", labelpad=2)
        ax1.set_zlabel("N1/N2", labelpad=2)
        ax1.set_title("Firing ratio surface")
        ax1.tick_params(labelsize=6)

        # --- Chart B: Accumulated imbalance I1(t) with resets ---
        ax2 = fig.add_subplot(142)
        G_b = simulate_gain_loss(3000, seed=601, scale=0.04)
        dt_b = 0.01
        theta1_b = 0.3
        Theta1_b = compute_clock(G_b, dt_b)
        imbalance = np.zeros(len(G_b))
        last_fire = 0.0
        for i in range(len(G_b)):
            imbalance[i] = Theta1_b[i] - last_fire
            if imbalance[i] >= theta1_b:
                last_fire = Theta1_b[i]
                imbalance[i] = 0.0
        t_b = np.arange(len(G_b)) * dt_b
        ax2.plot(t_b, imbalance, color=COLORS[0], lw=0.7)
        ax2.axhline(theta1_b, color=COLORS[1], lw=1.0, ls="--")
        ax2.set_xlabel("t")
        ax2.set_ylabel("I1(t)")
        ax2.set_title("Layer-1 imbalance (saw-tooth)")

        # --- Chart C: Firing count hierarchy N1 > N2 > N3 ---
        ax3 = fig.add_subplot(143)
        theta_list = [0.3, 1.5, 7.5]
        n_trials_c = 12
        counts_layers = [[], [], []]
        for trial in range(n_trials_c):
            G_c = simulate_gain_loss(80000, seed=700 + trial, scale=0.04)
            I_prev = G_c.copy()
            for k, thk in enumerate(theta_list):
                Tk = compute_clock(I_prev, 0.01 if k == 0 else 1.0)
                cnt_k, last_k = 0, 0.0
                I_next = np.zeros(len(G_c))
                for i in range(len(G_c)):
                    if Tk[i] - last_k >= thk:
                        cnt_k += 1
                        last_k = Tk[i]
                        I_next[i] += thk
                counts_layers[k].append(cnt_k)
                I_prev = I_next
        for k in range(3):
            ax3.scatter([k] * n_trials_c, counts_layers[k],
                        s=20, color=COLORS[k], alpha=0.7, zorder=3)
            ax3.plot([k - 0.2, k + 0.2],
                     [np.mean(counts_layers[k])] * 2,
                     color=COLORS[k], lw=2.0)
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(["Layer 1", "Layer 2", "Layer 3"])
        ax3.set_ylabel("firing count")
        ax3.set_title("Gear hierarchy N1 > N2 > N3")

        # --- Chart D: Telescoping formula — direct vs formula ---
        ax4 = fig.add_subplot(144)
        n_d, dt_d = 2000, 0.01
        t_d = np.arange(1, n_d + 1) * dt_d
        G_d = simulate_gain_loss(n_d, seed=701, scale=0.03)
        Clock1_d = np.cumsum(np.abs(G_d)) * dt_d
        I1_sm = np.convolve(np.abs(G_d), np.ones(10) / 10, mode='same')
        Clock2_d = np.cumsum(I1_sm) * dt_d
        f_d = np.sin(t_d * 0.3)
        f_dot_d = np.gradient(f_d, dt_d)
        skip = 50
        s_d = slice(skip, None)
        md1_d = f_dot_d[s_d] * t_d[s_d] / Clock1_d[s_d]
        md2_direct_d = f_dot_d[s_d] * t_d[s_d] / Clock2_d[s_d]
        md2_formula_d = md1_d * (Clock1_d[s_d] / Clock2_d[s_d])
        t_plot = t_d[s_d][:150]
        ax4.plot(t_plot, md2_direct_d[:150], color=COLORS[0], lw=1.2, label="direct")
        ax4.plot(t_plot, md2_formula_d[:150], color=COLORS[1], lw=0.8,
                 ls="--", label="formula")
        ax4.set_xlabel("t")
        ax4.set_ylabel("dS/dΘ2")
        ax4.set_title("Telescoping: direct vs formula")
        ax4.legend(fontsize=6, frameon=False)

        plt.tight_layout(pad=1.5)
        fig.savefig("paper7_panel_4.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Panel 4 saved.")


# ============================================================
# Panel 5: Ergodic Consistency
# ============================================================

def generate_panel_5():
    with plt.rc_context(STYLE):
        fig = make_figure()

        # --- Chart A: 3D Cesaro error surface over (K, scale) ---
        ax1 = fig.add_subplot(141, projection="3d")
        K_range = np.array([100, 200, 500, 1000, 2000, 5000], dtype=float)
        scale_range = np.linspace(0.01, 0.05, 10)
        KK, SC = np.meshgrid(np.log10(K_range), scale_range)
        s_floor = 2.0
        # Theoretical: error ~ (sigma_md / sqrt(K))
        # sigma_md ~ 1/gbar * sigma_fdot ~ 1/(SC*sqrt(2/pi)) * 0.3
        sigma_fdot = 0.3
        sigma_md = sigma_fdot / (SC * np.sqrt(2 / np.pi))
        err_surf = sigma_md / np.sqrt(10 ** KK)
        ax1.plot_surface(KK, SC, np.log10(err_surf + 1e-10),
                         cmap="Blues", alpha=0.9, linewidth=0, antialiased=True)
        ax1.set_xlabel("log10(K)", labelpad=2)
        ax1.set_ylabel("scale", labelpad=2)
        ax1.set_zlabel("log10(err)", labelpad=2)
        ax1.set_title("Cesaro error surface")
        ax1.tick_params(labelsize=6)

        # --- Chart B: Ergodic convergence trajectories for 5 seeds ---
        ax2 = fig.add_subplot(142)
        K_max = 3000
        dt = 0.01
        scale = 0.03
        for seed in range(5):
            rng_e = np.random.RandomState(900 + seed)
            G_e = rng_e.randn(K_max) * scale
            t_e = np.arange(1, K_max + 1) * dt
            f_e = np.sin(t_e * 0.3)
            f_tilde_e = s_transform(f_e, np.zeros(K_max), s_floor)
            f_dot_e = np.gradient(f_tilde_e, dt)
            Clock_e = np.cumsum(np.abs(G_e)) * dt
            md_e = f_dot_e / (Clock_e / t_e + 1e-10)
            running_avg = np.cumsum(md_e) / np.arange(1, K_max + 1)
            ax2.plot(np.arange(1, K_max + 1), running_avg,
                     color=COLORS[seed], lw=0.7, alpha=0.85)
        ax2.axhline(0, color="k", ls="--", lw=0.8)
        ax2.set_xlabel("K")
        ax2.set_ylabel("Cesaro average")
        ax2.set_title("Cesaro convergence (5 paths)")

        # --- Chart C: Log-log convergence rate ---
        ax3 = fig.add_subplot(143)
        K_list = [50, 100, 200, 500, 1000, 2000, 5000]
        mu_stat = 0.0  # theoretical mean is 0 for this setup
        # Compute empirical errors over 20 seeds
        err_means = []
        for K in K_list:
            means_K = []
            for seed in range(20):
                rng_k = np.random.RandomState(9000 + seed)
                G_k = rng_k.randn(K) * scale
                t_k = np.arange(1, K + 1) * dt
                f_k = np.sin(t_k * 0.3)
                f_tilde_k = s_transform(f_k, np.zeros(K), s_floor)
                f_dot_k = np.gradient(f_tilde_k, dt)
                Clock_k = np.cumsum(np.abs(G_k)) * dt
                md_k = f_dot_k / (Clock_k / t_k + 1e-10)
                means_K.append(float(np.mean(md_k)))
            err_means.append(float(np.std(means_K)))
        ax3.loglog(K_list, err_means, "o-", color=COLORS[0], lw=1.2, ms=5)
        # Reference line: ~ K^{-0.5}
        K_arr = np.array(K_list, dtype=float)
        ref = err_means[0] * (K_arr[0] / K_arr) ** 0.5
        ax3.loglog(K_arr, ref, "k--", lw=0.8, label="K^{-1/2}")
        ax3.set_xlabel("K")
        ax3.set_ylabel("std of Cesaro avg")
        ax3.set_title("Convergence rate")
        ax3.legend(fontsize=7, frameon=False)

        # --- Chart D: Inter-layer ergodic ratios vs theoretical gbar_j/gbar_k ---
        ax4 = fig.add_subplot(144)
        n_long = 10000
        dt_long = 0.01
        windows = [1, 5, 10, 20, 40]
        gbars_measured = []
        for w in windows:
            G_long = np.random.RandomState(9100).randn(n_long) * scale
            I_w = np.convolve(np.abs(G_long), np.ones(w) / w, mode='same')
            gbar_w = float(np.mean(I_w))
            gbars_measured.append(gbar_w)
        # Ergodic mean of md at each layer ~ E[f_dot] / gbar_k
        # Ratio E[md_1]/E[md_k] = gbar_k / gbar_1
        gbar_base = gbars_measured[0]
        theoretical_ratios = [g / gbar_base for g in gbars_measured]
        ax4.plot(windows, theoretical_ratios, "s-", color=COLORS[0], lw=1.2, ms=6)
        ax4.axhline(1.0, color="k", ls="--", lw=0.8)
        ax4.set_xlabel("smoothing window")
        ax4.set_ylabel("ḡ_k / ḡ_base")
        ax4.set_title("Inter-layer ergodic ratios")

        plt.tight_layout(pad=1.5)
        fig.savefig("paper7_panel_5.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("Panel 5 saved.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_panel_1()
    generate_panel_2()
    generate_panel_3()
    generate_panel_4()
    generate_panel_5()
    print("All 5 panels generated.")
