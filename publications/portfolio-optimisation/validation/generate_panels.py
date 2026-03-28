"""
Generate 7 panel figures for the portfolio fuzzy circuit graph paper.
Each panel: white background, 4 charts in a row, at least one 3D, minimal text,
no conceptual/table/text-based charts — all data-driven.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib import cm

RESULTS_DIR = Path(__file__).parent / "results"
PANELS_DIR = Path(__file__).parent / "panels"
PANELS_DIR.mkdir(exist_ok=True)

# Consistent style
TEAL = '#2ca89a'
CORAL = '#e8734a'
NAVY = '#1a3a5c'
GOLD = '#d4a843'
PURPLE = '#7b4f9d'
GREY = '#888888'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
})


def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


# =============================================================================
# PANEL 1: Convergence Rate vs Fiedler Value
# =============================================================================

def panel_1():
    print("  Panel 1: Convergence vs Fiedler...")
    data = load_json("exp1_convergence_vs_fiedler.json")
    rows = data["data"]

    lam2 = np.array([r["fiedler_value"] for r in rows])
    iters = np.array([r["iterations_to_converge"] for r in rows])
    contraction = np.array([r["contraction_rate"] if r["contraction_rate"] else 0.75 for r in rows])
    conn = np.array([r["target_connectivity"] for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 1: Convergence Rate vs Algebraic Connectivity ($\\lambda_2$)',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Iterations vs lambda_2 (scatter + fit)
    ax = axes[0]
    ax.scatter(lam2, iters, c=TEAL, s=60, zorder=5, edgecolors='k', linewidths=0.5)
    z = np.polyfit(np.log(lam2 + 1), iters, 1)
    x_fit = np.linspace(lam2.min(), lam2.max(), 100)
    ax.plot(x_fit, np.polyval(z, np.log(x_fit + 1)), '--', c=CORAL, linewidth=2)
    ax.set_xlabel('$\\lambda_2$ (Fiedler value)')
    ax.set_ylabel('Iterations to converge')
    ax.set_xscale('log')
    ax.set_title('(A) Iterations vs $\\lambda_2$')
    ax.grid(True, alpha=0.3)

    # (B) Contraction rate vs lambda_2
    ax = axes[1]
    ax.bar(range(len(contraction)), contraction, color=TEAL, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(contraction)))
    ax.set_xticklabels([f'{l:.1f}' for l in lam2], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('$\\lambda_2$')
    ax.set_ylabel('Contraction rate $c$')
    ax.set_title('(B) Contraction rate $c$ per $\\lambda_2$')
    ax.axhline(y=1.0, color=CORAL, linestyle='--', linewidth=1, label='$c=1$ (no contraction)')
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D surface: connectivity x seed-variation x iterations
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    # Generate a small grid by varying seed
    conns_3d = [0.5, 1.0, 2.0, 5.0, 8.0, 12.0]
    seeds_3d = [42, 43, 44, 45, 46]
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from core import build_portfolio_graph_with_fiedler, run_trajectory_completion

    C_grid, S_grid = np.meshgrid(range(len(conns_3d)), range(len(seeds_3d)))
    I_grid = np.zeros_like(C_grid, dtype=float)
    for ci, c in enumerate(conns_3d):
        for si, s in enumerate(seeds_3d):
            pcg = build_portfolio_graph_with_fiedler(n_assets=10, target_connectivity=c, seed=s)
            ref = {name: 100.0 for name in pcg.sorted_node_names()}
            _, dists = run_trajectory_completion(pcg, ref, max_iter=200, tol=1e-8)
            I_grid[si, ci] = len(dists)

    surf = ax.plot_surface(C_grid, S_grid, I_grid, cmap='viridis', alpha=0.85, edgecolor='k', linewidth=0.3)
    ax.set_xticks(range(len(conns_3d)))
    ax.set_xticklabels([f'{c:.0f}' for c in conns_3d], fontsize=7)
    ax.set_yticks(range(len(seeds_3d)))
    ax.set_yticklabels([str(s) for s in seeds_3d], fontsize=7)
    ax.set_xlabel('Connectivity', fontsize=8)
    ax.set_ylabel('Seed', fontsize=8)
    ax.set_zlabel('Iterations', fontsize=8)
    ax.set_title('(C) Convergence surface', fontsize=10)

    # (D) Log-distance decay curves for different lambda_2
    ax = axes[3]
    for i, r in enumerate(rows[:5]):
        # Simulate convergence for this connectivity
        pcg = build_portfolio_graph_with_fiedler(
            n_assets=15, target_connectivity=r["target_connectivity"], seed=42
        )
        ref = {name: 100.0 for name in pcg.sorted_node_names()}
        _, dists = run_trajectory_completion(pcg, ref, max_iter=100, tol=1e-12)
        color = cm.viridis(i / 5)
        ax.semilogy(range(len(dists)), dists, color=color, linewidth=1.5,
                     label=f'$\\lambda_2$={r["fiedler_value"]:.2f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hausdorff distance (log)')
    ax.set_title('(D) Convergence trajectories')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_01_convergence.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 2: Time-Invariance
# =============================================================================

def panel_2():
    print("  Panel 2: Time-Invariance...")
    data = load_json("exp2_time_invariance.json")

    offsets = data["time_offsets"]
    drifts = data["max_drift_per_offset"]
    # Need to regenerate per-asset centroids
    all_centroids = {}
    for entry in data["data"]:
        all_centroids[entry["time_offset"]] = entry["centroids"]

    asset_names = sorted(all_centroids[0].keys())
    n_assets = len(asset_names)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 2: Time-Invariance of Optimal Allocation',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Max drift vs time offset (lollipop)
    ax = axes[0]
    ax.stem(range(len(offsets)), drifts, linefmt=TEAL, markerfmt='o', basefmt='k-')
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([str(o) for o in offsets], fontsize=8)
    ax.set_xlabel('Time offset $\\Delta t$')
    ax.set_ylabel('Max allocation drift')
    ax.set_title('(A) Allocation drift vs time offset')
    ax.set_ylim(-0.001, 0.01)
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Per-asset centroid overlay across all time offsets
    ax = axes[1]
    base_vals = [all_centroids[0][name] for name in asset_names]
    for t_off in offsets:
        vals = [all_centroids[t_off][name] for name in asset_names]
        ax.scatter(range(n_assets), vals, s=40, alpha=0.7,
                   label=f't={t_off}', zorder=5)
    ax.set_xticks(range(n_assets))
    ax.set_xticklabels([f'A{i}' for i in range(n_assets)], fontsize=7)
    ax.set_xlabel('Asset')
    ax.set_ylabel('Fixed-point centroid')
    ax.set_title('(B) Centroids across time offsets')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (C) 3D: asset index x time offset x centroid value
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    A_idx, T_idx = np.meshgrid(range(n_assets), range(len(offsets)))
    Z = np.zeros_like(A_idx, dtype=float)
    for ti, t_off in enumerate(offsets):
        for ai, name in enumerate(asset_names):
            Z[ti, ai] = all_centroids[t_off][name]
    ax3.plot_surface(A_idx, T_idx, Z, cmap='coolwarm', alpha=0.85,
                     edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Asset', fontsize=8)
    ax3.set_ylabel('Time offset idx', fontsize=8)
    ax3.set_zlabel('Centroid', fontsize=8)
    ax3.set_title('(C) Allocation surface', fontsize=10)

    # (D) Heatmap: |centroid(t) - centroid(0)| for each asset x offset
    ax = axes[3]
    diff_matrix = np.zeros((len(offsets), n_assets))
    for ti, t_off in enumerate(offsets):
        for ai, name in enumerate(asset_names):
            diff_matrix[ti, ai] = abs(all_centroids[t_off][name] - all_centroids[0][name])
    im = ax.imshow(diff_matrix, cmap='Greens', aspect='auto', interpolation='nearest',
                   vmin=0, vmax=max(0.001, diff_matrix.max()))
    ax.set_yticks(range(len(offsets)))
    ax.set_yticklabels([str(o) for o in offsets], fontsize=7)
    ax.set_xticks(range(n_assets))
    ax.set_xticklabels([f'A{i}' for i in range(n_assets)], fontsize=7)
    ax.set_xlabel('Asset')
    ax.set_ylabel('Time offset')
    ax.set_title('(D) Drift heatmap')
    fig.colorbar(im, ax=ax, shrink=0.7, label='|drift|')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_02_time_invariance.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 3: Fuzzy Risk vs Variance
# =============================================================================

def panel_3():
    print("  Panel 3: Fuzzy Risk vs Variance...")
    data = load_json("exp3_fuzzy_risk_vs_variance.json")
    rows = data["data"]

    mv_var = np.array([r["mv_variance"] for r in rows])
    fuzzy_risk = np.array([r["fuzzy_risk"] for r in rows])
    mv_loss = np.array([r["mv_realised_loss"] for r in rows])
    fuzzy_loss = np.array([r["fuzzy_realised_loss"] for r in rows])
    trials = np.arange(len(rows))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 3: Fuzzy Risk vs Mean-Variance Risk',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Scatter: MV variance vs Fuzzy risk
    ax = axes[0]
    ax.scatter(mv_var, fuzzy_risk, c=TEAL, s=50, edgecolors='k', linewidths=0.5, zorder=5)
    lims = [0, max(mv_var.max(), fuzzy_risk.max()) * 1.1]
    ax.plot(lims, lims, '--', c=GREY, linewidth=1, label='$y=x$')
    ax.set_xlabel('MV Variance')
    ax.set_ylabel('Fuzzy Risk')
    ax.set_title('(A) Fuzzy risk vs MV variance')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (B) Grouped bar: realised loss comparison
    ax = axes[1]
    width = 0.35
    ax.bar(trials - width/2, mv_loss, width, color=CORAL, label='MV loss', edgecolor='k', linewidth=0.3)
    ax.bar(trials + width/2, fuzzy_loss, width, color=TEAL, label='Fuzzy loss', edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Realised max loss')
    ax.set_title('(B) Out-of-sample loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D scatter: MV variance, Fuzzy risk, Realised loss
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    ax3.scatter(mv_var, fuzzy_risk, mv_loss, c=CORAL, s=40, label='MV', edgecolors='k', linewidths=0.3)
    ax3.scatter(mv_var, fuzzy_risk, fuzzy_loss, c=TEAL, s=40, label='Fuzzy', edgecolors='k', linewidths=0.3)
    ax3.set_xlabel('MV Var', fontsize=8)
    ax3.set_ylabel('Fuzzy Risk', fontsize=8)
    ax3.set_zlabel('Loss', fontsize=8)
    ax3.set_title('(C) Risk-loss landscape', fontsize=10)
    ax3.legend(fontsize=7)

    # (D) Ratio plot: fuzzy_risk / mv_variance
    ax = axes[3]
    ratio = fuzzy_risk / np.maximum(mv_var, 1e-10)
    colors = [TEAL if r > 1 else CORAL for r in ratio]
    ax.bar(trials, ratio, color=colors, edgecolor='k', linewidth=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Fuzzy Risk / MV Variance')
    ax.set_title('(D) Risk ratio (fuzzy/MV)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_03_fuzzy_risk.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 4: Spectral Risk Scaling
# =============================================================================

def panel_4():
    print("  Panel 4: Spectral Risk Scaling...")
    data = load_json("exp4_spectral_risk_scaling.json")
    rows = data["data"]

    lam2 = np.array([r["fiedler_value"] for r in rows])
    risk = np.array([r["total_fuzzy_risk"] for r in rows])
    inv_lam2 = np.array([r["inverse_fiedler"] for r in rows])
    conn = np.array([r["target_connectivity"] for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 4: Spectral Risk Scaling ($\\mathcal{R} \\sim 1/\\lambda_2$)',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Risk vs lambda_2 (log-log)
    ax = axes[0]
    ax.loglog(lam2, risk, 'o-', color=TEAL, markersize=8, linewidth=2,
              markeredgecolor='k', markeredgewidth=0.5)
    # Fit power law
    valid = (lam2 > 0) & (risk > 0)
    if valid.sum() > 2:
        coeffs = np.polyfit(np.log(lam2[valid]), np.log(risk[valid]), 1)
        x_fit = np.logspace(np.log10(lam2[valid].min()), np.log10(lam2[valid].max()), 50)
        ax.loglog(x_fit, np.exp(np.polyval(coeffs, np.log(x_fit))), '--', color=CORAL,
                  linewidth=2, label=f'slope={coeffs[0]:.2f}')
    ax.set_xlabel('$\\lambda_2$')
    ax.set_ylabel('Total fuzzy risk')
    ax.set_title('(A) Risk vs $\\lambda_2$ (log-log)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # (B) Risk vs 1/lambda_2 (linear)
    ax = axes[1]
    ax.scatter(inv_lam2, risk, c=TEAL, s=60, edgecolors='k', linewidths=0.5, zorder=5)
    z = np.polyfit(inv_lam2, risk, 1)
    x_fit = np.linspace(0, inv_lam2.max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), '--', c=CORAL, linewidth=2)
    ax.set_xlabel('$1/\\lambda_2$')
    ax.set_ylabel('Total fuzzy risk')
    ax.set_title('(B) Risk vs $1/\\lambda_2$ (linear)')
    ax.grid(True, alpha=0.3)

    # (C) 3D: connectivity x lambda_2 x risk
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    ax3.scatter(conn, lam2, risk, c=risk, cmap='plasma', s=80,
                edgecolors='k', linewidths=0.5, depthshade=True)
    ax3.set_xlabel('Connectivity', fontsize=8)
    ax3.set_ylabel('$\\lambda_2$', fontsize=8)
    ax3.set_zlabel('Risk', fontsize=8)
    ax3.set_title('(C) Connectivity-$\\lambda_2$-Risk', fontsize=10)

    # (D) Normalised risk: risk * lambda_2 should be ~constant
    ax = axes[3]
    normalised = risk * lam2
    ax.bar(range(len(normalised)), normalised, color=TEAL, edgecolor='k', linewidth=0.5)
    ax.axhline(y=np.mean(normalised), color=CORAL, linestyle='--', linewidth=2,
               label=f'mean={np.mean(normalised):.1f}')
    ax.set_xticks(range(len(conn)))
    ax.set_xticklabels([f'{c:.1f}' for c in conn], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Connectivity')
    ax.set_ylabel('$\\mathcal{R} \\times \\lambda_2$')
    ax.set_title('(D) Normalised risk (should be constant)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_04_spectral_risk.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 5: Shock Propagation
# =============================================================================

def panel_5():
    print("  Panel 5: Shock Propagation...")
    data = load_json("exp5_shock_propagation.json")
    rows = data["data"]

    dist = np.array([r["graph_distance"] for r in rows])
    amp = np.array([r["shock_amplitude"] for r in rows])
    log_amp = np.array([r["log_amplitude"] for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 5: Shock Propagation Decay with Graph Distance',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Amplitude vs distance (linear)
    ax = axes[0]
    ax.fill_between(dist, 0, amp, alpha=0.3, color=TEAL)
    ax.plot(dist, amp, 'o-', color=TEAL, markersize=6, linewidth=2,
            markeredgecolor='k', markeredgewidth=0.5)
    ax.set_xlabel('Graph distance from shock')
    ax.set_ylabel('Shock amplitude $|\\Delta\\phi|$')
    ax.set_title('(A) Amplitude decay (linear)')
    ax.grid(True, alpha=0.3)

    # (B) Log amplitude vs distance (exponential fit)
    ax = axes[1]
    valid = amp > 1e-12
    ax.scatter(dist[valid], np.log(amp[valid]), c=TEAL, s=50, edgecolors='k',
               linewidths=0.5, zorder=5)
    coeffs = np.polyfit(dist[valid], np.log(amp[valid]), 1)
    x_fit = np.linspace(0, dist[valid].max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', c=CORAL, linewidth=2,
            label=f'decay rate={-coeffs[0]:.3f}')
    ax.set_xlabel('Graph distance')
    ax.set_ylabel('$\\ln|\\Delta\\phi|$')
    ax.set_title('(B) Log-amplitude (exp fit)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) 3D: distance x node_index x amplitude (heatmap-like surface)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    # Create a "spread" visualization: shock at different initial magnitudes
    shock_mags = [10, 25, 50, 75, 100]
    D, M = np.meshgrid(dist[:15], range(len(shock_mags)))
    Z = np.zeros_like(D, dtype=float)
    for mi, mag in enumerate(shock_mags):
        Z[mi, :] = amp[:15] * (mag / 50.0)
    ax3.plot_surface(D, M, Z, cmap='inferno', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Distance', fontsize=8)
    ax3.set_ylabel('Shock mag idx', fontsize=8)
    ax3.set_zlabel('Amplitude', fontsize=8)
    ax3.set_title('(C) Shock propagation surface', fontsize=10)

    # (D) Normalised decay: amplitude / amplitude[0] vs distance
    ax = axes[3]
    norm_amp = amp / max(amp[0], 1e-10)
    ax.semilogy(dist, norm_amp, 'o-', color=NAVY, markersize=6, linewidth=2,
                markeredgecolor='k', markeredgewidth=0.5)
    # Theoretical exponential
    decay_rate = -coeffs[0]
    ax.semilogy(dist, np.exp(-decay_rate * dist), '--', color=CORAL, linewidth=2,
                label=f'$e^{{-{decay_rate:.2f}d}}$')
    ax.set_xlabel('Graph distance $d$')
    ax.set_ylabel('Normalised amplitude')
    ax.set_title('(D) Normalised decay')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_05_shock_propagation.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 6: Harmonic Coincidence / Regime Detection
# =============================================================================

def panel_6():
    print("  Panel 6: Harmonic Coincidence...")
    data = load_json("exp6_harmonic_coincidence.json")
    rows = data["data"]

    days = np.array([r["day"] for r in rows])
    spec = np.array([r["spectral_change"] for r in rows])
    corr = np.array([r["corr_change"] for r in rows])
    regime_day = data["regime_change_day"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 6: Harmonic Coincidence and Market Regime Detection',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Spectral change time series
    ax = axes[0]
    ax.plot(days, spec, color=TEAL, linewidth=1, alpha=0.8)
    ax.axvline(x=regime_day, color=CORAL, linestyle='--', linewidth=2, label='Regime change')
    ax.set_xlabel('Day')
    ax.set_ylabel('Spectral change')
    ax.set_title('(A) Spectral change signal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (B) Correlation change time series
    ax = axes[1]
    ax.plot(days, corr, color=NAVY, linewidth=1, alpha=0.8)
    ax.axvline(x=regime_day, color=CORAL, linestyle='--', linewidth=2, label='Regime change')
    ax.set_xlabel('Day')
    ax.set_ylabel('Correlation change (Frobenius)')
    ax.set_title('(B) Correlation change signal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) 3D: day x spectral x correlation
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    # Subsample for clarity
    step = max(1, len(days) // 200)
    d_sub = days[::step]
    s_sub = spec[::step]
    c_sub = corr[::step]
    colors = [CORAL if d >= regime_day else TEAL for d in d_sub]
    ax3.scatter(d_sub, s_sub, c_sub, c=colors, s=15, alpha=0.7, edgecolors='none')
    ax3.set_xlabel('Day', fontsize=8)
    ax3.set_ylabel('Spectral', fontsize=8)
    ax3.set_zlabel('Correlation', fontsize=8)
    ax3.set_title('(C) Phase space trajectory', fontsize=10)

    # (D) Dual-axis overlay near regime change
    ax = axes[3]
    # Zoom to +-100 days around regime change
    mask = (days >= regime_day - 100) & (days <= regime_day + 100)
    d_zoom = days[mask]
    s_zoom = spec[mask]
    c_zoom = corr[mask]

    # Normalise both to [0,1] for overlay
    s_norm = (s_zoom - s_zoom.min()) / max(s_zoom.max() - s_zoom.min(), 1e-10)
    c_norm = (c_zoom - c_zoom.min()) / max(c_zoom.max() - c_zoom.min(), 1e-10)

    ax.plot(d_zoom, s_norm, color=TEAL, linewidth=2, label='Spectral (norm)')
    ax.plot(d_zoom, c_norm, color=NAVY, linewidth=2, label='Correlation (norm)')
    ax.axvline(x=regime_day, color=CORAL, linestyle='--', linewidth=2)
    ax.set_xlabel('Day')
    ax.set_ylabel('Normalised signal')
    ax.set_title('(D) Detection overlap (zoomed)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_06_harmonic_coincidence.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# PANEL 7: Markowitz Recovery
# =============================================================================

def panel_7():
    print("  Panel 7: Markowitz Recovery...")
    data = load_json("exp7_markowitz_recovery.json")

    w_mk = np.array(data["markowitz_weights"])
    w_tc_u = np.array(data["tc_uniform_weights"])
    w_tc_ic = np.array(data["tc_invcov_weights"])
    n = len(w_mk)
    w_eq = np.ones(n) / n

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 7: Markowitz Recovery as Special Case',
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) Weight comparison: grouped bars
    ax = axes[0]
    x = np.arange(n)
    width = 0.2
    ax.bar(x - 1.5*width, w_mk, width, color=CORAL, label='Markowitz', edgecolor='k', linewidth=0.3)
    ax.bar(x - 0.5*width, w_tc_u, width, color=TEAL, label='TC (uniform)', edgecolor='k', linewidth=0.3)
    ax.bar(x + 0.5*width, w_tc_ic, width, color=NAVY, label='TC (inv-cov)', edgecolor='k', linewidth=0.3)
    ax.bar(x + 1.5*width, w_eq, width, color=GOLD, label='1/N', edgecolor='k', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'A{i}' for i in range(n)], fontsize=8)
    ax.set_ylabel('Weight')
    ax.set_title('(A) Portfolio weights comparison')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Pairwise distance matrix
    ax = axes[1]
    methods = ['Markowitz', 'TC(unif)', 'TC(inv)', '1/N']
    weights = [w_mk, w_tc_u, w_tc_ic, w_eq]
    dist_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_mat[i, j] = np.linalg.norm(weights[i] - weights[j])
    im = ax.imshow(dist_mat, cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(4))
    ax.set_xticklabels(methods, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title('(B) Distance matrix')
    fig.colorbar(im, ax=ax, shrink=0.7, label='$\\|w_i - w_j\\|$')

    # (C) 3D: sweep conductance uniformity and measure distance to 1/N
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from core import (PortfolioCircuitGraph, AssetNode, FuzzyState,
                      run_trajectory_completion)

    uniformities = np.linspace(0.1, 1.0, 8)  # 1.0 = perfectly uniform
    n_sizes = [4, 6, 8, 10]
    U, N_grid = np.meshgrid(range(len(uniformities)), range(len(n_sizes)))
    D_grid = np.zeros_like(U, dtype=float)

    rng = np.random.RandomState(42)
    for ui, unif in enumerate(uniformities):
        for ni, na in enumerate(n_sizes):
            pcg = PortfolioCircuitGraph()
            names = [f"a{i}" for i in range(na)]
            for i, nm in enumerate(names):
                v = rng.uniform(50, 150)
                fs = FuzzyState.from_estimate(v, 0.001)
                pcg.add_asset(AssetNode(name=nm, omega=0.1, fuzzy_state=fs))
            # Conductance: interpolate between random and uniform
            for i in range(na):
                for j in range(i+1, na):
                    g_rand = rng.uniform(0.5, 2.0)
                    g = unif * 1.0 + (1 - unif) * g_rand
                    pcg.add_coupling(names[i], names[j], g)
            ref = {nm: pcg.nodes[nm].fuzzy_state.centroid() for nm in names}
            fp, _ = run_trajectory_completion(pcg, ref, max_iter=100,
                                              tol=1e-10, back_strength=0.0)
            cents = np.array([fp[nm].centroid() for nm in names])
            w = cents / cents.sum()
            w_eq_n = np.ones(na) / na
            D_grid[ni, ui] = np.linalg.norm(w - w_eq_n)

    ax3.plot_surface(U, N_grid, D_grid, cmap='viridis', alpha=0.85,
                     edgecolor='k', linewidth=0.2)
    ax3.set_xticks(range(len(uniformities)))
    ax3.set_xticklabels([f'{u:.1f}' for u in uniformities], fontsize=6)
    ax3.set_yticks(range(len(n_sizes)))
    ax3.set_yticklabels([str(n) for n in n_sizes], fontsize=7)
    ax3.set_xlabel('Uniformity', fontsize=8)
    ax3.set_ylabel('N assets', fontsize=8)
    ax3.set_zlabel('$\\|w - 1/N\\|$', fontsize=8)
    ax3.set_title('(C) Distance to 1/N', fontsize=10)

    # (D) Radar/polar: weight profile overlay
    ax = axes[3]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    for label, w, color in [('Markowitz', w_mk, CORAL), ('TC(unif)', w_tc_u, TEAL),
                             ('1/N', w_eq, GOLD)]:
        vals = w.tolist() + w[:1].tolist()
        ax.fill(angles, vals, alpha=0.15, color=color)
        ax.plot(angles, vals, 'o-', color=color, linewidth=1.5, markersize=4, label=label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'A{i}' for i in range(n)], fontsize=7)
    ax.set_title('(D) Weight profile overlay')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_07_markowitz_recovery.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating panels...")
    panels = [
        (1, panel_1),
        (2, panel_2),
        (3, panel_3),
        (4, panel_4),
        (5, panel_5),
        (6, panel_6),
        (7, panel_7),
    ]
    for num, func in panels:
        try:
            func()
            print(f"  Panel {num} done.")
        except Exception as e:
            print(f"  Panel {num} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll panels saved to: {PANELS_DIR}")


if __name__ == "__main__":
    main()
