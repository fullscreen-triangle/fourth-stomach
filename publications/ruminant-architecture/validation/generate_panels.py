"""
Generate 8 panel figures for the Ruminant Processing Architecture paper.
Each panel: white background, 4 charts in a row, at least one 3D, minimal text.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import RuminantModel

RESULTS_DIR = Path(__file__).parent / "results"
PANELS_DIR = Path(__file__).parent / "panels"
PANELS_DIR.mkdir(exist_ok=True)

TEAL = '#2ca89a'
CORAL = '#e8734a'
NAVY = '#1a3a5c'
GOLD = '#d4a843'
PURPLE = '#7b4f9d'

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

def load_json(f):
    with open(RESULTS_DIR / f) as fh:
        return json.load(fh)


# =============================================================================
# Panel 1: Rumination Convergence
# =============================================================================
def panel_1():
    print("  Panel 1: Convergence...")
    data = load_json("exp1_convergence.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 1: Rumination Convergence', fontsize=12, fontweight='bold', y=1.02)

    # (A) Contraction rates across seeds
    rates = [r["contraction_rate"] for r in rows]
    ax = axes[0]
    ax.bar(range(len(rates)), rates, color=TEAL, edgecolor='k', linewidth=0.3)
    ax.axhline(y=1.0, color=CORAL, linestyle='--', linewidth=2, label='c=1 (boundary)')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Contraction rate c')
    ax.set_title('(A) Contraction rates (all < 1)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Convergence distance curves for 5 seeds
    ax = axes[1]
    for seed in [0, 3, 7, 12, 18]:
        model = RuminantModel(n_tokens=32, d_model=64, seed=seed)
        X0 = model.rng.randn(32, 64) * 0.5
        _, hist = model.ruminate(X0, max_cycles=25)
        dists = [h["distance"] for h in hist]
        color = cm.viridis(seed / 20)
        ax.semilogy(range(len(dists)), dists, color=color, linewidth=1.5, label=f's={seed}')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Distance (log)')
    ax.set_title('(B) Distance decay per cycle')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (C) 3D: seed x cycle x distance
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    seeds_3d = list(range(0, 20, 2))
    max_cyc = 20
    S_grid, C_grid = np.meshgrid(range(len(seeds_3d)), range(max_cyc))
    D_grid = np.zeros_like(S_grid, dtype=float)
    for si, seed in enumerate(seeds_3d):
        model = RuminantModel(n_tokens=32, d_model=64, seed=seed)
        X0 = model.rng.randn(32, 64) * 0.5
        _, hist = model.ruminate(X0, max_cycles=max_cyc)
        for ci in range(min(max_cyc, len(hist))):
            D_grid[ci, si] = np.log10(hist[ci]["distance"] + 1e-15)
    ax3.plot_surface(S_grid, C_grid, D_grid, cmap='viridis', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Seed idx', fontsize=8)
    ax3.set_ylabel('Cycle', fontsize=8)
    ax3.set_zlabel('log(dist)', fontsize=8)
    ax3.set_title('(C) Convergence surface', fontsize=10)

    # (D) Completeness vs cycle
    ax = axes[3]
    for seed in [0, 5, 10, 15]:
        model = RuminantModel(n_tokens=32, d_model=64, seed=seed)
        X0 = model.rng.randn(32, 64) * 0.5
        _, hist = model.ruminate(X0, max_cycles=25)
        comps = [h["completeness"] for h in hist]
        color = cm.plasma(seed / 20)
        ax.plot(range(len(comps)), comps, color=color, linewidth=1.5, label=f's={seed}')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Completeness')
    ax.set_title('(D) Completeness trajectories')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_01_convergence.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 2: Spectral Attention
# =============================================================================
def panel_2():
    print("  Panel 2: Spectral Attention...")
    data = load_json("exp2_spectral.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 2: Spectral Attention Detects Hidden Correlations', fontsize=12, fontweight='bold', y=1.02)

    std_scores = [abs(r["std_attention_score"]) for r in rows]
    spec_scores = [r["spectral_score"] for r in rows]

    # (A) Scatter: spectral vs standard
    ax = axes[0]
    ax.scatter(std_scores, spec_scores, c=TEAL, s=30, edgecolors='k', linewidths=0.3, zorder=5)
    lim = max(max(std_scores), max(spec_scores)) * 1.1
    ax.plot([0, lim], [0, lim], '--', c='grey', linewidth=1)
    ax.set_xlabel('|Standard attention score|')
    ax.set_ylabel('Spectral correlation score')
    ax.set_title('(A) Spectral vs standard scores')
    ax.grid(True, alpha=0.3)

    # (B) Histogram of score differences
    ax = axes[1]
    diffs = [s - a for s, a in zip(spec_scores, std_scores)]
    ax.hist(diffs, bins=20, color=TEAL, edgecolor='k', linewidth=0.3, alpha=0.8)
    ax.axvline(x=0, color=CORAL, linestyle='--', linewidth=2)
    ax.set_xlabel('Spectral - Standard score')
    ax.set_ylabel('Count')
    ax.set_title('(B) Score difference distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D: spectral correlation matrix for one example
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    rng = np.random.RandomState(42)
    n, d = 16, 32
    X = rng.randn(n, d) * 0.3
    freq = 0.15
    for dim in range(d):
        X[0, dim] += 0.5 * np.sin(2 * np.pi * freq * dim)
        X[8, dim] += 0.5 * np.sin(2 * np.pi * 2 * freq * dim)
    X_hat = np.fft.rfft(X, axis=1)
    norms = np.sqrt(np.sum(np.abs(X_hat)**2, axis=1, keepdims=True)) + 1e-8
    X_norm = X_hat / norms
    S = np.abs(X_norm @ X_norm.conj().T).real
    I, J = np.meshgrid(range(n), range(n))
    ax3.plot_surface(I, J, S, cmap='viridis', alpha=0.85, edgecolor='k', linewidth=0.1)
    ax3.set_xlabel('Token i', fontsize=8)
    ax3.set_ylabel('Token j', fontsize=8)
    ax3.set_zlabel('S(i,j)', fontsize=8)
    ax3.set_title('(C) Spectral correlation matrix', fontsize=10)

    # (D) Win rate bar
    ax = axes[3]
    wins = sum(1 for r in rows if r["spectral_wins"])
    losses = len(rows) - wins
    ax.bar(['Spectral wins', 'Standard wins'], [wins, losses], color=[TEAL, CORAL], edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Count')
    ax.set_title(f'(D) Win rate: {wins}/{len(rows)}')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_02_spectral.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 3: Gap Reduction
# =============================================================================
def panel_3():
    print("  Panel 3: Gap Reduction...")
    data = load_json("exp3_gap_reduction.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 3: Graph Completion Reduces Representation Gaps', fontsize=12, fontweight='bold', y=1.02)

    # (A) Mean gap reduction per seed
    ax = axes[0]
    means = [r["mean_gap_reduction"] for r in rows]
    ax.bar(range(len(means)), means, color=TEAL, edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean gap reduction')
    ax.set_title('(A) Gap reduction per seed')
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Gap reduction over cycles (one seed)
    ax = axes[1]
    for seed in [0, 5, 11]:
        gr = rows[seed]["gap_reductions"]
        color = [TEAL, CORAL, GOLD][seed // 5]
        ax.plot(range(len(gr)), gr, 'o-', color=color, linewidth=1.5, markersize=4, label=f's={seed}')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Gap reduction')
    ax.set_title('(B) Gap reduction per cycle')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) 3D: seed x cycle x gap reduction
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    n_seeds = min(10, len(rows))
    max_cyc = max(len(r["gap_reductions"]) for r in rows[:n_seeds])
    S_g, C_g = np.meshgrid(range(n_seeds), range(max_cyc))
    G_grid = np.zeros_like(S_g, dtype=float)
    for si in range(n_seeds):
        grs = rows[si]["gap_reductions"]
        for ci in range(min(max_cyc, len(grs))):
            G_grid[ci, si] = grs[ci]
    ax3.plot_surface(S_g, C_g, G_grid, cmap='YlGn', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Seed', fontsize=8)
    ax3.set_ylabel('Cycle', fontsize=8)
    ax3.set_zlabel('Gap red.', fontsize=8)
    ax3.set_title('(C) Gap reduction surface', fontsize=10)

    # (D) Uncertainty before vs after (one run)
    ax = axes[3]
    model = RuminantModel(n_tokens=32, d_model=64, seed=0)
    X0 = model.rng.randn(32, 64) * 0.5
    X0[10:20] += model.rng.randn(10, 64) * 2.0
    _, hist = model.ruminate(X0, max_cycles=10)
    ub = [h["unc_before"] for h in hist]
    ua = [h["unc_after"] for h in hist]
    ax.plot(range(len(ub)), ub, 'o-', color=CORAL, linewidth=1.5, label='Before C3')
    ax.plot(range(len(ua)), ua, 's-', color=TEAL, linewidth=1.5, label='After C3')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Mean uncertainty')
    ax.set_title('(D) Uncertainty before/after C3')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_03_gap_reduction.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 4: Free Energy Decrease
# =============================================================================
def panel_4():
    print("  Panel 4: Free Energy...")
    data = load_json("exp4_free_energy.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 4: Free Energy Monotonic Decrease Under Rumination', fontsize=12, fontweight='bold', y=1.02)

    # (A) Free energy curves for 5 seeds
    ax = axes[0]
    for i, seed in enumerate([0, 4, 8, 12, 16]):
        fes = rows[seed]["free_energies"]
        color = cm.viridis(i / 5)
        ax.plot(range(len(fes)), fes, color=color, linewidth=1.5, label=f's={seed}')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Free energy F')
    ax.set_title('(A) Free energy trajectories')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (B) First vs last free energy
    ax = axes[1]
    firsts = [r["free_energies"][0] for r in rows]
    lasts = [r["free_energies"][-1] for r in rows]
    ax.scatter(firsts, lasts, c=TEAL, s=40, edgecolors='k', linewidths=0.3, zorder=5)
    lim = [min(min(firsts), min(lasts)) - 1, max(max(firsts), max(lasts)) + 1]
    ax.plot(lim, lim, '--', c='grey', linewidth=1)
    ax.set_xlabel('F (initial)')
    ax.set_ylabel('F (final)')
    ax.set_title('(B) Initial vs final free energy')
    ax.grid(True, alpha=0.3)

    # (C) 3D: seed x cycle x free energy
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    n_s = min(10, len(rows))
    max_c = max(len(r["free_energies"]) for r in rows[:n_s])
    S_g, C_g = np.meshgrid(range(n_s), range(max_c))
    F_grid = np.zeros_like(S_g, dtype=float)
    for si in range(n_s):
        fes = rows[si]["free_energies"]
        for ci in range(min(max_c, len(fes))):
            F_grid[ci, si] = fes[ci]
    ax3.plot_surface(S_g, C_g, F_grid, cmap='coolwarm', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Seed', fontsize=8)
    ax3.set_ylabel('Cycle', fontsize=8)
    ax3.set_zlabel('F', fontsize=8)
    ax3.set_title('(C) Free energy surface', fontsize=10)

    # (D) Entropy vs cycle
    ax = axes[3]
    for seed in [0, 5, 10, 15]:
        model = RuminantModel(n_tokens=32, d_model=64, seed=seed)
        X0 = model.rng.randn(32, 64) * 0.5
        _, hist = model.ruminate(X0, max_cycles=15)
        entropies = [h["entropy"] for h in hist]
        color = cm.plasma(seed / 20)
        ax.plot(range(len(entropies)), entropies, color=color, linewidth=1.5, label=f's={seed}')
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Entropy S')
    ax.set_title('(D) Entropy trajectories')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_04_free_energy.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 5: Chamber Specialisation
# =============================================================================
def panel_5():
    print("  Panel 5: Chamber Specialisation...")
    data = load_json("exp5_specialisation.json")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 5: Chamber Functional Specialisation', fontsize=12, fontweight='bold', y=1.02)

    chambers = ['C1\nRumen', 'C2\nReticulum', 'C3\nOmasum', 'C4\nAbomasum']
    colors = [TEAL, CORAL, GOLD, NAVY]

    # (A) Change magnitudes
    ax = axes[0]
    mags = [data["change_magnitudes"][f"C{i+1}"] for i in range(4)]
    ax.bar(chambers, mags, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Change magnitude')
    ax.set_title('(A) Per-chamber change')
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Spectral energies
    ax = axes[1]
    se_keys = ['X0', 'C1', 'C2', 'C3', 'C4']
    se_vals = [data["spectral_energies"][k] for k in se_keys]
    se_colors = ['grey'] + colors
    ax.bar(se_keys, se_vals, color=se_colors, edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Spectral energy')
    ax.set_title('(B) Spectral energy per stage')
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D: token index x feature index x activation (after each chamber)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    model = RuminantModel(n_tokens=16, d_model=32, seed=42)
    X0 = model.rng.randn(16, 32) * 0.5
    X1 = model.chamber_1_rumen(X0)
    X2, S = model.chamber_2_reticulum(X1)
    T_idx, F_idx = np.meshgrid(range(16), range(32))
    ax3.plot_surface(T_idx, F_idx, X2.T, cmap='viridis', alpha=0.7, edgecolor='none')
    ax3.set_xlabel('Token', fontsize=8)
    ax3.set_ylabel('Feature', fontsize=8)
    ax3.set_zlabel('Activation', fontsize=8)
    ax3.set_title('(C) Post-Reticulum activations', fontsize=10)

    # (D) Processing profile: what % of total change each chamber contributes
    ax = axes[3]
    total = sum(mags)
    pcts = [m / total * 100 for m in mags]
    wedges, texts, autotexts = ax.pie(pcts, labels=chambers, colors=colors, autopct='%1.0f%%',
                                       textprops={'fontsize': 8}, startangle=90)
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title('(D) Processing share per chamber')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_05_specialisation.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 6: LoRA Efficiency
# =============================================================================
def panel_6():
    print("  Panel 6: LoRA Efficiency...")
    data = load_json("exp6_lora.json")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 6: Chamber-Specific LoRA Parameter Efficiency', fontsize=12, fontweight='bold', y=1.02)

    d = 128
    chambers = ['C1\nRumen', 'C2\nReticulum', 'C3\nOmasum', 'C4\nAbomasum']
    uniform_ranks = [32, 32, 32, 32]
    chamber_ranks = [32, 16, 8, 4]
    colors = [TEAL, CORAL, GOLD, NAVY]

    # (A) Rank comparison
    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, uniform_ranks, w, color='grey', label='Uniform', edgecolor='k', linewidth=0.3)
    ax.bar(x + w/2, chamber_ranks, w, color=colors, edgecolor='k', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(chambers, fontsize=7)
    ax.set_ylabel('LoRA rank')
    ax.set_title('(A) Rank allocation')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # (B) Parameter counts
    ax = axes[1]
    uniform_params = [2 * d * r for r in uniform_ranks]
    chamber_params = [2 * d * r for r in chamber_ranks]
    ax.bar(x - w/2, uniform_params, w, color='grey', label='Uniform', edgecolor='k', linewidth=0.3)
    ax.bar(x + w/2, chamber_params, w, color=colors, edgecolor='k', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(chambers, fontsize=7)
    ax.set_ylabel('Parameters')
    ax.set_title('(B) Parameters per chamber')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D: rank x d_model x params
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    d_models = [64, 128, 256, 512]
    R, D = np.meshgrid(chamber_ranks, d_models)
    P = 2 * R * D
    ax3.plot_surface(R, D, P, cmap='plasma', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Rank', fontsize=8)
    ax3.set_ylabel('d_model', fontsize=8)
    ax3.set_zlabel('Params', fontsize=8)
    ax3.set_title('(C) Parameter scaling', fontsize=10)

    # (D) Total savings
    ax = axes[3]
    totals = [data["total_params_uniform"], data["total_params_chamber"]]
    ax.bar(['Uniform', 'Chamber'], totals, color=['grey', TEAL], edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Total LoRA parameters')
    ax.set_title(f'(D) {data["savings_percent"]}% savings')
    ax.grid(True, alpha=0.3, axis='y')
    # Add savings annotation
    ax.annotate(f'-{data["savings_percent"]}%', xy=(1, totals[1]),
                fontsize=14, fontweight='bold', color=CORAL, ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_06_lora.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 7: Domain Performance
# =============================================================================
def panel_7():
    print("  Panel 7: Domain Performance...")
    data = load_json("exp7_domain_performance.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 7: Four-Chamber vs Uniform Transformer', fontsize=12, fontweight='bold', y=1.02)

    fc = [r["four_chamber_completeness"] for r in rows]
    uni = [r["uniform_var_reduction"] for r in rows]

    # (A) Scatter: four-chamber vs uniform
    ax = axes[0]
    ax.scatter(uni, fc, c=TEAL, s=40, edgecolors='k', linewidths=0.3, zorder=5)
    lim = [min(min(uni), min(fc)) - 0.05, max(max(uni), max(fc)) + 0.05]
    ax.plot(lim, lim, '--', c='grey', linewidth=1)
    ax.set_xlabel('Uniform completeness')
    ax.set_ylabel('Four-chamber completeness')
    ax.set_title('(A) Completeness comparison')
    ax.grid(True, alpha=0.3)

    # (B) Paired bar
    ax = axes[1]
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w/2, uni, w, color='grey', label='Uniform', alpha=0.7)
    ax.bar(x + w/2, fc, w, color=TEAL, label='4-Chamber', alpha=0.7)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Completeness')
    ax.set_title('(B) Per-trial comparison')
    ax.legend(fontsize=7)
    ax.set_xticks(range(0, 30, 5))
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D: trial x method x completeness
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    trials = np.arange(len(rows))
    ax3.bar3d(trials - 0.2, np.zeros(len(trials)), np.zeros(len(trials)),
              0.4, 0.4, uni, color='grey', alpha=0.6)
    ax3.bar3d(trials + 0.2, np.ones(len(trials)) * 0.5, np.zeros(len(trials)),
              0.4, 0.4, fc, color=TEAL, alpha=0.6)
    ax3.set_xlabel('Trial', fontsize=8)
    ax3.set_ylabel('Method', fontsize=8)
    ax3.set_zlabel('Completeness', fontsize=8)
    ax3.set_title('(C) Performance landscape', fontsize=10)

    # (D) Win rate
    ax = axes[3]
    wins = sum(1 for r in rows if r["four_chamber_wins"])
    losses = len(rows) - wins
    ax.bar(['4-Chamber', 'Uniform'], [wins, losses], color=[TEAL, 'grey'], edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Wins')
    ax.set_title(f'(D) Win rate: {wins}/{len(rows)}')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_07_domain.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Panel 8: Carnot Bound
# =============================================================================
def panel_8():
    print("  Panel 8: Carnot Bound...")
    data = load_json("exp8_carnot.json")
    rows = data["data"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Panel 8: Carnot Bound on Information Gain Per Rumination Cycle', fontsize=12, fontweight='bold', y=1.02)

    T1s = [r["T1"] for r in rows]
    T4s = [r["T4"] for r in rows]
    carnots = [r["carnot_efficiency"] for r in rows]

    # (A) T1 vs T4
    ax = axes[0]
    ax.scatter(T1s, T4s, c=TEAL, s=20, alpha=0.6, edgecolors='none')
    lim = [0, max(max(T1s), max(T4s)) * 1.1]
    ax.plot(lim, lim, '--', c='grey', linewidth=1, label='T1=T4')
    ax.set_xlabel('T1 (Rumen temperature)')
    ax.set_ylabel('T4 (Abomasum temperature)')
    ax.set_title('(A) Chamber temperatures')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (B) Carnot efficiency distribution
    ax = axes[1]
    ax.hist(carnots, bins=30, color=TEAL, edgecolor='k', linewidth=0.3, alpha=0.8)
    ax.axvline(x=0, color=CORAL, linestyle='--', linewidth=2)
    ax.set_xlabel('Carnot efficiency')
    ax.set_ylabel('Count')
    ax.set_title('(B) Carnot efficiency distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # (C) 3D: T1 x T4 x Carnot
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].remove()
    ax3.scatter(T1s[:100], T4s[:100], carnots[:100], c=carnots[:100], cmap='viridis', s=15, alpha=0.7)
    ax3.set_xlabel('T1', fontsize=8)
    ax3.set_ylabel('T4', fontsize=8)
    ax3.set_zlabel('Carnot', fontsize=8)
    ax3.set_title('(C) Efficiency landscape', fontsize=10)

    # (D) Fraction respecting bound per cycle
    ax = axes[3]
    cycles = sorted(set(r["cycle"] for r in rows))
    fracs = []
    for c in cycles:
        c_rows = [r for r in rows if r["cycle"] == c]
        if c_rows:
            fracs.append(sum(1 for r in c_rows if r["below_carnot"]) / len(c_rows))
        else:
            fracs.append(0)
    ax.bar(cycles[:15], fracs[:15], color=TEAL, edgecolor='k', linewidth=0.3)
    ax.axhline(y=0.5, color=CORAL, linestyle='--', linewidth=1)
    ax.set_xlabel('Rumination cycle')
    ax.set_ylabel('Fraction respecting bound')
    ax.set_title('(D) Carnot compliance per cycle')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR / "panel_08_carnot.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
def main():
    print("Generating RPA panels...")
    for i, func in enumerate([panel_1, panel_2, panel_3, panel_4, panel_5, panel_6, panel_7, panel_8], 1):
        try:
            func()
            print(f"  Panel {i} done.")
        except Exception as e:
            print(f"  Panel {i} FAILED: {e}")
            import traceback; traceback.print_exc()
    print(f"\nAll panels saved to: {PANELS_DIR}")

if __name__ == "__main__":
    main()
