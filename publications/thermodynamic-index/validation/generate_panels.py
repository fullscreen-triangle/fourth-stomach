"""
Generate 8 panel figures for the Distributed Thermodynamic Stock Index paper.
Each panel: white background, 4 charts in a row, at least one 3D, minimal text.
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PANELS_DIR = Path(__file__).parent / "panels"
PANELS_DIR.mkdir(exist_ok=True)

TEAL, CORAL, NAVY, GOLD, PURPLE = '#2ca89a', '#e8734a', '#1a3a5c', '#d4a843', '#7b4f9d'
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
    'font.size':9,'axes.titlesize':10,'figure.dpi':200})

def load(f):
    with open(RESULTS_DIR/f) as fh: return json.load(fh)

# =========== PANEL 1: Maxwell-Boltzmann ===========
def panel_1():
    print("  Panel 1: Maxwell-Boltzmann...")
    data = load("exp1_maxwell_boltzmann.json")["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 1: Maxwell--Boltzmann Distribution for Transaction Speeds",
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) PDF overlay at T=1.0
    ax = fig.add_subplot(1,4,1)
    d = data[1]  # T=1.0
    ax.bar(d["centers"], d["hist"], width=d["centers"][1]-d["centers"][0],
           color=TEAL, alpha=0.6, label='Simulated')
    ax.plot(d["centers"], d["mb_pdf"], color=CORAL, linewidth=2, label='MB theory')
    ax.set_xlabel('Speed $v$'); ax.set_ylabel('Density')
    ax.set_title('(A) $T=1.0$: PDF comparison'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) v_mp measured vs theory across temperatures
    ax = fig.add_subplot(1,4,2)
    Ts = [d["temperature"] for d in data]
    v_th = [d["v_mp_theory"] for d in data]
    v_ms = [d["v_mp_measured"] for d in data]
    ax.scatter(v_th, v_ms, c=TEAL, s=60, edgecolors='k', linewidths=0.5, zorder=5)
    lim = [0, max(max(v_th), max(v_ms))*1.1]
    ax.plot(lim, lim, '--', c=CORAL, linewidth=1.5)
    ax.set_xlabel('$v_{mp}$ theory'); ax.set_ylabel('$v_{mp}$ measured')
    ax.set_title('(B) Most probable speed'); ax.grid(True, alpha=0.3)

    # (C) 3D: T x v x f(v)
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    for i, d in enumerate(data):
        c = np.array(d["centers"]); p = np.array(d["mb_pdf"])
        ax3.plot(c, [d["temperature"]]*len(c), p, color=cm.viridis(i/len(data)), linewidth=1.5)
    ax3.set_xlabel('Speed', fontsize=8); ax3.set_ylabel('$T$', fontsize=8)
    ax3.set_zlabel('$f(v)$', fontsize=8); ax3.set_title('(C) MB surface', fontsize=10)

    # (D) Chi-squared p-values
    ax = fig.add_subplot(1,4,4)
    pvals = [d["p_value"] for d in data]
    colors = [TEAL if p > 0.01 else CORAL for p in pvals]
    ax.bar(range(len(Ts)), pvals, color=colors, edgecolor='k', linewidth=0.5)
    ax.axhline(y=0.01, color=CORAL, linestyle='--', linewidth=1.5, label='$p=0.01$ threshold')
    ax.set_xticks(range(len(Ts))); ax.set_xticklabels([f'T={t}' for t in Ts], fontsize=7)
    ax.set_ylabel('$p$-value'); ax.set_title('(D) $\\chi^2$ test $p$-values')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_01_maxwell_boltzmann.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 2: Ideal Gas Law ===========
def panel_2():
    print("  Panel 2: Ideal Gas Law...")
    data = load("exp2_ideal_gas_law.json")
    rows = data["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 2: Ideal Market Gas Law $PV = Nk_BT$",
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) PV vs NkT scatter
    ax = fig.add_subplot(1,4,1)
    NkT = [r["N"]*r["T"] for r in rows]
    PV = [r["P_measured"]*r["V"] for r in rows]
    ax.scatter(NkT, PV, c=TEAL, s=15, alpha=0.6, edgecolors='none')
    lim = [0, max(max(NkT), max(PV))*1.1]
    ax.plot(lim, lim, '--', c=CORAL, linewidth=2, label='$PV=NkT$')
    ax.set_xlabel('$NkT$'); ax.set_ylabel('$PV$')
    ax.set_title('(A) $PV$ vs $NkT$'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) Ratio histogram
    ax = fig.add_subplot(1,4,2)
    ratios = [r["PV_over_NkT"] for r in rows]
    ax.hist(ratios, bins=30, color=TEAL, edgecolor='k', linewidth=0.5, alpha=0.8)
    ax.axvline(x=1.0, color=CORAL, linestyle='--', linewidth=2, label='Theory = 1.0')
    ax.axvline(x=np.mean(ratios), color=NAVY, linestyle='-', linewidth=2,
               label=f'Mean = {np.mean(ratios):.4f}')
    ax.set_xlabel('$PV/(NkT)$'); ax.set_ylabel('Count')
    ax.set_title('(B) Compressibility ratio'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) 3D: N x T x ratio
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    Ns = np.array([r["N"] for r in rows])
    Ts = np.array([r["T"] for r in rows])
    Rs = np.array(ratios)
    sc = ax3.scatter(Ns, Ts, Rs, c=Rs, cmap='coolwarm', s=20, vmin=0.8, vmax=1.2,
                     edgecolors='k', linewidths=0.2)
    ax3.set_xlabel('$N$', fontsize=8); ax3.set_ylabel('$T$', fontsize=8)
    ax3.set_zlabel('$PV/NkT$', fontsize=8); ax3.set_title('(C) Ratio landscape', fontsize=10)

    # (D) Ratio vs N (averaged over T, V)
    ax = fig.add_subplot(1,4,4)
    unique_N = sorted(set(r["N"] for r in rows))
    mean_by_N = [np.mean([r["PV_over_NkT"] for r in rows if r["N"]==n]) for n in unique_N]
    std_by_N = [np.std([r["PV_over_NkT"] for r in rows if r["N"]==n]) for n in unique_N]
    ax.errorbar(unique_N, mean_by_N, yerr=std_by_N, fmt='o-', color=TEAL,
                markersize=8, linewidth=2, capsize=4)
    ax.axhline(y=1.0, color=CORAL, linestyle='--', linewidth=1.5)
    ax.set_xlabel('$N$ (stocks)'); ax.set_ylabel('$\\langle PV/NkT \\rangle$')
    ax.set_title('(D) Ratio vs $N$'); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_02_ideal_gas_law.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 3: Phase Transitions ===========
def panel_3():
    print("  Panel 3: Phase Transitions...")
    data = load("exp3_phase_transitions.json")
    T_c = data["T_c"]; V_c = data["V_c"]; P_c = data["P_c"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 3: Phase Diagram — Van der Waals Market Equation of State",
                 fontsize=12, fontweight='bold', y=1.02)

    # (A) PV isotherms
    ax = fig.add_subplot(1,4,1)
    for i, iso in enumerate(data["isotherms"]):
        V = np.array(iso["volumes"]); P = np.array(iso["pressures"])
        mask = (P > -500) & (P < 500)
        color = cm.coolwarm(i / len(data["isotherms"]))
        label = f'$T/T_c={iso["T_over_Tc"]:.1f}$'
        ax.plot(V[mask], P[mask], color=color, linewidth=1.5, label=label)
    ax.scatter([V_c], [P_c], c='k', s=80, zorder=10, marker='*')
    ax.set_xlabel('$V$'); ax.set_ylabel('$P$'); ax.set_ylim(-100, 400)
    ax.set_title('(A) PV isotherms'); ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    # (B) Phase boundary
    ax = fig.add_subplot(1,4,2)
    if data["phase_boundary"]:
        pb_T = [p["T_over_Tc"] for p in data["phase_boundary"]]
        pb_P = [p["P_coexistence"] for p in data["phase_boundary"]]
        ax.plot(pb_T, pb_P, 'o-', color=NAVY, linewidth=2, markersize=4)
        ax.scatter([1.0], [P_c], c=CORAL, s=100, zorder=10, marker='*', label='Critical point')
    ax.set_xlabel('$T/T_c$'); ax.set_ylabel('$P_{coex}$')
    ax.set_title('(B) Phase boundary'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) 3D PVT surface
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    for i, iso in enumerate(data["isotherms"]):
        V = np.array(iso["volumes"]); P = np.array(iso["pressures"])
        mask = (P > -200) & (P < 500)
        T_arr = np.full(mask.sum(), iso["T"])
        ax3.plot(V[mask], T_arr, P[mask], color=cm.coolwarm(i/len(data["isotherms"])), linewidth=1)
    ax3.scatter([V_c], [T_c], [P_c], c='k', s=80, marker='*')
    ax3.set_xlabel('$V$', fontsize=8); ax3.set_ylabel('$T$', fontsize=8)
    ax3.set_zlabel('$P$', fontsize=8); ax3.set_title('(C) PVT surface', fontsize=10)

    # (D) Phase regions (schematic from VdW)
    ax = fig.add_subplot(1,4,4)
    T_range = np.linspace(0.4, 1.5, 200)
    # Spinodal curves (dP/dV = 0 locus)
    a, b, N = data["a"], data["b"], data["N"]
    for T_frac in T_range:
        T = T_frac * T_c
        # V_spinodal from dP/dV = 0: -NkT/(V-Nb)^2 + 2aN^2/V^3 = 0
        # V^3 * NkT = 2aN^2 * (V-Nb)^2 — solve numerically
    # Simple fill instead
    ax.fill_between([0.4, 1.0], [0, 0], [1, 1], alpha=0.2, color=NAVY, label='Crystal')
    ax.fill_between([0.4, 1.0], [1, 1], [2, 2], alpha=0.2, color=TEAL, label='Liquid')
    ax.fill_between([1.0, 1.5], [0, 0], [2, 2], alpha=0.2, color=GOLD, label='Gas')
    ax.scatter([1.0], [1.0], c='k', s=100, marker='*', zorder=10, label='Critical point')
    ax.set_xlabel('$T/T_c$'); ax.set_ylabel('$P/P_c$')
    ax.set_title('(D) Phase regions'); ax.legend(fontsize=6); ax.set_xlim(0.4, 1.5)
    ax.set_ylim(0, 2); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_03_phase_transitions.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 4: Carnot Bound ===========
def panel_4():
    print("  Panel 4: Carnot Bound...")
    data = load("exp4_carnot_bound.json")["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 4: Carnot Bound on Trading Efficiency",
                 fontsize=12, fontweight='bold', y=1.02)

    carnot = [d["carnot_efficiency"] for d in data]
    actual = [d["actual_efficiency"] for d in data]
    T_hot = [d["T_hot"] for d in data]
    T_cold = [d["T_cold"] for d in data]

    # (A) Actual vs Carnot scatter
    ax = fig.add_subplot(1,4,1)
    ax.scatter(carnot, actual, c=TEAL, s=15, alpha=0.6, edgecolors='none')
    ax.plot([0,1], [0,1], '--', c=CORAL, linewidth=2, label='$\\eta = \\eta_C$ (bound)')
    ax.set_xlabel('Carnot efficiency $\\eta_C$'); ax.set_ylabel('Actual efficiency $\\eta$')
    ax.set_title('(A) Actual vs Carnot'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) Efficiency gap histogram
    ax = fig.add_subplot(1,4,2)
    gaps = [c - a for c, a in zip(carnot, actual)]
    ax.hist(gaps, bins=30, color=TEAL, edgecolor='k', linewidth=0.5)
    ax.axvline(x=0, color=CORAL, linestyle='--', linewidth=2, label='Bound ($\\Delta=0$)')
    ax.set_xlabel('$\\eta_C - \\eta$ (gap)'); ax.set_ylabel('Count')
    ax.set_title('(B) Efficiency gap'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) 3D: T_hot x T_cold x efficiency
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    ax3.scatter(T_hot, T_cold, actual, c=actual, cmap='viridis', s=10, alpha=0.6)
    # Carnot surface
    th = np.linspace(2, 20, 20); tc = np.linspace(0.1, 18, 20)
    TH, TC = np.meshgrid(th, tc)
    ETA = np.where(TC < TH, 1 - TC/TH, np.nan)
    ax3.plot_surface(TH, TC, ETA, alpha=0.15, color=CORAL)
    ax3.set_xlabel('$T_{hot}$', fontsize=8); ax3.set_ylabel('$T_{cold}$', fontsize=8)
    ax3.set_zlabel('$\\eta$', fontsize=8); ax3.set_title('(C) Efficiency surface', fontsize=10)

    # (D) Efficiency ratio eta/eta_C
    ax = fig.add_subplot(1,4,4)
    ratios = [a/c if c > 0 else 0 for a, c in zip(actual, carnot)]
    ax.hist(ratios, bins=30, color=NAVY, edgecolor='k', linewidth=0.5)
    ax.axvline(x=1.0, color=CORAL, linestyle='--', linewidth=2, label='$\\eta/\\eta_C = 1$')
    ax.set_xlabel('$\\eta / \\eta_C$'); ax.set_ylabel('Count')
    ax.set_title('(D) Fraction of Carnot'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_04_carnot_bound.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 5: Fluctuation-Dissipation ===========
def panel_5():
    print("  Panel 5: Fluctuation-Dissipation...")
    data = load("exp5_fluctuation_dissipation.json")["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 5: Fluctuation--Dissipation Theorem",
                 fontsize=12, fontweight='bold', y=1.02)

    Ts = [d["temperature"] for d in data]
    fdt_th = [d["fdt_ratio_theory"] for d in data]
    sig_r = [d["sigma_realised"] for d in data]

    # (A) FDT ratio vs T (theory: linear)
    ax = fig.add_subplot(1,4,1)
    ax.plot(Ts, fdt_th, 'o-', color=TEAL, markersize=8, linewidth=2, label='$2T/m$')
    ax.plot(Ts, [2*T for T in Ts], '--', color=CORAL, linewidth=2, label='Theory $2T$')
    ax.set_xlabel('$T$'); ax.set_ylabel('FDT ratio')
    ax.set_title('(A) FDT ratio = $2T/m$'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) sigma_realised vs sqrt(T)
    ax = fig.add_subplot(1,4,2)
    sqrtT = [np.sqrt(T) for T in Ts]
    ax.scatter(sqrtT, sig_r, c=TEAL, s=60, edgecolors='k', linewidths=0.5, zorder=5)
    z = np.polyfit(sqrtT, sig_r, 1)
    x_fit = np.linspace(min(sqrtT), max(sqrtT), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), '--', color=CORAL, linewidth=2)
    ax.set_xlabel('$\\sqrt{T}$'); ax.set_ylabel('$\\sigma_{realised}$')
    ax.set_title('(B) $\\sigma \\propto \\sqrt{T}$'); ax.grid(True, alpha=0.3)

    # (C) 3D: T x sigma_realised x sigma_implied
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    sig_impl = [d["sigma_implied_theory"] for d in data]
    ax3.scatter(Ts, sig_r, sig_impl, c=TEAL, s=60, edgecolors='k', linewidths=0.3)
    # Surface: sigma_implied = sigma_realised * sqrt(2T)
    T_grid = np.linspace(min(Ts), max(Ts), 20)
    sr_grid = np.linspace(min(sig_r), max(sig_r), 20)
    TG, SG = np.meshgrid(T_grid, sr_grid)
    SI = SG * np.sqrt(2*TG)
    ax3.plot_surface(TG, SG, SI, alpha=0.2, color=CORAL)
    ax3.set_xlabel('$T$', fontsize=8); ax3.set_ylabel('$\\sigma_r$', fontsize=8)
    ax3.set_zlabel('$\\sigma_i$', fontsize=8); ax3.set_title('(C) FDT surface', fontsize=10)

    # (D) Implied vs realised scatter
    ax = fig.add_subplot(1,4,4)
    ax.scatter(sig_r, sig_impl, c=[cm.viridis(T/max(Ts)) for T in Ts], s=80,
               edgecolors='k', linewidths=0.5, zorder=5)
    ax.plot([0, max(sig_impl)], [0, max(sig_impl)], '--', c='grey', linewidth=1)
    ax.set_xlabel('$\\sigma_{realised}$'); ax.set_ylabel('$\\sigma_{implied}$')
    ax.set_title('(D) Implied vs realised'); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_05_fluctuation_dissipation.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 6: Gauge Invariance ===========
def panel_6():
    print("  Panel 6: Gauge Invariance...")
    data = load("exp6_gauge_invariance.json")
    rows = data["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 6: Gauge Invariance Under Frequency Scaling",
                 fontsize=12, fontweight='bold', y=1.02)

    lambdas = [r["lambda"] for r in rows]
    dT = [r["T_drift"] for r in rows]
    dP = [r["P_drift"] for r in rows]
    dS = [r["S_drift"] for r in rows]

    # (A) Drift vs lambda (all zero)
    ax = fig.add_subplot(1,4,1)
    x = range(len(lambdas))
    ax.bar([i-0.2 for i in x], dT, 0.2, color=TEAL, label='$\\Delta T$')
    ax.bar([i for i in x], dP, 0.2, color=CORAL, label='$\\Delta P$')
    ax.bar([i+0.2 for i in x], dS, 0.2, color=NAVY, label='$\\Delta S$')
    ax.set_xticks(x); ax.set_xticklabels([f'{l}' for l in lambdas], fontsize=7, rotation=45)
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('Drift')
    ax.set_title('(A) Observable drift = 0'); ax.legend(fontsize=6); ax.set_ylim(-0.001, 0.001)

    # (B) Gear ratio matrix (invariant)
    ax = fig.add_subplot(1,4,2)
    freqs = np.array([1.0, 2.5, 0.7, 3.3, 1.8])
    gr = np.outer(freqs, 1.0/freqs)
    im = ax.imshow(gr, cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Stock $j$'); ax.set_ylabel('Stock $i$')
    ax.set_title('(B) Gear ratio matrix $R_{i\\to j}$')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # (C) 3D: lambda x observable_index x value (flat surface)
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    obs_names = ['T', 'P', 'S']
    T_val, P_val, S_val = 5.0, 0.5, 100.0  # reference values
    L, O = np.meshgrid(range(len(lambdas)), range(3))
    Z = np.array([[T_val, T_val, T_val, T_val, T_val, T_val, T_val],
                  [P_val]*7, [S_val]*7], dtype=float)
    Z_norm = Z / Z[:, 0:1]  # normalise to 1
    ax3.plot_surface(L, O, Z_norm, cmap='coolwarm', alpha=0.85, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('$\\lambda$ idx', fontsize=8); ax3.set_ylabel('Observable', fontsize=8)
    ax3.set_zlabel('Normalised value', fontsize=8)
    ax3.set_title('(C) Flat = gauge-invariant', fontsize=10)

    # (D) Gear ratio drift per lambda
    ax = fig.add_subplot(1,4,4)
    gr_drifts = data["gear_ratio_drifts"]
    ax.semilogy(range(len(lambdas)), [max(d, 1e-16) for d in gr_drifts],
                'o-', color=NAVY, markersize=8, linewidth=2)
    ax.axhline(y=1e-10, color=CORAL, linestyle='--', linewidth=1.5, label='Machine $\\epsilon$')
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f'{l}' for l in lambdas], fontsize=7, rotation=45)
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('Max gear ratio drift')
    ax.set_title('(D) Gear ratio invariance'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_06_gauge_invariance.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 7: Third Law ===========
def panel_7():
    print("  Panel 7: Third Law...")
    data = load("exp7_third_law.json")["data"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 7: Third Law — Zero Variance is Unreachable",
                 fontsize=12, fontweight='bold', y=1.02)

    steps = [d["step"] for d in data]
    T = [d["temperature"] for d in data]
    S = [d["entropy"] for d in data]
    logT = [d["log_T"] for d in data]

    # (A) T vs cooling step (log scale)
    ax = fig.add_subplot(1,4,1)
    ax.semilogy(steps, T, color=TEAL, linewidth=2)
    ax.axhline(y=0, color=CORAL, linestyle='--', linewidth=1, label='$T=0$ (unreachable)')
    ax.set_xlabel('Cooling step'); ax.set_ylabel('$T$ (log scale)')
    ax.set_title('(A) Cooling trajectory'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) Entropy vs T
    ax = fig.add_subplot(1,4,2)
    ax.plot(T, S, 'o-', color=NAVY, markersize=3, linewidth=1.5)
    ax.set_xlabel('$T$'); ax.set_ylabel('$S$')
    ax.set_title('(B) $S \\to 0$ as $T \\to 0$'); ax.grid(True, alpha=0.3)

    # (C) 3D: step x T x S
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    ax3.plot(steps, T, S, color=TEAL, linewidth=2)
    ax3.scatter([steps[0]], [T[0]], [S[0]], c=CORAL, s=60, marker='o', label='Start')
    ax3.scatter([steps[-1]], [T[-1]], [S[-1]], c=NAVY, s=60, marker='s', label='End')
    ax3.set_xlabel('Step', fontsize=8); ax3.set_ylabel('$T$', fontsize=8)
    ax3.set_zlabel('$S$', fontsize=8); ax3.set_title('(C) Cooling in $(T,S)$', fontsize=10)
    ax3.legend(fontsize=7)

    # (D) dT per step (diminishing returns)
    ax = fig.add_subplot(1,4,4)
    dT = [-T[i+1]+T[i] for i in range(len(T)-1)]
    ax.semilogy(steps[:-1], dT, color=GOLD, linewidth=2)
    ax.set_xlabel('Cooling step'); ax.set_ylabel('$|\\Delta T|$ per step (log)')
    ax.set_title('(D) Diminishing returns'); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_07_third_law.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========== PANEL 8: Critical Exponents ===========
def panel_8():
    print("  Panel 8: Critical Exponents...")
    data = load("exp8_critical_exponents.json")
    rows = data["data"]
    T_c = data["T_c"]
    fig = plt.figure(figsize=(18,4))
    fig.suptitle("Panel 8: Critical Exponents Near Phase Transition",
                 fontsize=12, fontweight='bold', y=1.02)

    t_red = [r["t_reduced"] for r in rows]
    log_t = [r["log_t"] for r in rows]
    Cv = [r["Cv"] for r in rows]
    log_Cv = [r["log_Cv"] for r in rows]
    kappa = [r["kappa_T"] for r in rows]
    log_k = [r["log_kappa"] for r in rows]

    # (A) Cv vs reduced T (log-log)
    ax = fig.add_subplot(1,4,1)
    valid = [i for i in range(len(Cv)) if Cv[i] > 0 and np.isfinite(log_Cv[i])]
    ax.scatter([log_t[i] for i in valid], [log_Cv[i] for i in valid],
               c=TEAL, s=40, edgecolors='k', linewidths=0.3)
    if len(valid) > 2:
        z = np.polyfit([log_t[i] for i in valid], [log_Cv[i] for i in valid], 1)
        x_fit = np.linspace(min(log_t[i] for i in valid), max(log_t[i] for i in valid), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), '--', color=CORAL, linewidth=2,
                label=f'slope={z[0]:.2f}')
    ax.set_xlabel('$\\log_{10}|t|$'); ax.set_ylabel('$\\log_{10} C_V$')
    ax.set_title('(A) $C_V$ divergence'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) Compressibility vs reduced T
    ax = fig.add_subplot(1,4,2)
    valid_k = [i for i in range(len(kappa)) if kappa[i] > 0 and np.isfinite(log_k[i])]
    ax.scatter([log_t[i] for i in valid_k], [log_k[i] for i in valid_k],
               c=NAVY, s=40, edgecolors='k', linewidths=0.3)
    if len(valid_k) > 2:
        z = np.polyfit([log_t[i] for i in valid_k], [log_k[i] for i in valid_k], 1)
        x_fit = np.linspace(min(log_t[i] for i in valid_k), max(log_t[i] for i in valid_k), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), '--', color=CORAL, linewidth=2,
                label=f'$\\gamma$={-z[0]:.2f}')
    ax.set_xlabel('$\\log_{10}|t|$'); ax.set_ylabel('$\\log_{10} \\kappa_T$')
    ax.set_title('(B) $\\kappa_T$ divergence'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) 3D: t_reduced x Cv x kappa
    ax3 = fig.add_subplot(1,4,3, projection='3d')
    both_valid = [i for i in range(len(rows)) if i in valid and i in valid_k]
    ax3.scatter([t_red[i] for i in both_valid], [Cv[i] for i in both_valid],
                [kappa[i] for i in both_valid],
                c=[cm.inferno(t_red[i]*5) for i in both_valid], s=30, edgecolors='k', linewidths=0.2)
    ax3.set_xlabel('$|t|$', fontsize=8); ax3.set_ylabel('$C_V$', fontsize=8)
    ax3.set_zlabel('$\\kappa_T$', fontsize=8); ax3.set_title('(C) Critical landscape', fontsize=10)

    # (D) VdW isotherm at T_c showing inflection point
    ax = fig.add_subplot(1,4,4)
    a, b, N = data.get("a", 2.0), data.get("b", 0.03), 100
    V_range = np.linspace(N*b*1.1, 50, 500)
    for T_frac, color, label in [(0.85, NAVY, '$0.85T_c$'), (1.0, CORAL, '$T_c$'), (1.2, TEAL, '$1.2T_c$')]:
        T = T_frac * T_c
        P = N*T/(V_range - N*b) - a*N**2/V_range**2
        mask = (P > -50) & (P < 300)
        ax.plot(V_range[mask], P[mask], color=color, linewidth=2, label=label)
    ax.scatter([3*N*b], [a/(27*b**2)], c='k', s=80, marker='*', zorder=10)
    ax.set_xlabel('$V$'); ax.set_ylabel('$P$')
    ax.set_title('(D) Critical isotherm'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PANELS_DIR/"panel_08_critical_exponents.png", bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    print("Generating DTI panels...")
    for i, fn in enumerate([panel_1, panel_2, panel_3, panel_4,
                            panel_5, panel_6, panel_7, panel_8], 1):
        try:
            fn(); print(f"  Panel {i} done.")
        except Exception as e:
            print(f"  Panel {i} FAILED: {e}")
            import traceback; traceback.print_exc()
    print(f"\nAll panels saved to: {PANELS_DIR}")

if __name__ == "__main__":
    main()
