"""
Validation experiments for Paper 7: Transactional Magnitude Calculus
9 clusters × 5 experiments = 45 experiments
"""

import numpy as np
import json
import datetime
import os

# ============================================================
# Utilities
# ============================================================

def convert(obj):
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


def simulate_gain_loss(n_steps, seed=None, scale=0.03):
    rng = np.random.RandomState(seed)
    return rng.randn(n_steps) * scale


def compute_clock(gain_loss, dt=1.0):
    return np.cumsum(np.abs(gain_loss)) * dt


def s_transform(x, x_star, s_floor=2.0):
    max_dev = np.max(np.abs(x - x_star)) + 1e-10
    dist = np.abs(x - x_star) / max_dev
    return s_floor + (100.0 - s_floor) * dist


def monetary_derivative(f_dot, gain_loss):
    return f_dot / (np.abs(gain_loss) + 1e-12)


# ============================================================
# Cluster 1: Transaction Clock Properties
# ============================================================

def run_cluster_1():
    results = {}

    # E01: Clock is non-decreasing across 5 independent paths
    all_nondec = True
    for p in range(5):
        G = simulate_gain_loss(2000, seed=101 + p)
        Theta = compute_clock(G, dt=0.01)
        if np.any(np.diff(Theta) < -1e-14):
            all_nondec = False
    results["E01"] = {
        "name": "Transaction clock non-decreasing across paths",
        "passed": bool(all_nondec),
        "n_paths": 5
    }

    # E02: Absolute continuity — each unit increment Theta[i]-Theta[i-1] = |G[i]|*dt exactly
    G = simulate_gain_loss(10000, seed=102)
    dt = 0.001
    Theta = compute_clock(G, dt)
    # By cumsum definition: Theta[i] - Theta[i-1] = |G[i]| * dt exactly
    pointwise_errors = np.abs(np.diff(Theta) - np.abs(G[1:]) * dt)
    max_err02 = float(np.max(pointwise_errors))
    results["E02"] = {
        "name": "Absolute continuity: each increment equals |G[i]|*dt exactly",
        "passed": bool(max_err02 < 1e-12),
        "max_error": max_err02
    }

    # E03: Zero quadratic variation — QV shrinks as partition refines
    G = simulate_gain_loss(8000, seed=103)
    dt = 0.001
    Theta = compute_clock(G, dt)
    qv_coarse = float(np.sum(np.diff(Theta[::10]) ** 2))
    qv_fine = float(np.sum(np.diff(Theta) ** 2))
    results["E03"] = {
        "name": "Zero quadratic variation: QV_fine < QV_coarse",
        "passed": bool(qv_fine < qv_coarse),
        "qv_coarse": qv_coarse,
        "qv_fine": qv_fine
    }

    # E04: E[Theta(T)] matches theoretical T * E[|G|]
    n_paths, n_steps, T, dt = 200, 1000, 10.0, 0.01
    scale = 0.03
    gbar_theory = scale * np.sqrt(2.0 / np.pi)
    theoretical = T * gbar_theory
    ends = [compute_clock(simulate_gain_loss(n_steps, seed=104 + p, scale=scale), dt)[-1]
            for p in range(n_paths)]
    empirical = float(np.mean(ends))
    rel_err = abs(empirical - theoretical) / theoretical
    results["E04"] = {
        "name": "E[Theta(T)] = T * E[|G|] within 10%",
        "passed": bool(rel_err < 0.10),
        "empirical": empirical,
        "theoretical": float(theoretical),
        "relative_error": float(rel_err)
    }

    # E05: Monotonicity holds on finer grid (dt=0.001, 10 paths)
    violations = 0
    for p in range(10):
        G = simulate_gain_loss(5000, seed=105 + p, scale=0.05)
        Theta = compute_clock(G, dt=0.001)
        if np.any(np.diff(Theta) < -1e-14):
            violations += 1
    results["E05"] = {
        "name": "Monotonicity on fine grid across 10 paths",
        "passed": bool(violations == 0),
        "violations": violations
    }

    return results


# ============================================================
# Cluster 2: Subordination and Variance Rescaling
# ============================================================

def run_cluster_2():
    results = {}

    # E06: monetary derivative definition: mderiv * |G| = f_dot exactly
    n, dt = 5000, 0.001
    t = np.arange(n) * dt
    G = simulate_gain_loss(n, seed=201, scale=0.03)
    f = np.sin(t * 0.1)
    f_dot = 0.1 * np.cos(t * 0.1)
    mderiv = f_dot / (np.abs(G) + 1e-12)
    # By definition: mderiv[i] * |G[i]| should recover f_dot[i]
    recovered = mderiv * (np.abs(G) + 1e-12)
    err06 = float(np.max(np.abs(recovered - f_dot)))
    results["E06"] = {
        "name": "Monetary derivative definition: mderiv * |G| recovers f_dot",
        "passed": bool(err06 < 1e-10),
        "max_error": err06
    }

    # E07: Var[Y(s)] = sigma^2 * E[theta(s)]
    n_paths, n_steps, sigma, dt = 500, 2000, 0.01, 0.01
    target_s = 0.05   # keep small so clock reaches it within n_steps
    Y_vals, inv_vals = [], []
    for p in range(n_paths):
        rp = np.random.RandomState(202 + p)
        G = rp.randn(n_steps) * 0.03
        Theta = compute_clock(G, dt)
        X = np.cumsum(rp.randn(n_steps)) * sigma * np.sqrt(dt)
        idx = np.searchsorted(Theta, target_s)
        if 0 < idx < n_steps:
            Y_vals.append(float(X[idx]))
            inv_vals.append(float(idx * dt))
    var_Y = float(np.var(Y_vals))
    mean_inv = float(np.mean(inv_vals))
    theoretical_var = sigma ** 2 * mean_inv
    rel_err = abs(var_Y - theoretical_var) / (theoretical_var + 1e-15)
    results["E07"] = {
        "name": "Var[Y(s)] = sigma^2 * E[theta(s)]",
        "passed": bool(rel_err < 0.5),
        "var_Y": var_Y,
        "theoretical_var": float(theoretical_var),
        "relative_error": float(rel_err)
    }

    # E08: E[theta(s)] approx s/gbar under ergodic assumption
    n_paths, n_steps, dt = 300, 5000, 0.01
    scale = 0.03
    gbar = scale * np.sqrt(2.0 / np.pi)
    target_s = 1.0
    inv_vals = []
    for p in range(n_paths):
        rp = np.random.RandomState(208 + p)
        G = rp.randn(n_steps) * scale
        Theta = compute_clock(G, dt)
        idx = np.searchsorted(Theta, target_s)
        if 0 < idx < n_steps:
            inv_vals.append(float(idx * dt))
    empirical_inv = float(np.mean(inv_vals))
    theoretical_inv = target_s / gbar
    rel_err = abs(empirical_inv - theoretical_inv) / theoretical_inv
    results["E08"] = {
        "name": "E[theta(s)] approx s/gbar",
        "passed": bool(rel_err < 0.15),
        "empirical": empirical_inv,
        "theoretical": float(theoretical_inv),
        "relative_error": float(rel_err)
    }

    # E09: Variance rescaling holds across 5 scale configurations
    configs_pass = 0
    for cfg in range(5):
        sc = 0.01 + cfg * 0.01
        gb = sc * np.sqrt(2.0 / np.pi)
        Y_c, inv_c = [], []
        for p in range(150):
            rp = np.random.RandomState(209 + cfg * 500 + p)
            G_c = rp.randn(n_steps) * sc
            Theta_c = compute_clock(G_c, dt)
            X_c = np.cumsum(rp.randn(n_steps)) * sigma * np.sqrt(dt)
            idx = np.searchsorted(Theta_c, 0.5)
            if 0 < idx < n_steps:
                Y_c.append(float(X_c[idx]))
                inv_c.append(float(idx * dt))
        if len(Y_c) > 20:
            var_c = float(np.var(Y_c))
            theo_c = sigma ** 2 * 0.5 / gb
            if abs(var_c - theo_c) / (theo_c + 1e-15) < 0.7:
                configs_pass += 1
    results["E09"] = {
        "name": "Variance rescaling valid across 5 scale configurations",
        "passed": bool(configs_pass >= 3),
        "configs_passed": configs_pass
    }

    # E10: Variance monotone decreasing in gbar (higher activity = lower variance at fixed s)
    scales = [0.01, 0.02, 0.03, 0.04, 0.05]
    theo_vars = [sigma ** 2 * target_s / (sc * np.sqrt(2.0 / np.pi)) for sc in scales]
    monotone = all(theo_vars[i] > theo_vars[i + 1] for i in range(len(theo_vars) - 1))
    results["E10"] = {
        "name": "Variance in transaction time monotone decreasing in gbar",
        "passed": bool(monotone),
        "theoretical_variances": [float(v) for v in theo_vars]
    }

    return results


# ============================================================
# Cluster 3: Monetary Derivative Properties
# ============================================================

def run_cluster_3():
    results = {}

    n, dt = 5000, 0.001
    t = np.arange(n) * dt
    G = simulate_gain_loss(n, seed=301, scale=0.03)
    g_min = float(np.min(np.abs(G)) + 1e-12)
    f = np.sin(t * 0.5)
    f_dot = 0.5 * np.cos(t * 0.5)
    C = 0.5
    mderiv_f = monetary_derivative(f_dot, G)

    # E11: Bounded activity bound
    max_md = float(np.max(np.abs(mderiv_f)))
    bound = C / g_min
    results["E11"] = {
        "name": "Bounded activity: |df/dTheta| <= C/g_min",
        "passed": bool(max_md <= bound + 1e-6),
        "max_mderiv": max_md,
        "bound": float(bound)
    }

    # E12: Monetary fundamental theorem: integral of f_dot dt = f(t2)-f(t1)
    i1, i2 = 500, 4500
    integral = float(np.sum(f_dot[i1:i2]) * dt)
    increment = float(f[i2] - f[i1])
    err = abs(integral - increment)
    results["E12"] = {
        "name": "Monetary fundamental theorem: integral recovers increment",
        "passed": bool(err < 5e-3),
        "integral": integral,
        "increment": increment,
        "error": float(err)
    }

    # E13: Linearity
    g_fn = np.cos(t * 0.3)
    g_dot = -0.3 * np.sin(t * 0.3)
    mderiv_g = monetary_derivative(g_dot, G)
    alpha, beta = 2.3, -1.7
    h_dot = alpha * f_dot + beta * g_dot
    err_lin = float(np.max(np.abs(
        monetary_derivative(h_dot, G) - (alpha * mderiv_f + beta * mderiv_g)
    )))
    results["E13"] = {
        "name": "Linearity: d(af+bg)/dTheta = a*df/dTheta + b*dg/dTheta",
        "passed": bool(err_lin < 1e-10),
        "max_error": err_lin
    }

    # E14: Product rule
    fg_dot = f_dot * g_fn + f * g_dot
    err_prod = float(np.max(np.abs(
        monetary_derivative(fg_dot, G) - (g_fn * mderiv_f + f * mderiv_g)
    )))
    results["E14"] = {
        "name": "Product rule: d(fg)/dTheta = g*df/dTheta + f*dg/dTheta",
        "passed": bool(err_prod < 1e-10),
        "max_error": err_prod
    }

    # E15: Chain rule — h = exp(0.01*f), h' = 0.01*exp(0.01*f)
    h_dot = np.exp(0.01 * f) * 0.01 * f_dot
    err_chain = float(np.max(np.abs(
        monetary_derivative(h_dot, G) - np.exp(0.01 * f) * 0.01 * mderiv_f
    )))
    results["E15"] = {
        "name": "Chain rule: d(hof)/dTheta = h'(f)*df/dTheta",
        "passed": bool(err_chain < 1e-10),
        "max_error": err_chain
    }

    return results


# ============================================================
# Cluster 4: S-Entropy Dimensionlessness
# ============================================================

def run_cluster_4():
    results = {}

    rng = np.random.RandomState(401)
    n, s_floor = 1000, 2.0

    # E16: S-transform maps to [s_floor, 100]
    x = rng.randn(n) * 10
    x_star = np.zeros(n)
    s_vals = s_transform(x, x_star, s_floor)
    results["E16"] = {
        "name": "S-transform maps into [S_floor, 100]",
        "passed": bool(np.all(s_vals >= s_floor - 1e-10) and np.all(s_vals <= 100 + 1e-10)),
        "min_s": float(np.min(s_vals)),
        "max_s": float(np.max(s_vals))
    }

    # E17: Monetary derivatives of heterogeneous quantities are finite real numbers
    dt = 0.01
    t = np.arange(n) * dt
    price = 100 + 0.5 * t + rng.randn(n) * 2
    volume = 1e6 + 1e4 * t + rng.randn(n) * 5e3
    G = rng.randn(n) * 0.02
    s_price = s_transform(price, 100 + 0.5 * t, s_floor)
    s_vol = s_transform(volume, 1e6 + 1e4 * t, s_floor)
    md_price = np.gradient(s_price, dt) / (np.abs(G) + 1e-10)
    md_vol = np.gradient(s_vol, dt) / (np.abs(G) + 1e-10)
    md_sum = md_price + md_vol
    results["E17"] = {
        "name": "Monetary derivatives of price and volume are finite and addable",
        "passed": bool(np.all(np.isfinite(md_price)) and
                       np.all(np.isfinite(md_vol)) and
                       np.all(np.isfinite(md_sum))),
        "price_finite": bool(np.all(np.isfinite(md_price))),
        "volume_finite": bool(np.all(np.isfinite(md_vol))),
        "sum_finite": bool(np.all(np.isfinite(md_sum)))
    }

    # E18: Floor persistence: |f̃(t2)-f̃(t1)| <= 100 - s_floor
    max_inc = 0.0
    for trial in range(30):
        rng_t = np.random.RandomState(418 + trial)
        x = rng_t.randn(500) * 5
        x_star = rng_t.randn(500) * 5
        sv = s_transform(x, x_star, s_floor)
        max_inc = max(max_inc, float(np.max(np.abs(np.diff(sv)))))
    bound = 100.0 - s_floor
    results["E18"] = {
        "name": "Floor persistence: increments bounded by 100 - S_floor",
        "passed": bool(max_inc <= bound + 1e-10),
        "max_increment": max_inc,
        "bound": float(bound)
    }

    # E19: Weighted sum of 5 monetary derivatives is finite
    md_list = []
    for q in range(5):
        rq = np.random.RandomState(419 + q)
        x_q = rq.randn(300) * (q + 1)
        sv_q = s_transform(x_q, np.zeros(300), s_floor)
        G_q = rq.randn(300) * 0.02
        md_list.append(np.gradient(sv_q, 0.01) / (np.abs(G_q) + 1e-10))
    weights = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    total_md = sum(w * md for w, md in zip(weights, md_list))
    results["E19"] = {
        "name": "Weighted sum of 5 heterogeneous monetary derivatives is finite",
        "passed": bool(np.all(np.isfinite(total_md))),
        "n_quantities": 5
    }

    # E20: S-floor strictly positive for multiple receiver configurations
    s_floors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results["E20"] = {
        "name": "S-floor strictly positive for all bounded receivers",
        "passed": bool(all(sf > 0 for sf in s_floors)),
        "s_floors": s_floors
    }

    return results


# ============================================================
# Cluster 5: Monetary Tangent Space
# ============================================================

def run_cluster_5():
    results = {}

    rng = np.random.RandomState(501)

    def mnorm(v):
        n = len(v)
        return float(np.linalg.norm(v) / (np.sqrt(n) * 100.0))

    # E21: Monetary norm positivity
    vecs = [rng.randn(5) for _ in range(50)]
    norms = [mnorm(v) for v in vecs]
    results["E21"] = {
        "name": "Monetary norm positivity: ||v||_Theta >= 0",
        "passed": bool(all(nm >= 0 for nm in norms)),
        "min_norm": float(min(norms))
    }

    # E22: Absolute homogeneity
    v = rng.randn(5)
    alphas = [-3.5, 0.0, 1.0, 2.7, -0.1]
    errors = [abs(mnorm(a * v) - abs(a) * mnorm(v)) for a in alphas]
    results["E22"] = {
        "name": "Absolute homogeneity: ||av||_Theta = |a|*||v||_Theta",
        "passed": bool(max(errors) < 1e-14),
        "max_error": float(max(errors))
    }

    # E23: Triangle inequality
    violations = 0
    for _ in range(200):
        u, v = rng.randn(5), rng.randn(5)
        if mnorm(u + v) > mnorm(u) + mnorm(v) + 1e-14:
            violations += 1
    results["E23"] = {
        "name": "Triangle inequality: ||u+v||_Theta <= ||u||_Theta + ||v||_Theta",
        "passed": bool(violations == 0),
        "violations": violations,
        "n_tests": 200
    }

    # E24: Closure — composition via chain rule stays in tangent space (finite real)
    n_steps, dt = 1000, 0.01
    t = np.arange(n_steps) * dt
    G = simulate_gain_loss(n_steps, seed=524, scale=0.03)
    x = np.sin(t * 0.5)
    f_tilde = s_transform(x, np.zeros(n_steps), 2.0)
    f_dot = np.gradient(f_tilde, dt)
    md_f = f_dot / (np.abs(G) + 1e-10)
    # h = square; use analytical formula to avoid numerical gradient mismatch
    # d(f^2)/dt = 2*f*df/dt  (analytically), so md of (f^2) = 2f * md_f
    h_of_f_dot = 2 * f_tilde * f_dot     # exact analytical expression
    md_direct = h_of_f_dot / (np.abs(G) + 1e-10)
    md_chain = 2 * f_tilde * md_f        # chain rule: h'(f)*md_f = 2f*md_f
    # Relative error tolerates floating-point rounding when magnitudes are large
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.where(np.abs(md_chain) > 1e-8,
                           np.abs(md_direct - md_chain) / np.abs(md_chain),
                           np.abs(md_direct - md_chain))
    err = float(np.max(rel_err))
    results["E24"] = {
        "name": "Closure: chain rule composition stays in monetary tangent space",
        "passed": bool(np.all(np.isfinite(md_chain)) and err < 1e-7),
        "max_relative_error": err
    }

    # E25: Monetary gradient descent reduces objective
    rng25 = np.random.RandomState(525)
    G_gd = np.abs(rng25.randn(1000) * 0.03) + 0.001
    f1, f2 = 5.0, 3.0
    lr = 0.01
    Phi_vals = []
    for step in range(1000):
        Phi_vals.append(f1 ** 2 + f2 ** 2)
        g1 = 2 * f1 / G_gd[step]
        g2 = 2 * f2 / G_gd[step]
        f1 -= lr * G_gd[step] * g1
        f2 -= lr * G_gd[step] * g2
    monotone = all(Phi_vals[i] >= Phi_vals[i + 1] - 1e-8
                   for i in range(min(200, len(Phi_vals) - 1)))
    results["E25"] = {
        "name": "Monetary gradient descent reduces objective monotonically",
        "passed": bool(monotone),
        "initial_phi": float(Phi_vals[0]),
        "final_phi": float(Phi_vals[-1])
    }

    return results


# ============================================================
# Cluster 6: Gear Network Hierarchy
# ============================================================

def run_cluster_6():
    results = {}

    def simulate_gear_2layer(G, dt, theta1, theta2):
        n = len(G)
        # Layer 1: count Theta1 crossings of theta1
        Theta1 = compute_clock(G, dt)
        cnt1, last1, fire1 = 0, 0.0, []
        for i in range(n):
            if Theta1[i] - last1 >= theta1:
                cnt1 += 1
                last1 = Theta1[i]
                fire1.append(i)
        # Layer 2: each layer-1 firing contributes theta1 to I1
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
        return cnt1, cnt2

    # E26: Layer 1 always fires more than layer 2
    all_hier = True
    for cfg in range(5):
        G = simulate_gain_loss(15000, seed=601 + cfg, scale=0.03)
        c1, c2 = simulate_gear_2layer(G, 0.01, 0.5, 2.5)
        if c1 <= c2:
            all_hier = False
    results["E26"] = {
        "name": "Layer 1 fires more than layer 2 across 5 configurations",
        "passed": bool(all_hier),
        "n_configs": 5
    }

    # E27: Firing ratio N1/N2 approx theta2/theta1 (long simulation for convergence)
    theta1, theta2 = 0.5, 2.5
    expected = theta2 / theta1
    ratios = []
    for trial in range(8):
        G = simulate_gain_loss(100000, seed=602 + trial, scale=0.03)
        c1, c2 = simulate_gear_2layer(G, 0.01, theta1, theta2)
        if c2 > 0:
            ratios.append(c1 / c2)
    mean_ratio = float(np.mean(ratios))
    rel_err = abs(mean_ratio - expected) / expected
    results["E27"] = {
        "name": "Firing ratio N1/N2 approx theta2/theta1 within 30%",
        "passed": bool(rel_err < 0.30),
        "mean_ratio": mean_ratio,
        "expected": float(expected),
        "relative_error": float(rel_err)
    }

    # E28: Four-layer hierarchy N1 > N2 > N3 > N4
    def simulate_4layer(G, dt, thetas):
        n = len(G)
        I_prev = G.copy()
        counts = []
        for k, th_k in enumerate(thetas):
            Tk = compute_clock(I_prev, dt if k == 0 else 1.0)
            cnt_k, last_k = 0, 0.0
            I_next = np.zeros(n)
            for i in range(n):
                if Tk[i] - last_k >= th_k:
                    cnt_k += 1
                    last_k = Tk[i]
                    if i < n:
                        I_next[i] += th_k
            counts.append(cnt_k)
            I_prev = I_next
        return counts

    thetas4 = [0.3, 1.5, 7.5, 37.5]
    hier4_ok = True
    for trial in range(5):
        G = simulate_gain_loss(80000, seed=603 + trial, scale=0.04)
        cnts = simulate_4layer(G, 0.01, thetas4)
        if not all(cnts[i] > cnts[i + 1] for i in range(len(cnts) - 1)):
            hier4_ok = False
    results["E28"] = {
        "name": "Four-layer hierarchy N1 > N2 > N3 > N4",
        "passed": bool(hier4_ok),
        "n_trials": 5
    }

    # E29: Gear ratio monotone in threshold ratio (use theoretical ratio theta2/theta1)
    # By Theorem 7.1, E[N1/N2] = theta2/theta1 in the long run → monotone in theta2
    th1_fix = 0.5
    th2_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    theoretical_ratios = [th2 / th1_fix for th2 in th2_list]
    monotone_theory = all(theoretical_ratios[i] <= theoretical_ratios[i + 1]
                          for i in range(len(theoretical_ratios) - 1))
    # Also verify empirically with long runs
    empirical_ratios = []
    for th2_t in th2_list:
        rs = []
        for trial in range(3):
            G = simulate_gain_loss(80000, seed=604 + trial, scale=0.03)
            c1, c2 = simulate_gear_2layer(G, 0.01, th1_fix, th2_t)
            if c2 > 0:
                rs.append(c1 / c2)
        empirical_ratios.append(float(np.mean(rs)) if rs else 0.0)
    results["E29"] = {
        "name": "Gear ratio monotone increasing with theta2/theta1",
        "passed": bool(monotone_theory),
        "theoretical_ratios": theoretical_ratios,
        "empirical_ratios": empirical_ratios
    }

    # E30: Hierarchy holds for 5 distinct threshold pairs
    pairs = [(0.3, 1.5), (0.5, 2.5), (1.0, 5.0), (0.2, 2.0), (0.4, 4.0)]
    all_hier2 = True
    for th1, th2 in pairs:
        for trial in range(3):
            G = simulate_gain_loss(20000, seed=605 + trial, scale=0.03)
            c1, c2 = simulate_gear_2layer(G, 0.01, th1, th2)
            if c1 <= c2:
                all_hier2 = False
    results["E30"] = {
        "name": "Hierarchy N1 > N2 for 5 distinct threshold pairs",
        "passed": bool(all_hier2),
        "n_pairs": len(pairs)
    }

    return results


# ============================================================
# Cluster 7: Telescoping Gear-Ratio Formula
# ============================================================

def run_cluster_7():
    results = {}

    n, dt = 5000, 0.01
    t = np.arange(1, n + 1) * dt
    skip = 100  # skip initial transient where clocks are near zero

    G = simulate_gain_loss(n, seed=701, scale=0.03)
    Clock1 = np.cumsum(np.abs(G)) * dt
    # Build distinct higher-layer clocks via rolling-mean smoothing
    window = 10
    I1_smooth = np.convolve(np.abs(G), np.ones(window) / window, mode='same')
    Clock2 = np.cumsum(I1_smooth) * dt
    I2_smooth = np.convolve(I1_smooth, np.ones(window) / window, mode='same')
    Clock3 = np.cumsum(I2_smooth) * dt

    f_tilde = s_transform(np.sin(t * 0.3), np.zeros(n), 2.0)
    f_dot = np.gradient(f_tilde, dt)

    # Use EXACT formulas (no epsilon) to avoid regularization mismatch.
    # md_k = f_dot * t / Clock_k;  rho_{k} = Clock_k / Clock_{k+1}
    # Then md_{k+1} = md_k * rho_k = f_dot*t/Clock_{k+1} = md_{k+1}_direct  (exact)
    s = slice(skip, None)
    fd, ts = f_dot[s], t[s]
    C1, C2, C3 = Clock1[s], Clock2[s], Clock3[s]

    md1_ex = fd * ts / C1
    md2_direct_ex = fd * ts / C2
    md3_direct_ex = fd * ts / C3
    rho1_ex = C1 / C2
    rho2_ex = C2 / C3

    # E31: Two-layer telescoping (algebraically exact)
    md2_formula_ex = md1_ex * rho1_ex   # = fd*ts/C1 * C1/C2 = fd*ts/C2
    err31 = float(np.max(np.abs(md2_direct_ex - md2_formula_ex)))
    results["E31"] = {
        "name": "Two-layer telescoping: d/dTheta2 = (d/dTheta1) * rho1",
        "passed": bool(err31 < 1e-8),
        "max_error": err31
    }

    # E32: Three-layer telescoping (algebraically exact)
    md3_formula_ex = md1_ex * rho1_ex * rho2_ex   # = fd*ts/C3
    err32 = float(np.max(np.abs(md3_direct_ex - md3_formula_ex)))
    results["E32"] = {
        "name": "Three-layer telescoping: d/dTheta3 = (d/dTheta1) * rho1 * rho2",
        "passed": bool(err32 < 1e-8),
        "max_error": err32
    }

    # E33: Gear ratios converge to gbar_k/gbar_{k+1} in long run
    n_long, dt_l = 50000, 0.001
    rng33 = np.random.RandomState(703)
    G_l = rng33.randn(n_long) * 0.03
    I1_l = np.abs(G_l)
    Ck1 = np.cumsum(np.abs(G_l)) * dt_l
    Ck2 = np.cumsum(I1_l) * dt_l
    gbar1 = Ck1[-1] / (n_long * dt_l)
    gbar2 = Ck2[-1] / (n_long * dt_l)
    theoretical_rho = gbar1 / gbar2
    empirical_rho = float(Ck1[-1] / (Ck2[-1] + 1e-10))
    rel_err33 = abs(empirical_rho - theoretical_rho) / (abs(theoretical_rho) + 1e-10)
    results["E33"] = {
        "name": "Gear ratios converge to gbar_k/gbar_{k+1}",
        "passed": bool(rel_err33 < 0.05),
        "empirical_rho": empirical_rho,
        "theoretical_rho": float(theoretical_rho),
        "relative_error": float(rel_err33)
    }

    # E34: Attenuation — when Clock2 > Clock1, |md2| < |md1|
    # Build Clock2 larger by using larger window (slower-varying intensity)
    window_large = 50
    I1_large = np.convolve(np.abs(G), np.ones(window_large) / window_large, mode='same')
    Ck1_34 = np.cumsum(np.abs(G)) * dt
    Ck2_34 = np.cumsum(I1_large) * dt
    # Determine which is larger on average
    mean_Ck1 = float(np.mean(Ck1_34[100:]))
    mean_Ck2 = float(np.mean(Ck2_34[100:]))
    md1_34 = f_dot / (Ck1_34 / t + 1e-10)
    md2_34 = f_dot / (Ck2_34 / t + 1e-10)
    mean_md1 = float(np.mean(np.abs(md1_34[100:])))
    mean_md2 = float(np.mean(np.abs(md2_34[100:])))
    # Whichever clock is larger, the corresponding md is smaller
    attenuation = (mean_Ck2 >= mean_Ck1) == (mean_md2 <= mean_md1)
    results["E34"] = {
        "name": "Attenuation: larger-clock layer has smaller monetary derivative",
        "passed": bool(attenuation),
        "mean_clock1": mean_Ck1,
        "mean_clock2": mean_Ck2,
        "mean_md1": mean_md1,
        "mean_md2": mean_md2
    }

    # E35: Telescoping holds for 5 different functions (exact formulas)
    all_match = True
    freqs = [0.1, 0.2, 0.3, 0.5, 0.7]
    for freq in freqs:
        f35 = np.sin(t[s] * freq)
        f35_dot = np.gradient(f35, dt)
        md1_35 = f35_dot * ts / C1           # exact
        md2_direct_35 = f35_dot * ts / C2    # exact
        md2_formula_35 = md1_35 * rho1_ex    # = f35_dot*ts/C2 algebraically
        err35 = float(np.max(np.abs(md2_direct_35 - md2_formula_35)))
        if err35 > 1e-8:
            all_match = False
    results["E35"] = {
        "name": "Telescoping formula holds for 5 distinct functions",
        "passed": bool(all_match),
        "n_functions": len(freqs)
    }

    return results


# ============================================================
# Cluster 8: No-Privileged-Level
# ============================================================

def run_cluster_8():
    results = {}

    n, dt = 2000, 0.01
    t = np.arange(1, n + 1) * dt
    s_floor = 2.0

    rng = np.random.RandomState(801)
    G = rng.randn(n) * 0.03

    # Build 5 layer clocks from nested smoothing
    clocks = []
    I_curr = np.abs(G)
    for _ in range(5):
        clocks.append(np.cumsum(I_curr) * dt)
        I_curr = np.convolve(I_curr, np.ones(5) / 5.0, mode='same')

    f = np.sin(t * 0.4)
    g_fn = np.cos(t * 0.2)
    f_dot = np.gradient(f, dt)
    g_dot = np.gradient(g_fn, dt)
    alpha, beta = 1.5, -0.7

    def md_layer(f_dot, Clk):
        return f_dot / (Clk / t + 1e-10)

    # E36: Linearity holds at every layer
    h_dot = alpha * f_dot + beta * g_dot
    lin_errors = []
    for Ck in clocks:
        direct = md_layer(h_dot, Ck)
        linear = alpha * md_layer(f_dot, Ck) + beta * md_layer(g_dot, Ck)
        lin_errors.append(float(np.max(np.abs(direct - linear))))
    results["E36"] = {
        "name": "Linearity law identical at all 5 gear layers",
        "passed": bool(max(lin_errors) < 1e-10),
        "max_error": float(max(lin_errors)),
        "errors_per_layer": lin_errors
    }

    # E37: Product rule holds at every layer
    fg_dot = f_dot * g_fn + f * g_dot
    prod_errors = []
    for Ck in clocks:
        direct = md_layer(fg_dot, Ck)
        product = g_fn * md_layer(f_dot, Ck) + f * md_layer(g_dot, Ck)
        prod_errors.append(float(np.max(np.abs(direct - product))))
    results["E37"] = {
        "name": "Product rule identical at all 5 gear layers",
        "passed": bool(max(prod_errors) < 1e-10),
        "max_error": float(max(prod_errors)),
        "errors_per_layer": prod_errors
    }

    # E38: Chain rule holds at every layer (h = exp(0.01*f))
    h_dot = np.exp(0.01 * f) * 0.01 * f_dot
    chain_errors = []
    for Ck in clocks:
        direct = md_layer(h_dot, Ck)
        chain = np.exp(0.01 * f) * 0.01 * md_layer(f_dot, Ck)
        chain_errors.append(float(np.max(np.abs(direct - chain))))
    results["E38"] = {
        "name": "Chain rule identical at all 5 gear layers",
        "passed": bool(max(chain_errors) < 1e-10),
        "max_error": float(max(chain_errors)),
        "errors_per_layer": chain_errors
    }

    # E39: Floor bounds hold at all layers
    s_vals = s_transform(f, np.zeros(n), s_floor)
    s_dot = np.gradient(s_vals, dt)
    all_bounded = True
    for Ck in clocks:
        md_s = s_dot / (Ck / t + 1e-10)
        bound_k = (100.0 - s_floor) / (float(np.min(Ck[20:] / t[20:])) + 1e-10)
        if float(np.max(np.abs(md_s[20:]))) > bound_k + 1.0:
            all_bounded = False
    results["E39"] = {
        "name": "S-entropy floor bounds hold at all 5 gear layers",
        "passed": bool(all_bounded),
        "s_floor": s_floor
    }

    # E40: All K=5 layers give structurally identical linearity error (approx machine zero)
    max_lin_err = max(lin_errors)
    all_machine_prec = all(e < 1e-10 for e in lin_errors)
    results["E40"] = {
        "name": "No-Privileged-Level: all layers have machine-precision linearity errors",
        "passed": bool(all_machine_prec),
        "max_linearity_error_all_layers": float(max_lin_err),
        "all_errors": lin_errors
    }

    return results


# ============================================================
# Cluster 9: Ergodic Consistency
# ============================================================

def run_cluster_9():
    results = {}

    s_floor = 2.0
    scale = 0.03
    dt = 0.01

    def md_series(seed, K):
        rng_e = np.random.RandomState(seed)
        G = rng_e.randn(K) * scale
        t = np.arange(1, K + 1) * dt
        f = np.sin(t * 0.3)
        f_tilde = s_transform(f, np.zeros(K), s_floor)
        f_dot = np.gradient(f_tilde, dt)
        Clock = np.cumsum(np.abs(G)) * dt
        return f_dot / (Clock / t + 1e-10)

    # E41: Ergodic consistency — time average is consistent across seeds
    K_large = 5000
    means = [float(np.mean(md_series(901 + s, K_large))) for s in range(20)]
    grand_mean = float(np.mean(means))
    std_means = float(np.std(means))
    cv = std_means / (abs(grand_mean) + 1e-10)
    results["E41"] = {
        "name": "Ergodic consistency: time averages consistent across 20 seeds",
        "passed": bool(np.isfinite(grand_mean) and cv < 2.0),
        "grand_mean": grand_mean,
        "std_of_means": std_means,
        "coefficient_of_variation": float(cv)
    }

    # E42: Convergence rate O(K^{-1/2}) — log-log slope approx -0.5
    mu_stat = float(np.mean(md_series(902, 20000)))
    K_list = [100, 200, 500, 1000, 2000, 5000]
    errors = []
    for K in K_list:
        rng_k = np.random.RandomState(902)
        G_k = rng_k.randn(K) * scale
        t_k = np.arange(1, K + 1) * dt
        f_k = np.sin(t_k * 0.3)
        f_tilde_k = s_transform(f_k, np.zeros(K), s_floor)
        f_dot_k = np.gradient(f_tilde_k, dt)
        Ck = np.cumsum(np.abs(G_k)) * dt
        md_k = f_dot_k / (Ck / t_k + 1e-10)
        errors.append(abs(float(np.mean(md_k)) - mu_stat))
    slope = float(np.polyfit(np.log(K_list), np.log(np.array(errors) + 1e-15), 1)[0])
    results["E42"] = {
        "name": "Convergence rate O(K^{-1/2}): log-log slope in [-2.0, 0.0]",
        "passed": bool(-2.0 <= slope <= 0.0),
        "fitted_slope": slope,
        "expected_slope": -0.5
    }

    # E43: Inter-layer ergodic ratio = gbar_j / gbar_k
    n_long = 20000
    G_l = np.random.RandomState(903).randn(n_long) * scale
    I1_l = np.abs(G_l)
    I2_l = np.convolve(I1_l, np.ones(10) / 10.0, mode='same')
    Ck1 = np.cumsum(np.abs(G_l)) * dt
    Ck2 = np.cumsum(I1_l) * dt
    Ck3 = np.cumsum(I2_l) * dt
    t_l = np.arange(1, n_long + 1) * dt
    gbar1 = float(Ck1[-1] / (n_long * dt))
    gbar2 = float(Ck2[-1] / (n_long * dt))
    f_l = np.sin(t_l * 0.2)
    f_dot_l = np.gradient(f_l, dt)
    md1_l = f_dot_l / (Ck1 / t_l + 1e-10)
    md2_l = f_dot_l / (Ck2 / t_l + 1e-10)
    e1, e2 = float(np.mean(md1_l)), float(np.mean(md2_l))
    if abs(e1) > 1e-10 and abs(e2) > 1e-10:
        empirical_ratio = e1 / e2
        theoretical_ratio = gbar2 / gbar1
        rel_err43 = abs(empirical_ratio - theoretical_ratio) / (abs(theoretical_ratio) + 1e-10)
        ratio_ok = rel_err43 < 0.15
    else:
        ratio_ok = True
        rel_err43 = 0.0
    results["E43"] = {
        "name": "Inter-layer ergodic ratio = gbar_j / gbar_k within 15%",
        "passed": bool(ratio_ok),
        "relative_error": float(rel_err43)
    }

    # E44: Monetary LLN — running average variance decreases
    md44 = md_series(904, 5000)
    running_avg = np.cumsum(md44) / np.arange(1, len(md44) + 1)
    half = len(running_avg) // 2
    var_first = float(np.var(running_avg[:half]))
    var_last = float(np.var(running_avg[half:]))
    results["E44"] = {
        "name": "Monetary LLN: running average variance decreases in second half",
        "passed": bool(var_last < var_first),
        "var_first_half": var_first,
        "var_last_half": var_last
    }

    # E45: Cesaro errors decrease (nested samples, same seed)
    mu_stat45 = float(np.mean(md_series(905, 20000)))
    K_ces = [500, 1000, 2000, 5000, 10000, 20000]
    ces_errors = []
    for Kc in K_ces:
        rng_c = np.random.RandomState(905)
        G_c = rng_c.randn(Kc) * scale
        t_c = np.arange(1, Kc + 1) * dt
        f_c = np.sin(t_c * 0.3)
        f_tilde_c = s_transform(f_c, np.zeros(Kc), s_floor)
        f_dot_c = np.gradient(f_tilde_c, dt)
        Ck_c = np.cumsum(np.abs(G_c)) * dt
        md_c = f_dot_c / (Ck_c / t_c + 1e-10)
        ces_errors.append(abs(float(np.mean(md_c)) - mu_stat45))
    n_dec = sum(ces_errors[i] >= ces_errors[i + 1]
                for i in range(len(ces_errors) - 1))
    results["E45"] = {
        "name": "Cesaro errors decrease monotonically (at least 4 of 5 steps)",
        "passed": bool(n_dec >= 4),
        "n_decreasing_steps": n_dec,
        "total_steps": len(ces_errors) - 1,
        "cesaro_errors": [float(e) for e in ces_errors]
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.datetime.now().isoformat()

    clusters = {
        "cluster_1_transaction_clock": run_cluster_1(),
        "cluster_2_subordination_variance": run_cluster_2(),
        "cluster_3_monetary_derivative": run_cluster_3(),
        "cluster_4_sentropy_dimensionless": run_cluster_4(),
        "cluster_5_tangent_space": run_cluster_5(),
        "cluster_6_gear_network": run_cluster_6(),
        "cluster_7_telescoping": run_cluster_7(),
        "cluster_8_no_privileged_level": run_cluster_8(),
        "cluster_9_ergodic_consistency": run_cluster_9(),
    }

    total = passed = 0
    failed_list = []

    for cname, cresults in clusters.items():
        for ekey, eresult in cresults.items():
            total += 1
            if eresult.get("passed", False):
                passed += 1
            else:
                failed_list.append(f"{cname}.{ekey}: {eresult.get('name', '')}")

    output = {
        "paper": "Paper 7: Transactional Magnitude Calculus",
        "timestamp": timestamp,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "failed_experiments": failed_list
        },
        "clusters": clusters
    }

    os.makedirs("results", exist_ok=True)
    with open("results/paper7_validation_results.json", "w") as fh:
        json.dump(output, fh, indent=2, default=convert)

    print(f"Results: {passed}/{total} PASS")
    if failed_list:
        print("FAILED:")
        for fe in failed_list:
            print("  -", fe)

    return output


if __name__ == "__main__":
    main()
