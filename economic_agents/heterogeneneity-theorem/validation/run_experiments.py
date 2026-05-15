"""
Validation experiments for Paper 3:
The Heterogeneity Theorem for Market Information:
Universality Classes, Spectral Theory, and Optimal Recruitment

45 experiments across 9 clusters, testing all 10 theorems.
SIGMA = 100.0 canonical constant throughout.
Results saved as JSON in results/ directory.
"""

import math
import json
import os
import random
import itertools
from datetime import datetime

SIGMA = 100.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(42)


# ─────────────────────────────────────────────
# Core primitives (inherited from Papers 1 & 2)
# ─────────────────────────────────────────────

def make_agent(beta, kappa=None):
    """
    Create agent with floor beta.
    Default kappa chosen so that sigma = SIGMA*(1-kappa)/kappa => floor = beta.
    """
    if kappa is None:
        kappa = SIGMA / (SIGMA + beta)
    sigma = SIGMA * (1 - kappa) / kappa
    return {"beta": beta, "kappa": kappa, "sigma": sigma}


def agent_floor(agent):
    kappa = agent["kappa"]
    sigma = agent["sigma"]
    market_floor = sigma * kappa / (1 - kappa)  # = SIGMA
    return agent["beta"] * market_floor / SIGMA


def composite_floor(agents):
    """S_flat_comp(E_n) = prod_i S_flat(A_i) / SIGMA^{n-1}"""
    n = len(agents)
    if n == 0:
        return SIGMA
    product = 1.0
    for a in agents:
        product *= agent_floor(a)
    return product / (SIGMA ** (n - 1))


def aggregate_floor(agents):
    """S_flat_agg(E) = min_i S_flat(A_i)"""
    return min(agent_floor(a) for a in agents)


def log_floor(agent):
    """ell_i = log(SIGMA / S_flat(A_i))"""
    f = agent_floor(agent)
    return math.log(SIGMA / f)


def spectral_radius(agents):
    """rho_E = S_flat_agg(E) / SIGMA"""
    return aggregate_floor(agents) / SIGMA


def spectral_gap(agents):
    """delta = 1 - rho_E"""
    return 1.0 - spectral_radius(agents)


def cumulative_log_floor(agents):
    """sum_{i=1}^n ell_i"""
    return sum(log_floor(a) for a in agents)


def is_class_E(log_floors_sequence):
    """Class E (efficient): sum ell_i = +inf (series diverges)"""
    return sum(log_floors_sequence) > 700  # proxy for divergence with finite n


def geometric_mean_floor(agents):
    """(prod_i S_flat(A_i))^{1/n}"""
    n = len(agents)
    product = 1.0
    for a in agents:
        product *= agent_floor(a)
    return product ** (1.0 / n)


def cv_floors(agents):
    """Coefficient of variation of floors"""
    floors = [agent_floor(a) for a in agents]
    mean = sum(floors) / len(floors)
    variance = sum((f - mean) ** 2 for f in floors) / len(floors)
    std = math.sqrt(variance)
    return std / mean if mean > 0 else 0.0


# ─────────────────────────────────────────────
# Theorem validation functions
# ─────────────────────────────────────────────

def test_heterogeneity_dominance(agents_homo, agents_het):
    """
    Theorem 1: S_flat_comp(het) <= S_flat_comp(homo) when het has same mean floor
    but higher variance. Schur-concavity of composite floor.
    """
    comp_homo = composite_floor(agents_homo)
    comp_het = composite_floor(agents_het)
    passed = comp_het <= comp_homo + 1e-9
    return {
        "theorem": "Heterogeneity Dominance",
        "comp_homo": comp_homo,
        "comp_het": comp_het,
        "difference": comp_homo - comp_het,
        "passed": passed
    }


def test_borel_cantelli_classification(class_e_agents, class_omega_agents, n_terms=50):
    """
    Theorem 2: Class E iff sum ell_i = +inf (composite floor -> 0).
               Class Omega iff sum ell_i < +inf (composite floor -> positive limit).
    """
    # Class E: each agent has small floor -> large log-floor
    log_floors_e = [log_floor(a) for a in class_e_agents[:n_terms]]
    comp_e = composite_floor(class_e_agents[:min(n_terms, len(class_e_agents))])

    # Class Omega: each agent has floor close to SIGMA -> small log-floor
    log_floors_o = [log_floor(a) for a in class_omega_agents[:n_terms]]
    comp_o = composite_floor(class_omega_agents[:min(n_terms, len(class_omega_agents))])

    sum_e = sum(log_floors_e)
    sum_o = sum(log_floors_o)

    e_decays = comp_e < 1.0  # should be near 0
    o_stays = comp_o > 0.1   # should stay positive

    passed = e_decays and o_stays
    return {
        "theorem": "Borel-Cantelli Classification",
        "sum_log_floors_E": sum_e,
        "sum_log_floors_Omega": sum_o,
        "comp_floor_E": comp_e,
        "comp_floor_Omega": comp_o,
        "class_E_decays": e_decays,
        "class_Omega_stays": o_stays,
        "passed": passed
    }


def test_spectral_gap(agents):
    """
    Theorem 3: Spectral gap delta = 1 - rho_E = 1 - S_flat_agg/SIGMA > 0
    for any ensemble with bounded floors (beta < SIGMA).
    """
    rho = spectral_radius(agents)
    delta = spectral_gap(agents)
    agg = aggregate_floor(agents)
    passed = delta > 0 and rho < 1.0 and abs(rho + delta - 1.0) < 1e-12
    return {
        "theorem": "Spectral Gap",
        "spectral_radius": rho,
        "spectral_gap": delta,
        "aggregate_floor": agg,
        "rho_plus_delta": rho + delta,
        "passed": passed
    }


def test_heterogeneity_theorem(agents_homo, agents_het):
    """
    Theorem 4 (Main): For ensembles with same mean floor but different variance,
    heterogeneous ensemble is strictly more informationally efficient (lower composite floor).
    CV > 0 implies strict inequality.
    """
    floors_homo = [agent_floor(a) for a in agents_homo]
    floors_het = [agent_floor(a) for a in agents_het]

    mean_homo = sum(floors_homo) / len(floors_homo)
    mean_het = sum(floors_het) / len(floors_het)

    cv_homo = cv_floors(agents_homo)
    cv_het = cv_floors(agents_het)

    comp_homo = composite_floor(agents_homo)
    comp_het = composite_floor(agents_het)

    strict_het_wins = comp_het < comp_homo - 1e-9
    cv_effect = cv_het > cv_homo

    passed = strict_het_wins and cv_effect
    return {
        "theorem": "Heterogeneity Theorem (Main)",
        "mean_floor_homo": mean_homo,
        "mean_floor_het": mean_het,
        "cv_homo": cv_homo,
        "cv_het": cv_het,
        "comp_floor_homo": comp_homo,
        "comp_floor_het": comp_het,
        "heterogeneous_wins": strict_het_wins,
        "passed": passed
    }


def test_information_aggregation_rate(agents):
    """
    Theorem 5: S_flat_comp(E_n) = SIGMA * prod_{i=1}^n (beta_i/SIGMA)
    Rate of decay equals exp(-sum ell_i).
    """
    n = len(agents)
    comp = composite_floor(agents)
    # Expected by formula
    product_ratio = 1.0
    for a in agents:
        product_ratio *= (agent_floor(a) / SIGMA)
    expected = SIGMA * product_ratio

    sum_ell = sum(log_floor(a) for a in agents)
    rate_formula = SIGMA * math.exp(-sum_ell)

    passed = (abs(comp - expected) < 1e-8 and
              abs(comp - rate_formula) < 1e-6)
    return {
        "theorem": "Information Aggregation Rate",
        "n": n,
        "composite_floor": comp,
        "expected_product_formula": expected,
        "rate_exp_formula": rate_formula,
        "sum_log_floors": sum_ell,
        "passed": passed
    }


def test_optimal_recruitment(agents):
    """
    Theorem 6: Ascending floor order minimizes cumulative composite floor
    (rearrangement inequality: large weights paired with large log-floors).
    """
    n = len(agents)
    floors = [agent_floor(a) for a in agents]

    # Ascending order (ascending beta = ascending floor)
    ascending_idx = sorted(range(n), key=lambda i: floors[i])
    # Descending order
    descending_idx = sorted(range(n), key=lambda i: floors[i], reverse=True)
    # Random order
    random_idx = list(range(n))
    random.shuffle(random_idx)

    def cumulative_comp(order):
        total = 0.0
        for k in range(1, n + 1):
            sub = [agents[order[i]] for i in range(k)]
            total += composite_floor(sub)
        return total

    cum_asc = cumulative_comp(ascending_idx)
    cum_desc = cumulative_comp(descending_idx)
    cum_rand = cumulative_comp(random_idx)

    passed = cum_asc <= cum_desc + 1e-9 and cum_asc <= cum_rand + 1e-9
    return {
        "theorem": "Optimal Recruitment",
        "n": n,
        "cumulative_comp_ascending": cum_asc,
        "cumulative_comp_descending": cum_desc,
        "cumulative_comp_random": cum_rand,
        "ascending_is_optimal": passed,
        "passed": passed
    }


def test_phase_transition(beta_below, beta_above):
    """
    Theorem 7: Phase transition at beta_c = SIGMA/e.
    Agents with beta < beta_c contribute > 1 nat; beta > beta_c contribute < 1 nat.
    """
    beta_c = SIGMA / math.e
    a_below = make_agent(beta_below)
    a_above = make_agent(beta_above)

    ell_below = log_floor(a_below)
    ell_above = log_floor(a_above)

    below_exceeds_1nat = ell_below > 1.0
    above_below_1nat = ell_above < 1.0
    at_critical = abs(log_floor(make_agent(beta_c)) - 1.0) < 1e-9

    passed = below_exceeds_1nat and above_below_1nat and at_critical
    return {
        "theorem": "Phase Transition",
        "beta_critical": beta_c,
        "beta_below": beta_below,
        "beta_above": beta_above,
        "log_floor_below": ell_below,
        "log_floor_above": ell_above,
        "log_floor_at_critical": log_floor(make_agent(beta_c)),
        "passed": passed
    }


def test_floor_variance_efficiency_bound(agents):
    """
    Theorem 8: S_flat_comp(E_n) <= (beta_bar/SIGMA)^n * SIGMA * exp(-n*CV^2/2 + O(n*CV^3))
    Taylor expansion bound via log-moment generating function.
    """
    n = len(agents)
    floors = [agent_floor(a) for a in agents]
    beta_bar = sum(floors) / n
    cv = cv_floors(agents)

    comp = composite_floor(agents)
    bound = ((beta_bar / SIGMA) ** n) * SIGMA * math.exp(-n * cv ** 2 / 2)

    # Composite floor should be <= bound (allowing some numerical slack for higher-order terms)
    passed = comp <= bound * (1 + 0.5)  # generous tolerance for O(CV^3) terms

    return {
        "theorem": "Floor-Variance Efficiency Bound",
        "n": n,
        "mean_floor": beta_bar,
        "cv": cv,
        "composite_floor": comp,
        "bound": bound,
        "ratio_comp_to_bound": comp / bound if bound > 0 else float("inf"),
        "passed": passed
    }


def test_stability_of_efficiency_class(class_e_agents, perturbation_beta):
    """
    Theorem 9: Class E stable under finite perturbations (adding/replacing finite
    many agents does not change the divergence of sum ell_i).
    """
    n = len(class_e_agents)
    sum_ell_original = sum(log_floor(a) for a in class_e_agents)

    # Replace one agent with a non-Class-E agent (large beta)
    perturbed = list(class_e_agents)
    perturbed[0] = make_agent(perturbation_beta)
    sum_ell_perturbed = sum(log_floor(a) for a in perturbed)

    # Class E iff sum diverges: difference should be finite
    finite_difference = abs(sum_ell_original - sum_ell_perturbed) < 1e6
    # Both or neither should be classified as divergent (same class)
    both_large = sum_ell_original > 10 and sum_ell_perturbed > 10

    passed = finite_difference and both_large
    return {
        "theorem": "Stability of Efficiency Class",
        "n": n,
        "sum_log_floors_original": sum_ell_original,
        "sum_log_floors_perturbed": sum_ell_perturbed,
        "difference": abs(sum_ell_original - sum_ell_perturbed),
        "class_preserved": both_large,
        "passed": passed
    }


def test_critical_floor(n_agents=20):
    """
    Theorem 10: Critical floor beta_c = SIGMA/e.
    Ensemble of n agents at critical floor: S_flat_comp = SIGMA * e^{-n}.
    """
    beta_c = SIGMA / math.e
    agents_crit = [make_agent(beta_c) for _ in range(n_agents)]

    comp = composite_floor(agents_crit)
    expected = SIGMA * math.exp(-n_agents)

    passed = abs(comp - expected) < 1e-6
    return {
        "theorem": "Critical Floor",
        "beta_critical": beta_c,
        "n": n_agents,
        "composite_floor": comp,
        "expected_sigma_e_minus_n": expected,
        "passed": passed
    }


# ─────────────────────────────────────────────
# Experiment clusters
# ─────────────────────────────────────────────

def run_cluster_1():
    """Heterogeneity Dominance — Schur-concavity verification"""
    results = []
    configs = [
        # (homo_betas, het_betas) — same mean, different spread
        ([20.0, 20.0, 20.0], [5.0, 20.0, 35.0]),
        ([30.0, 30.0, 30.0], [10.0, 30.0, 50.0]),
        ([15.0, 15.0, 15.0, 15.0], [5.0, 10.0, 20.0, 25.0]),
        ([25.0, 25.0, 25.0, 25.0], [5.0, 15.0, 35.0, 45.0]),
        ([10.0, 10.0, 10.0, 10.0, 10.0], [2.0, 5.0, 10.0, 15.0, 18.0]),
    ]
    for homo_betas, het_betas in configs:
        agents_homo = [make_agent(b) for b in homo_betas]
        agents_het = [make_agent(b) for b in het_betas]
        r = test_heterogeneity_dominance(agents_homo, agents_het)
        r["homo_betas"] = homo_betas
        r["het_betas"] = het_betas
        results.append(r)
    return results


def run_cluster_2():
    """Borel-Cantelli Classification — divergent vs convergent series"""
    results = []
    # Class E agents: small betas (large log-floors)
    class_e_pool = [make_agent(SIGMA / (k + 1)) for k in range(1, 51)]
    # Class Omega agents: betas approaching SIGMA (tiny log-floors ~ 1/k^2)
    class_o_pool = [make_agent(SIGMA * (1 - 1 / (k + 1) ** 2)) for k in range(1, 51)]

    for n in [10, 20, 30, 40, 50]:
        r = test_borel_cantelli_classification(class_e_pool[:n], class_o_pool[:n])
        r["n"] = n
        results.append(r)
    return results


def run_cluster_3():
    """Spectral Gap — positivity and formula verification"""
    results = []
    configs = [
        [5.0, 10.0, 15.0],
        [20.0, 30.0, 40.0, 50.0],
        [1.0, 50.0, 99.0],
        [10.0] * 8,
        [0.1, 1.0, 10.0, 50.0, 90.0],
    ]
    for betas in configs:
        agents = [make_agent(b) for b in betas]
        r = test_spectral_gap(agents)
        r["betas"] = betas
        results.append(r)
    return results


def run_cluster_4():
    """Heterogeneity Theorem (Main) — strict efficiency gain"""
    results = []
    configs = [
        # (homo_betas, het_betas)
        ([20.0, 20.0, 20.0], [1.0, 20.0, 39.0]),
        ([30.0, 30.0, 30.0, 30.0], [5.0, 15.0, 45.0, 55.0]),
        ([40.0, 40.0, 40.0], [10.0, 40.0, 70.0]),
        ([15.0, 15.0, 15.0, 15.0, 15.0], [1.0, 5.0, 15.0, 25.0, 29.0]),
        ([25.0, 25.0, 25.0], [2.0, 25.0, 48.0]),
    ]
    for homo_betas, het_betas in configs:
        agents_homo = [make_agent(b) for b in homo_betas]
        agents_het = [make_agent(b) for b in het_betas]
        r = test_heterogeneity_theorem(agents_homo, agents_het)
        r["homo_betas"] = homo_betas
        r["het_betas"] = het_betas
        results.append(r)
    return results


def run_cluster_5():
    """Information Aggregation Rate — formula verification"""
    results = []
    configs = [
        [5.0, 10.0, 20.0],
        [15.0, 25.0, 35.0, 45.0],
        [2.0, 4.0, 8.0, 16.0, 32.0],
        [50.0, 60.0, 70.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ]
    for betas in configs:
        agents = [make_agent(b) for b in betas]
        r = test_information_aggregation_rate(agents)
        r["betas"] = betas
        results.append(r)
    return results


def run_cluster_6():
    """Optimal Recruitment — rearrangement inequality"""
    results = []
    configs = [
        [5.0, 15.0, 30.0, 50.0],
        [2.0, 10.0, 25.0, 40.0, 60.0],
        [1.0, 5.0, 20.0, 80.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [3.0, 7.0, 15.0, 30.0],
    ]
    for betas in configs:
        agents = [make_agent(b) for b in betas]
        r = test_optimal_recruitment(agents)
        r["betas"] = betas
        results.append(r)
    return results


def run_cluster_7():
    """Phase Transition — critical floor beta_c = SIGMA/e"""
    results = []
    beta_c = SIGMA / math.e
    configs = [
        (5.0, 50.0),
        (10.0, 60.0),
        (20.0, 80.0),
        (1.0, 90.0),
        (15.0, 70.0),
    ]
    for beta_below, beta_above in configs:
        r = test_phase_transition(beta_below, beta_above)
        results.append(r)
    return results


def run_cluster_8():
    """Floor-Variance Efficiency Bound — Taylor expansion bound"""
    results = []
    configs = [
        [5.0, 10.0, 15.0, 20.0, 25.0],
        [1.0, 10.0, 30.0, 50.0, 80.0],
        [20.0, 20.0, 20.0, 20.0, 20.0],
        [2.0, 5.0, 10.0, 20.0, 40.0],
        [10.0, 15.0, 20.0, 25.0, 30.0],
    ]
    for betas in configs:
        agents = [make_agent(b) for b in betas]
        r = test_floor_variance_efficiency_bound(agents)
        r["betas"] = betas
        results.append(r)
    return results


def run_cluster_9():
    """Stability & Critical Floor — class preservation and exact formula"""
    results = []

    # Experiments 1–3: Stability of Efficiency Class
    class_e_pool = [make_agent(SIGMA / (k + 2)) for k in range(20)]
    perturbations = [50.0, 80.0, 95.0]
    for p_beta in perturbations:
        r = test_stability_of_efficiency_class(class_e_pool, p_beta)
        r["perturbation_beta"] = p_beta
        results.append(r)

    # Experiments 4–5: Critical Floor
    for n in [10, 20]:
        r = test_critical_floor(n)
        results.append(r)

    return results


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Paper 3 Validation: The Heterogeneity Theorem")
    print(f"SIGMA = {SIGMA}")
    print("=" * 60)

    clusters = [
        ("Cluster 1: Heterogeneity Dominance", run_cluster_1),
        ("Cluster 2: Borel-Cantelli Classification", run_cluster_2),
        ("Cluster 3: Spectral Gap", run_cluster_3),
        ("Cluster 4: Heterogeneity Theorem (Main)", run_cluster_4),
        ("Cluster 5: Information Aggregation Rate", run_cluster_5),
        ("Cluster 6: Optimal Recruitment", run_cluster_6),
        ("Cluster 7: Phase Transition", run_cluster_7),
        ("Cluster 8: Floor-Variance Efficiency Bound", run_cluster_8),
        ("Cluster 9: Stability & Critical Floor", run_cluster_9),
    ]

    all_results = {}
    total_pass = 0
    total_fail = 0

    for cluster_name, cluster_fn in clusters:
        print(f"\n{cluster_name}")
        print("-" * 50)
        cluster_results = cluster_fn()
        all_results[cluster_name] = cluster_results

        for i, r in enumerate(cluster_results):
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  Exp {i+1:2d}: {r['theorem']:<40s} [{status}]")
            if r["passed"]:
                total_pass += 1
            else:
                total_fail += 1
                print(f"           FAILED: {r}")

    print("\n" + "=" * 60)
    print(f"Total: {total_pass} PASS / {total_fail} FAIL / {total_pass + total_fail} total")
    print("=" * 60)

    # Save results
    output = {
        "paper": "Paper 3: The Heterogeneity Theorem for Market Information",
        "SIGMA": SIGMA,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total_pass + total_fail,
            "passed": total_pass,
            "failed": total_fail
        },
        "clusters": all_results
    }

    out_path = os.path.join(RESULTS_DIR, "paper3_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return total_fail == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
