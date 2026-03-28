"""
Core validation framework for:
  Portfolio Optimisation as Trajectory Completion in Fuzzy Oscillatory Circuit Networks

Implements: fuzzy membership functions, portfolio circuit graph,
Kirchhoff constraint operators, backward trajectory (Viterbi),
trajectory completion operator, and S-entropy coordinates.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


# =============================================================================
# Fuzzy Membership Functions
# =============================================================================

@dataclass
class FuzzyState:
    """Trapezoidal fuzzy membership function with parameters a <= b <= c <= d.

    mu(x) = 0           if x < a or x > d
           = (x-a)/(b-a) if a <= x < b
           = 1           if b <= x <= c
           = (d-x)/(d-c) if c < x <= d
    """
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        assert self.a <= self.b <= self.c <= self.d, \
            f"Must have a<=b<=c<=d, got {self.a},{self.b},{self.c},{self.d}"

    def membership(self, x: float) -> float:
        if x < self.a or x > self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b > self.a else 1.0
        # x > self.c
        return (self.d - x) / (self.d - self.c) if self.d > self.c else 1.0

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        """Return [lower, upper] interval at alpha-level."""
        if alpha <= 0:
            return (self.a, self.d)
        if alpha >= 1:
            return (self.b, self.c)
        lower = self.a + alpha * (self.b - self.a)
        upper = self.d - alpha * (self.d - self.c)
        return (lower, upper)

    def support_width(self) -> float:
        return self.d - self.a

    def core_width(self) -> float:
        return self.c - self.b

    def centroid(self, n_points: int = 1000) -> float:
        xs = np.linspace(self.a, self.d, n_points)
        mus = np.array([self.membership(x) for x in xs])
        denom = np.sum(mus)
        if denom < 1e-15:
            return (self.a + self.d) / 2.0
        return np.sum(xs * mus) / denom

    def intersect(self, other: 'FuzzyState') -> 'FuzzyState':
        """Fuzzy intersection via alpha-cut interval intersection."""
        n_alpha = 50
        alphas = np.linspace(0, 1, n_alpha)
        lowers, uppers = [], []
        for alpha in alphas:
            l1, u1 = self.alpha_cut(alpha)
            l2, u2 = other.alpha_cut(alpha)
            lo = max(l1, l2)
            hi = min(u1, u2)
            if lo > hi:
                lo = hi = (lo + hi) / 2.0
            lowers.append(lo)
            uppers.append(hi)
        # Fit trapezoidal from alpha-cut envelope
        a_new = lowers[0]
        d_new = uppers[0]
        b_new = lowers[-1]
        c_new = uppers[-1]
        return FuzzyState(
            a=min(a_new, b_new),
            b=min(b_new, c_new),
            c=max(b_new, c_new),
            d=max(c_new, d_new),
        )

    @staticmethod
    def from_crisp(value: float, epsilon: float = 1e-10) -> 'FuzzyState':
        return FuzzyState(value - epsilon, value, value, value + epsilon)

    @staticmethod
    def from_interval(low: float, high: float) -> 'FuzzyState':
        """Maximum-entropy (flat) fuzzy state on [low, high]."""
        return FuzzyState(low, low, high, high)

    @staticmethod
    def from_estimate(center: float, half_width: float) -> 'FuzzyState':
        """Triangular fuzzy state centered at `center`."""
        return FuzzyState(
            center - half_width,
            center,
            center,
            center + half_width,
        )


# =============================================================================
# Portfolio Circuit Graph
# =============================================================================

@dataclass
class AssetNode:
    """An oscillatory asset node in the portfolio circuit."""
    name: str
    omega: float                          # characteristic frequency (rad/s)
    fuzzy_state: FuzzyState               # current fuzzy valuation
    is_boundary: bool = False             # True = market boundary node
    boundary_potential: Optional[float] = None  # fixed potential if boundary

    @property
    def potential(self) -> float:
        """Categorical depth = -log2(P(state)). Approximated by centroid."""
        if self.boundary_potential is not None:
            return self.boundary_potential
        c = self.fuzzy_state.centroid()
        return -np.log2(max(c, 1e-15))


class PortfolioCircuitGraph:
    """Portfolio modelled as an oscillatory circuit graph with fuzzy states."""

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: dict[str, AssetNode] = {}

    def add_asset(self, node: AssetNode):
        self.nodes[node.name] = node
        self.graph.add_node(node.name, node=node)

    def add_coupling(self, name_i: str, name_j: str, conductance: float):
        self.graph.add_edge(name_i, name_j, conductance=conductance)

    @property
    def N(self) -> int:
        return len(self.nodes)

    def adjacency_matrix(self) -> np.ndarray:
        names = sorted(self.nodes.keys())
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        A = np.zeros((n, n))
        for u, v, data in self.graph.edges(data=True):
            i, j = idx[u], idx[v]
            g = data['conductance']
            A[i, j] = g
            A[j, i] = g
        return A

    def laplacian_matrix(self) -> np.ndarray:
        A = self.adjacency_matrix()
        D = np.diag(A.sum(axis=1))
        return D - A

    def fiedler_value(self) -> float:
        """Algebraic connectivity = second-smallest eigenvalue of L."""
        L = self.laplacian_matrix()
        eigvals = np.sort(np.linalg.eigvalsh(L))
        if len(eigvals) < 2:
            return 0.0
        return float(eigvals[1])

    def sorted_node_names(self) -> list[str]:
        return sorted(self.nodes.keys())

    def get_fuzzy_states(self) -> dict[str, FuzzyState]:
        return {name: node.fuzzy_state for name, node in self.nodes.items()}

    def set_fuzzy_states(self, states: dict[str, FuzzyState]):
        for name, fs in states.items():
            self.nodes[name].fuzzy_state = fs


# =============================================================================
# Kirchhoff Constraint Operators
# =============================================================================

def apply_kcl(pcg: PortfolioCircuitGraph) -> dict[str, FuzzyState]:
    """T_KCL: fuzzy capital conservation at every node.

    For each node, compute the KCL-consistent interval from neighbour states
    and intersect with the current fuzzy state.
    """
    new_states = {}
    names = pcg.sorted_node_names()

    for name in names:
        node = pcg.nodes[name]
        if node.is_boundary:
            # Boundary nodes keep their state
            new_states[name] = node.fuzzy_state
            continue

        neighbours = list(pcg.graph.neighbors(name))
        if not neighbours:
            new_states[name] = node.fuzzy_state
            continue

        # KCL: weighted average of neighbour potentials propagated through
        # conductance. For fuzzy states, this becomes interval arithmetic on
        # alpha-cuts.
        n_alpha = 20
        alphas = np.linspace(0, 1, n_alpha)
        kcl_lowers = []
        kcl_uppers = []

        for alpha in alphas:
            weighted_sum_lo = 0.0
            weighted_sum_hi = 0.0
            total_g = 0.0
            for nb in neighbours:
                g = pcg.graph[name][nb]['conductance']
                lo, hi = pcg.nodes[nb].fuzzy_state.alpha_cut(alpha)
                weighted_sum_lo += g * lo
                weighted_sum_hi += g * hi
                total_g += g
            if total_g > 0:
                kcl_lowers.append(weighted_sum_lo / total_g)
                kcl_uppers.append(weighted_sum_hi / total_g)
            else:
                lo0, hi0 = node.fuzzy_state.alpha_cut(alpha)
                kcl_lowers.append(lo0)
                kcl_uppers.append(hi0)

        # Build KCL-consistent fuzzy state (trapezoidal approximation)
        kcl_fuzzy = FuzzyState(
            a=kcl_lowers[0],
            b=kcl_lowers[-1],
            c=kcl_uppers[-1],
            d=kcl_uppers[0],
        )
        # Intersect with current state
        new_states[name] = node.fuzzy_state.intersect(kcl_fuzzy)

    return new_states


def apply_kvl(pcg: PortfolioCircuitGraph) -> dict[str, FuzzyState]:
    """T_KVL: fuzzy no-arbitrage around cycles.

    For each independent cycle, enforce that the sum of potential differences
    is zero by tightening node fuzzy states.
    """
    states = {name: node.fuzzy_state for name, node in pcg.nodes.items()}

    # Find a cycle basis
    try:
        cycles = nx.cycle_basis(pcg.graph)
    except Exception:
        return states

    for cycle in cycles:
        if len(cycle) < 3:
            continue
        # For each cycle, compute the fuzzy sum of potential differences
        # and restrict states to make the sum contain 0
        n = len(cycle)
        # Simple approach: for each pair in cycle, tighten toward mean
        centroids = [pcg.nodes[name].fuzzy_state.centroid() for name in cycle]
        mean_val = np.mean(centroids)

        for i, name in enumerate(cycle):
            if pcg.nodes[name].is_boundary:
                continue
            fs = states[name]
            # Shrink toward cycle mean (KVL tightening)
            shrink = 0.1  # contraction factor per cycle
            new_center = fs.centroid() * (1 - shrink) + mean_val * shrink
            hw = fs.support_width() / 2.0 * (1 - shrink * 0.5)
            states[name] = FuzzyState(
                a=new_center - hw,
                b=new_center - hw * 0.3,
                c=new_center + hw * 0.3,
                d=new_center + hw,
            )

    return states


def apply_backward_trajectory(
    pcg: PortfolioCircuitGraph,
    reference_centroids: dict[str, float],
    strength: float = 0.2,
    min_support_width: float = 0.0,
) -> dict[str, FuzzyState]:
    """T_Back: backward trajectory constraint via simplified Viterbi.

    Pulls each node's fuzzy state toward its reference categorical address.
    min_support_width: irreducible epistemic uncertainty floor.
    """
    states = {}
    for name, node in pcg.nodes.items():
        if node.is_boundary:
            states[name] = node.fuzzy_state
            continue

        fs = node.fuzzy_state
        ref = reference_centroids.get(name, fs.centroid())
        current = fs.centroid()

        # Pull centroid toward reference
        new_center = current * (1 - strength) + ref * strength
        # Shrink support (backward trajectory restricts consistent values)
        hw = fs.support_width() / 2.0 * (1 - strength * 0.3)
        # Enforce minimum residual fuzziness
        hw = max(hw, min_support_width / 2.0)

        states[name] = FuzzyState(
            a=new_center - hw,
            b=new_center - hw * 0.3,
            c=new_center + hw * 0.3,
            d=new_center + hw,
        )

    return states


# =============================================================================
# Trajectory Completion Operator
# =============================================================================

def trajectory_completion_step(
    pcg: PortfolioCircuitGraph,
    reference_centroids: dict[str, float],
    back_strength: float = 0.2,
    min_support_width: float = 0.0,
) -> dict[str, FuzzyState]:
    """One iteration of T = T_Back o T_KVL o T_KCL."""
    # Step 1: KCL
    states_kcl = apply_kcl(pcg)
    pcg.set_fuzzy_states(states_kcl)

    # Step 2: KVL
    states_kvl = apply_kvl(pcg)
    pcg.set_fuzzy_states(states_kvl)

    # Step 3: Backward trajectory
    states_back = apply_backward_trajectory(
        pcg, reference_centroids, back_strength, min_support_width
    )
    pcg.set_fuzzy_states(states_back)

    return states_back


def hausdorff_distance(states_a: dict[str, FuzzyState],
                       states_b: dict[str, FuzzyState]) -> float:
    """Hausdorff product metric on fuzzy state tuples."""
    max_dist = 0.0
    for name in states_a:
        fa, fb = states_a[name], states_b[name]
        # Hausdorff distance between two intervals at alpha=0
        la, ua = fa.alpha_cut(0)
        lb, ub = fb.alpha_cut(0)
        dist = max(abs(la - lb), abs(ua - ub))
        max_dist = max(max_dist, dist)
    return max_dist


def run_trajectory_completion(
    pcg: PortfolioCircuitGraph,
    reference_centroids: dict[str, float],
    max_iter: int = 500,
    tol: float = 1e-8,
    back_strength: float = 0.2,
    min_support_width: float = 0.0,
) -> tuple[dict[str, FuzzyState], list[float]]:
    """Run trajectory completion to convergence. Returns (fixed_point, distances)."""
    distances = []
    for iteration in range(max_iter):
        old_states = pcg.get_fuzzy_states()
        new_states = trajectory_completion_step(
            pcg, reference_centroids, back_strength, min_support_width
        )
        d = hausdorff_distance(old_states, new_states)
        distances.append(d)
        if d < tol:
            break
    return pcg.get_fuzzy_states(), distances


# =============================================================================
# Graph Construction Utilities
# =============================================================================

def build_random_portfolio_graph(
    n_assets: int,
    n_boundary: int = 2,
    edge_prob: float = 0.4,
    seed: int = 42,
    conductance_range: tuple[float, float] = (0.1, 2.0),
    value_range: tuple[float, float] = (10.0, 200.0),
    fuzz_width: float = 20.0,
) -> PortfolioCircuitGraph:
    """Build a random portfolio circuit graph for testing."""
    rng = np.random.RandomState(seed)
    pcg = PortfolioCircuitGraph()

    names = [f"asset_{i}" for i in range(n_assets)]
    for i, name in enumerate(names):
        val = rng.uniform(*value_range)
        omega = rng.uniform(0.01, 1.0)
        is_bdy = i < n_boundary
        fs = FuzzyState.from_estimate(val, fuzz_width) if not is_bdy \
            else FuzzyState.from_crisp(val)
        node = AssetNode(
            name=name, omega=omega, fuzzy_state=fs,
            is_boundary=is_bdy,
            boundary_potential=-np.log2(max(val, 1e-15)) if is_bdy else None,
        )
        pcg.add_asset(node)

    # Add edges (ensure connectivity)
    for i in range(n_assets - 1):
        g = rng.uniform(*conductance_range)
        pcg.add_coupling(names[i], names[i + 1], g)

    # Add random extra edges
    for i in range(n_assets):
        for j in range(i + 2, n_assets):
            if rng.random() < edge_prob:
                g = rng.uniform(*conductance_range)
                pcg.add_coupling(names[i], names[j], g)

    return pcg


def build_portfolio_graph_with_fiedler(
    n_assets: int,
    target_connectivity: float,
    seed: int = 42,
    value_range: tuple[float, float] = (10.0, 200.0),
    fuzz_width: float = 20.0,
) -> PortfolioCircuitGraph:
    """Build a portfolio graph aiming for a specific algebraic connectivity.

    Higher target_connectivity -> more/stronger edges -> higher lambda_2.
    """
    rng = np.random.RandomState(seed)
    pcg = PortfolioCircuitGraph()

    names = [f"asset_{i}" for i in range(n_assets)]
    for i, name in enumerate(names):
        val = rng.uniform(*value_range)
        omega = rng.uniform(0.01, 1.0)
        is_bdy = i < 2
        fs = FuzzyState.from_estimate(val, fuzz_width) if not is_bdy \
            else FuzzyState.from_crisp(val)
        node = AssetNode(
            name=name, omega=omega, fuzzy_state=fs,
            is_boundary=is_bdy,
            boundary_potential=-np.log2(max(val, 1e-15)) if is_bdy else None,
        )
        pcg.add_asset(node)

    # Chain for connectivity
    for i in range(n_assets - 1):
        pcg.add_coupling(names[i], names[i + 1], target_connectivity)

    # Extra edges proportional to target
    edge_prob = min(0.9, target_connectivity / 5.0)
    for i in range(n_assets):
        for j in range(i + 2, n_assets):
            if rng.random() < edge_prob:
                pcg.add_coupling(names[i], names[j], target_connectivity * rng.uniform(0.5, 1.5))

    return pcg


# =============================================================================
# Markowitz utilities
# =============================================================================

def markowitz_weights(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                      target_return: Optional[float] = None) -> np.ndarray:
    """Minimum-variance portfolio (optionally with target return)."""
    n = len(expected_returns)
    inv_cov = np.linalg.inv(cov_matrix + 1e-8 * np.eye(n))
    ones = np.ones(n)

    if target_return is None:
        # Global minimum variance
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
    else:
        # With return constraint (two-fund separation)
        A = ones @ inv_cov @ ones
        B = ones @ inv_cov @ expected_returns
        C = expected_returns @ inv_cov @ expected_returns
        det = A * C - B * B
        if abs(det) < 1e-12:
            w = inv_cov @ ones / (ones @ inv_cov @ ones)
        else:
            lam = (C - target_return * B) / det
            gamma = (target_return * A - B) / det
            w = inv_cov @ (lam * ones + gamma * expected_returns)

    return w / w.sum()
