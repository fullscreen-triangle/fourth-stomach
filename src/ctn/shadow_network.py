"""
Shadow Transaction Network (STN)
=================================

Transforms hierarchical transaction tree into harmonic network graph
through pattern coincidence detection.

Key Insight:
-----------
Just as molecules create network graphs when their harmonics coincide,
transaction patterns create "virtual edges" when their frequencies match.

These virtual/shadow transactions reveal the TRUE market structure:
- Systemic risk connections
- Hidden correlations
- Market intelligence
- Coordinated behavior detection

Based on molecular-gas-harmonic-timekeeping.tex Section:
"Harmonic Network Graph: From Tree to Graph Structure"

Mathematical Foundation:
-----------------------
When Transaction Pattern A observes frequency ω_A and independently
Transaction Pattern B observes frequency ω_B, if:

    |n·ω_A - m·ω_B| < ε_tolerance

then A and B are CONNECTED in frequency space, creating a graph edge
rather than separate tree branches.

This transforms:
    Tree: N nodes, N-1 edges, 1 path between any two nodes
    Graph: N nodes, >>N edges, multiple paths (redundancy!)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import scipy.fft as fft
from scipy.signal import find_peaks

from transaction_graph import CirculationTransactionNetwork, Node, Transaction


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

@dataclass
class TransactionPattern:
    """
    Frequency signature of a node's transaction behavior
    
    Analogous to molecular vibrational signature:
    - Fundamental frequency (daily/weekly/monthly rhythm)
    - Harmonics (sub-patterns at integer multiples)
    - Phase relationships (timing within cycle)
    - Amplitude (transaction volume at each frequency)
    """
    node_id: str
    
    # Frequency spectrum
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    phases: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Fundamental frequency
    omega_fundamental: float = 0.0
    
    # Harmonics (integer multiples of fundamental)
    harmonics: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # n: (amplitude, phase)
    
    # Statistical properties
    mean_transaction_amount: float = 0.0
    std_transaction_amount: float = 0.0
    transaction_count: int = 0
    
    # Temporal properties
    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # S-entropy coordinates
    s_knowledge: float = 0.0
    s_time: float = 0.0
    s_entropy: float = 0.0


@dataclass
class VirtualTransaction:
    """
    Shadow/virtual transaction representing pattern correlation
    
    NOT an actual money transfer!
    
    Represents: "Node A and Node B have coinciding transaction patterns,
                 suggesting they're influenced by the same market forces"
    
    Properties:
    - frequency_match: How closely their frequencies align
    - correlation_strength: Statistical correlation of patterns
    - harmonic_order: Which harmonics coincide (n, m)
    - betweenness: How critical this connection is to network
    """
    node_a: str
    node_b: str
    
    # Harmonic coincidence
    frequency_a: float
    frequency_b: float
    frequency_difference: float
    harmonic_order: Tuple[int, int]  # (n, m) such that n·ω_A ≈ m·ω_B
    
    # Correlation strength
    correlation: float = 0.0
    p_value: float = 1.0
    
    # Network properties
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Market intelligence
    risk_correlation: float = 0.0
    arbitrage_potential: float = 0.0
    influence_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Virtual transaction "amount" is correlation strength
        self.virtual_amount = abs(self.correlation) * 1000.0
        
        # "Resistance" is inverse of frequency match quality
        self.virtual_resistance = self.frequency_difference / (abs(self.frequency_a) + 1e-10)


class ShadowTransactionNetwork:
    """
    Analyzes transaction patterns to reveal hidden market structure
    
    Process:
    --------
    1. Extract transaction patterns (FFT of time series)
    2. Find harmonic coincidences between nodes
    3. Create virtual edges where patterns match
    4. Analyze resulting graph for market intelligence
    
    Complexity:
    ----------
    - Pattern extraction: O(N log N) per node (FFT)
    - Coincidence detection: O(N² H) where H = number of harmonics
    - Graph analysis: O(E log V) for various algorithms
    
    Key Insight:
    -----------
    The shadow network reveals the ACTUAL market structure,
    while the transaction network only shows the SURFACE flows.
    """
    
    def __init__(self, ctn: CirculationTransactionNetwork):
        self.ctn = ctn
        
        # Pattern analysis
        self.patterns: Dict[str, TransactionPattern] = {}
        
        # Shadow network (graph of pattern correlations)
        self.shadow_graph = nx.Graph()  # Undirected (correlation is symmetric)
        self.virtual_transactions: List[VirtualTransaction] = []
        
        # Harmonic tolerance (how close frequencies must be)
        self.epsilon_tolerance = 0.05  # 5% frequency difference
        
        # Analysis results
        self.hub_nodes: List[str] = []
        self.critical_connections: List[VirtualTransaction] = []
        self.risk_clusters: List[Set[str]] = []
        
    # ========================================================================
    # PATTERN EXTRACTION (FFT Analysis)
    # ========================================================================
    
    def extract_patterns(self, window_days: float = 30.0) -> Dict[str, TransactionPattern]:
        """
        Extract transaction patterns for all nodes using FFT
        
        For each node, analyze transaction history to find:
        - Fundamental frequency (dominant rhythm)
        - Harmonics (sub-patterns)
        - Phase relationships (timing within cycle)
        
        This is analogous to molecular spectroscopy!
        """
        current_time = datetime.now().timestamp()
        window_seconds = window_days * 24 * 3600
        
        for node_id, node in self.ctn.nodes.items():
            # Collect transactions for this node
            all_txs = node.inflows + node.outflows
            
            # Filter to window
            recent_txs = [
                tx for tx in all_txs
                if current_time - tx.timestamp < window_seconds
            ]
            
            if len(recent_txs) < 10:  # Need minimum data
                continue
            
            # Create time series
            timestamps = np.array([tx.timestamp for tx in recent_txs])
            amounts = np.array([
                tx.amount if tx in node.inflows else -tx.amount
                for tx in recent_txs
            ])
            
            # Normalize time to [0, window_seconds]
            timestamps_normalized = timestamps - timestamps.min()
            
            # Create uniform sampling grid (required for FFT)
            num_samples = 1024  # Power of 2 for efficiency
            time_grid = np.linspace(0, window_seconds, num_samples)
            
            # Interpolate onto grid
            amount_grid = np.interp(time_grid, timestamps_normalized, amounts)
            
            # Apply window function (reduce spectral leakage)
            window = np.hanning(num_samples)
            amount_windowed = amount_grid * window
            
            # Compute FFT
            fft_result = fft.fft(amount_windowed)
            frequencies = fft.fftfreq(num_samples, d=window_seconds/num_samples)
            
            # Take positive frequencies only
            positive_freq_idx = frequencies > 0
            frequencies_pos = frequencies[positive_freq_idx]
            amplitudes = np.abs(fft_result[positive_freq_idx])
            phases = np.angle(fft_result[positive_freq_idx])
            
            # Find peaks (harmonics)
            peaks, properties = find_peaks(amplitudes, height=amplitudes.max()*0.1)
            
            # Identify fundamental (lowest frequency with significant amplitude)
            if len(peaks) > 0:
                fundamental_idx = peaks[0]
                omega_fundamental = frequencies_pos[fundamental_idx]
            else:
                omega_fundamental = frequencies_pos[amplitudes.argmax()]
            
            # Extract harmonics
            harmonics = {}
            for n in range(1, 11):  # First 10 harmonics
                # Find frequency near n * fundamental
                target_freq = n * omega_fundamental
                nearby_idx = np.abs(frequencies_pos - target_freq).argmin()
                
                if np.abs(frequencies_pos[nearby_idx] - target_freq) / target_freq < 0.1:
                    harmonics[n] = (amplitudes[nearby_idx], phases[nearby_idx])
            
            # Create pattern
            pattern = TransactionPattern(
                node_id=node_id,
                frequencies=frequencies_pos,
                amplitudes=amplitudes,
                phases=phases,
                omega_fundamental=omega_fundamental,
                harmonics=harmonics,
                mean_transaction_amount=np.mean(np.abs(amounts)),
                std_transaction_amount=np.std(amounts),
                transaction_count=len(recent_txs),
                time_series=amount_grid,
                timestamps=time_grid
            )
            
            # Calculate S-entropy coordinates
            pattern.s_knowledge = len(harmonics) / 10.0  # More harmonics = more info
            pattern.s_time = 1.0 / (omega_fundamental + 1e-10)  # Period as time scale
            pattern.s_entropy = -np.sum(amplitudes * np.log(amplitudes + 1e-10))  # Shannon entropy
            
            self.patterns[node_id] = pattern
        
        return self.patterns
    
    # ========================================================================
    # HARMONIC COINCIDENCE DETECTION
    # ========================================================================
    
    def find_harmonic_coincidences(self) -> List[VirtualTransaction]:
        """
        Find nodes with coinciding harmonic frequencies
        
        Key Insight (from molecular timekeeping):
        =========================================
        When Node A has harmonic n·ω_A and Node B has harmonic m·ω_B,
        if |n·ω_A - m·ω_B| < ε, they are CONNECTED in frequency space.
        
        This creates a GRAPH, not a tree!
        
        Example:
        -------
        Node A: Daily rhythm (ω_A = 1/day)
            Harmonics: 1/day, 2/day, 3/day, ...
        
        Node B: 3-day rhythm (ω_B = 1/3day)
            Harmonics: 1/3day, 2/3day, 1/day, ...
        
        Coincidence: 1·ω_A = 3·ω_B = 1/day
        → Virtual edge A ↔ B with harmonic order (1, 3)
        """
        virtual_txs = []
        
        # Compare all pairs of nodes
        node_ids = list(self.patterns.keys())
        
        for i, node_a in enumerate(node_ids):
            pattern_a = self.patterns[node_a]
            
            for node_b in node_ids[i+1:]:
                pattern_b = self.patterns[node_b]
                
                # Check all harmonic pairs
                best_match = None
                min_difference = float('inf')
                
                for n, (amp_a, phase_a) in pattern_a.harmonics.items():
                    for m, (amp_b, phase_b) in pattern_b.harmonics.items():
                        freq_a = n * pattern_a.omega_fundamental
                        freq_b = m * pattern_b.omega_fundamental
                        
                        # Check if frequencies coincide
                        freq_diff = abs(freq_a - freq_b)
                        avg_freq = (abs(freq_a) + abs(freq_b)) / 2
                        
                        if avg_freq > 0:
                            relative_diff = freq_diff / avg_freq
                            
                            if relative_diff < self.epsilon_tolerance:
                                # Found coincidence!
                                if freq_diff < min_difference:
                                    min_difference = freq_diff
                                    best_match = (n, m, freq_a, freq_b, amp_a, amp_b, phase_a, phase_b)
                
                # Create virtual transaction if match found
                if best_match:
                    n, m, freq_a, freq_b, amp_a, amp_b, phase_a, phase_b = best_match
                    
                    # Calculate correlation strength
                    correlation = self._calculate_correlation(pattern_a, pattern_b)
                    
                    virtual_tx = VirtualTransaction(
                        node_a=node_a,
                        node_b=node_b,
                        frequency_a=freq_a,
                        frequency_b=freq_b,
                        frequency_difference=min_difference,
                        harmonic_order=(n, m),
                        correlation=correlation,
                        p_value=0.01  # Placeholder
                    )
                    
                    virtual_txs.append(virtual_tx)
        
        self.virtual_transactions = virtual_txs
        return virtual_txs
    
    def _calculate_correlation(self, pattern_a: TransactionPattern, 
                               pattern_b: TransactionPattern) -> float:
        """Calculate time-series correlation between two patterns"""
        # Ensure same length
        min_len = min(len(pattern_a.time_series), len(pattern_b.time_series))
        
        if min_len < 10:
            return 0.0
        
        ts_a = pattern_a.time_series[:min_len]
        ts_b = pattern_b.time_series[:min_len]
        
        # Normalize
        ts_a_norm = (ts_a - np.mean(ts_a)) / (np.std(ts_a) + 1e-10)
        ts_b_norm = (ts_b - np.mean(ts_b)) / (np.std(ts_b) + 1e-10)
        
        # Pearson correlation
        correlation = np.corrcoef(ts_a_norm, ts_b_norm)[0, 1]
        
        return correlation
    
    # ========================================================================
    # GRAPH CONSTRUCTION & ANALYSIS
    # ========================================================================
    
    def build_shadow_graph(self) -> nx.Graph:
        """
        Build shadow network from virtual transactions
        
        Transforms: Tree → Graph
        
        Tree structure:
            - Hierarchical transaction paths
            - Single path between nodes
            - Limited information flow
        
        Graph structure:
            - Pattern-based connections
            - Multiple paths (redundancy!)
            - Rich information flow
            - Hub detection possible
        """
        # Add all nodes
        for node_id, pattern in self.patterns.items():
            self.shadow_graph.add_node(
                node_id,
                pattern=pattern,
                omega=pattern.omega_fundamental,
                harmonics=len(pattern.harmonics)
            )
        
        # Add virtual edges
        for vtx in self.virtual_transactions:
            self.shadow_graph.add_edge(
                vtx.node_a,
                vtx.node_b,
                virtual_tx=vtx,
                weight=abs(vtx.correlation),
                frequency_a=vtx.frequency_a,
                frequency_b=vtx.frequency_b,
                harmonic_order=vtx.harmonic_order
            )
        
        # Calculate graph metrics
        self._calculate_network_metrics()
        
        return self.shadow_graph
    
    def _calculate_network_metrics(self):
        """Calculate graph metrics for network analysis"""
        if len(self.shadow_graph.nodes) == 0:
            return
        
        # Betweenness centrality (identifies hubs)
        betweenness = nx.betweenness_centrality(self.shadow_graph, weight='weight')
        
        # Update virtual transactions with betweenness
        for vtx in self.virtual_transactions:
            avg_betweenness = (betweenness.get(vtx.node_a, 0) + 
                              betweenness.get(vtx.node_b, 0)) / 2
            vtx.betweenness_centrality = avg_betweenness
        
        # Identify hubs (high betweenness)
        self.hub_nodes = sorted(
            betweenness.keys(),
            key=lambda n: betweenness[n],
            reverse=True
        )[:5]  # Top 5 hubs
        
        # Clustering coefficient (local connectivity)
        clustering = nx.clustering(self.shadow_graph, weight='weight')
        
        for vtx in self.virtual_transactions:
            avg_clustering = (clustering.get(vtx.node_a, 0) + 
                             clustering.get(vtx.node_b, 0)) / 2
            vtx.clustering_coefficient = avg_clustering
        
        # Identify critical connections (high betweenness edges)
        edge_betweenness = nx.edge_betweenness_centrality(self.shadow_graph, weight='weight')
        
        critical_edges = sorted(
            edge_betweenness.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 critical edges
        
        self.critical_connections = [
            vtx for vtx in self.virtual_transactions
            if ((vtx.node_a, vtx.node_b) in dict(critical_edges) or
                (vtx.node_b, vtx.node_a) in dict(critical_edges))
        ]
    
    # ========================================================================
    # MARKET INTELLIGENCE EXTRACTION
    # ========================================================================
    
    def detect_risk_clusters(self) -> List[Set[str]]:
        """
        Detect clusters of nodes with correlated risk
        
        Nodes in same cluster have:
        - Coinciding transaction patterns
        - High correlation
        - Connected through virtual transactions
        
        → They're exposed to same market forces!
        → Systemic risk if one fails!
        """
        # Use community detection algorithm
        communities = nx.community.greedy_modularity_communities(
            self.shadow_graph,
            weight='weight'
        )
        
        self.risk_clusters = [set(c) for c in communities]
        return self.risk_clusters
    
    def identify_arbitrage_opportunities(self) -> List[Dict]:
        """
        Find arbitrage opportunities through pattern analysis
        
        Opportunity exists when:
        - Nodes have coinciding patterns (correlated behavior)
        - But different actual transaction amounts
        - Price discrepancy between pattern-linked nodes!
        """
        opportunities = []
        
        for vtx in self.virtual_transactions:
            pattern_a = self.patterns[vtx.node_a]
            pattern_b = self.patterns[vtx.node_b]
            
            # Check if average amounts differ significantly
            amount_diff = abs(pattern_a.mean_transaction_amount - 
                            pattern_b.mean_transaction_amount)
            avg_amount = (pattern_a.mean_transaction_amount + 
                         pattern_b.mean_transaction_amount) / 2
            
            if avg_amount > 0:
                relative_diff = amount_diff / avg_amount
                
                if relative_diff > 0.2 and abs(vtx.correlation) > 0.7:
                    # Significant price difference + high correlation = arbitrage!
                    opportunity = {
                        'node_a': vtx.node_a,
                        'node_b': vtx.node_b,
                        'amount_a': pattern_a.mean_transaction_amount,
                        'amount_b': pattern_b.mean_transaction_amount,
                        'price_difference': amount_diff,
                        'correlation': vtx.correlation,
                        'arbitrage_potential': relative_diff * abs(vtx.correlation)
                    }
                    
                    vtx.arbitrage_potential = opportunity['arbitrage_potential']
                    opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x['arbitrage_potential'], reverse=True)
    
    def calculate_influence_scores(self) -> Dict[str, float]:
        """
        Calculate influence score for each node
        
        Influence = ability to affect other nodes through pattern correlation
        
        High influence nodes:
        - Have many virtual connections
        - Connected to hub nodes
        - High betweenness centrality
        - Their patterns affect many others
        """
        if len(self.shadow_graph.nodes) == 0:
            return {}
        
        # PageRank on shadow network (who influences whom)
        pagerank = nx.pagerank(self.shadow_graph, weight='weight')
        
        # Degree centrality (how many connections)
        degree = nx.degree_centrality(self.shadow_graph)
        
        # Eigenvector centrality (connected to important nodes)
        try:
            eigenvector = nx.eigenvector_centrality(self.shadow_graph, weight='weight', max_iter=1000)
        except:
            eigenvector = {n: 0.0 for n in self.shadow_graph.nodes}
        
        # Combined influence score
        influence = {}
        for node in self.shadow_graph.nodes:
            influence[node] = (
                0.4 * pagerank.get(node, 0) +
                0.3 * degree.get(node, 0) +
                0.3 * eigenvector.get(node, 0)
            )
        
        # Update virtual transactions
        for vtx in self.virtual_transactions:
            vtx.influence_score = (influence.get(vtx.node_a, 0) + 
                                  influence.get(vtx.node_b, 0)) / 2
        
        return influence
    
    def detect_coordinated_behavior(self, threshold: float = 0.9) -> List[Set[str]]:
        """
        Detect groups exhibiting coordinated behavior
        
        Coordination signatures:
        - Very high correlation (>0.9)
        - Similar phases in harmonics
        - Synchronized timing patterns
        
        Could indicate:
        - Market manipulation
        - Cartel behavior
        - Coordinated fraud
        - Or legitimate business relationships!
        """
        coordinated_groups = []
        
        # Find cliques of highly correlated nodes
        high_corr_graph = nx.Graph()
        
        for vtx in self.virtual_transactions:
            if abs(vtx.correlation) > threshold:
                high_corr_graph.add_edge(vtx.node_a, vtx.node_b)
        
        # Find maximal cliques
        cliques = list(nx.find_cliques(high_corr_graph))
        
        # Filter to significant size
        coordinated_groups = [set(c) for c in cliques if len(c) >= 3]
        
        return coordinated_groups
    
    # ========================================================================
    # REPORTING & VISUALIZATION
    # ========================================================================
    
    def generate_intelligence_report(self) -> Dict:
        """Generate comprehensive market intelligence report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'network_stats': {
                'total_nodes': len(self.patterns),
                'total_virtual_transactions': len(self.virtual_transactions),
                'graph_density': nx.density(self.shadow_graph),
                'avg_degree': np.mean([d for n, d in self.shadow_graph.degree()]) if len(self.shadow_graph) > 0 else 0,
                'clustering_coefficient': nx.average_clustering(self.shadow_graph, weight='weight') if len(self.shadow_graph) > 0 else 0
            },
            'hub_nodes': self.hub_nodes,
            'critical_connections': [
                {
                    'node_a': vtx.node_a,
                    'node_b': vtx.node_b,
                    'correlation': vtx.correlation,
                    'harmonic_order': vtx.harmonic_order,
                    'betweenness': vtx.betweenness_centrality
                }
                for vtx in self.critical_connections
            ],
            'risk_clusters': [list(cluster) for cluster in self.risk_clusters],
            'arbitrage_opportunities': self.identify_arbitrage_opportunities(),
            'influence_scores': self.calculate_influence_scores(),
            'coordinated_behavior': [list(group) for group in self.detect_coordinated_behavior()]
        }
        
        return report
    
    def get_node_intelligence(self, node_id: str) -> Dict:
        """Get detailed intelligence for specific node"""
        if node_id not in self.patterns:
            return {}
        
        pattern = self.patterns[node_id]
        
        # Find all virtual connections
        connections = [
            vtx for vtx in self.virtual_transactions
            if vtx.node_a == node_id or vtx.node_b == node_id
        ]
        
        # Calculate metrics
        influence = self.calculate_influence_scores().get(node_id, 0)
        
        intelligence = {
            'node_id': node_id,
            'pattern': {
                'fundamental_frequency': pattern.omega_fundamental,
                'period_days': 1.0 / (pattern.omega_fundamental * 86400) if pattern.omega_fundamental > 0 else float('inf'),
                'harmonics_count': len(pattern.harmonics),
                'transaction_count': pattern.transaction_count,
                'avg_amount': pattern.mean_transaction_amount,
                'std_amount': pattern.std_transaction_amount
            },
            's_coordinates': {
                's_knowledge': pattern.s_knowledge,
                's_time': pattern.s_time,
                's_entropy': pattern.s_entropy
            },
            'network_position': {
                'virtual_connections': len(connections),
                'influence_score': influence,
                'is_hub': node_id in self.hub_nodes
            },
            'correlations': [
                {
                    'partner': vtx.node_b if vtx.node_a == node_id else vtx.node_a,
                    'correlation': vtx.correlation,
                    'harmonic_order': vtx.harmonic_order
                }
                for vtx in sorted(connections, key=lambda v: abs(v.correlation), reverse=True)[:5]
            ]
        }
        
        return intelligence

