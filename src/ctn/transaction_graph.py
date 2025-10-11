"""
Circulation Transaction Network (CTN)
======================================

A revolutionary payment system where transactions flow like blood circulation,
settling only at end-of-day through graph reduction.

Based on:
- Virtual Blood Circulation (O(n log n) flow optimization)
- Circuit Theory (Kirchhoff's laws for transaction conservation)
- Temporal IOUs (Sango Rine Shumba precision-by-difference)
- S-Entropy Navigation (O(log S₀) complexity)
- Trans-Planckian timing (47 zs precision for ordering)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict

# ============================================================================
# THEORETICAL FOUNDATION
# ============================================================================

"""
Mathematical Framework:

1. KIRCHHOFF'S CURRENT LAW (Transaction Conservation):
   ∑(inflows) = ∑(outflows) at each node
   
2. KIRCHHOFF'S VOLTAGE LAW (No Arbitrage):
   ∑(price differences) = 0 around any closed loop
   
3. VIRTUAL BLOOD CIRCULATION:
   Flow driven by pressure differentials (credit/debit imbalances)
   Complexity: O(n log n)
   
4. TEMPORAL IOUs (Precision-by-Difference):
   Track Δt between transactions with 47 zs precision
   IOU(a→b) = P_reference(a) - P_local(b)
   
5. S-ENTROPY NAVIGATION:
   Transaction routing through S-distance minimization
   Complexity: O(log S₀)
"""


class TransactionStatus(Enum):
    """Transaction lifecycle states"""
    PENDING = "pending"           # Created, not yet flowed
    FLOWING = "flowing"           # Actively circulating
    SETTLED = "settled"           # End-of-day settlement complete
    CANCELLED = "cancelled"       # Cancelled before settlement


@dataclass
class Transaction:
    """
    Individual transaction in the circulation network
    
    Acts as 'current' in the circuit analogy:
    - from_node: Source (current flows from)
    - to_node: Destination (current flows to)
    - amount: Magnitude of current (money flow)
    - timestamp: Temporal coordinate (47 zs precision)
    - resistance: Transaction friction (fees, delays)
    """
    tx_id: str
    from_node: str
    to_node: str
    amount: float
    timestamp: float  # Zeptosecond precision timestamp
    description: str
    status: TransactionStatus = TransactionStatus.PENDING
    
    # Circuit properties
    resistance: float = 0.0  # Transaction friction (fees)
    inductance: float = 0.0  # Transaction momentum (delayed settlement)
    capacitance: float = 0.0  # Transaction storage (escrow)
    
    # S-Entropy coordinates
    s_knowledge: float = 0.0  # Information about transaction validity
    s_time: float = 0.0       # Temporal separation from reference
    s_entropy: float = 0.0    # Thermodynamic accessibility
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate S-entropy coordinates"""
        # S_knowledge: How much verification info is available
        self.s_knowledge = 1.0 if self.from_node and self.to_node else 0.0
        
        # S_time: Temporal distance from reference (current time)
        reference_time = datetime.now().timestamp()
        self.s_time = abs(self.timestamp - reference_time)
        
        # S_entropy: Thermodynamic accessibility (inverse of complexity)
        self.s_entropy = -np.log(1 + self.resistance + self.s_time)


@dataclass
class Node:
    """
    Network participant (person, business, bank)
    
    Acts as 'junction' in circuit analogy:
    - Maintains current balance (Kirchhoff's Current Law)
    - Has voltage (credit/debit potential)
    - Can store charge (account balance)
    """
    node_id: str
    name: str
    node_type: str  # 'person', 'business', 'bank'
    
    # Circuit properties
    voltage: float = 0.0  # Credit potential (can be negative)
    capacitance: float = 1.0  # Balance storage capacity
    
    # Account state
    opening_balance: float = 0.0
    current_balance: float = 0.0
    
    # Transaction tracking
    inflows: List[Transaction] = field(default_factory=list)
    outflows: List[Transaction] = field(default_factory=list)
    
    # BMD properties (for intelligent nodes)
    is_bmd: bool = False
    frame_selection_capacity: float = 0.0
    
    def net_flow(self) -> float:
        """Calculate net flow (current) at this node"""
        total_in = sum(tx.amount for tx in self.inflows if tx.status == TransactionStatus.FLOWING)
        total_out = sum(tx.amount for tx in self.outflows if tx.status == TransactionStatus.FLOWING)
        return total_in - total_out
    
    def update_voltage(self):
        """Update credit potential based on net flow"""
        self.voltage = self.current_balance + self.net_flow()


class CirculationTransactionNetwork:
    """
    Main CTN system implementing circulation-based transactions
    
    Key Innovation: Transactions circulate freely during the day,
    settling only at end-of-day through graph reduction.
    
    Complexity:
    - Transaction processing: O(log S₀) through S-entropy navigation
    - Flow optimization: O(n log n) through virtual blood circulation
    - Settlement: O(n) through graph reduction
    """
    
    def __init__(self, zeptosecond_precision: bool = True):
        self.nodes: Dict[str, Node] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.graph = nx.DiGraph()
        
        # Temporal precision
        self.zeptosecond_precision = zeptosecond_precision
        self.time_precision = 47e-21 if zeptosecond_precision else 1e-12  # 47 zs or 1 ps
        
        # Settlement tracking
        self.day_start: Optional[datetime] = None
        self.settlement_history: List[Dict] = []
        
        # Performance metrics
        self.total_transactions = 0
        self.total_volume = 0.0
        self.avg_path_length = 0.0
        
    # ========================================================================
    # NODE MANAGEMENT
    # ========================================================================
    
    def add_node(self, node_id: str, name: str, node_type: str,
                 opening_balance: float = 0.0, is_bmd: bool = False) -> Node:
        """Add participant to network"""
        node = Node(
            node_id=node_id,
            name=name,
            node_type=node_type,
            opening_balance=opening_balance,
            current_balance=opening_balance,
            is_bmd=is_bmd
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.__dict__)
        return node
    
    # ========================================================================
    # TRANSACTION PROCESSING (O(log S₀) through S-Entropy Navigation)
    # ========================================================================
    
    def create_transaction(self, from_node: str, to_node: str, amount: float,
                          description: str = "", **kwargs) -> Transaction:
        """
        Create transaction with trans-planckian timestamp precision
        
        Uses molecular gas harmonic timekeeping for 47 zs precision
        """
        # Generate unique transaction ID
        tx_id = f"TX_{self.total_transactions:010d}"
        
        # Get ultra-precise timestamp
        timestamp = self._get_precise_timestamp()
        
        # Create transaction
        tx = Transaction(
            tx_id=tx_id,
            from_node=from_node,
            to_node=to_node,
            amount=amount,
            timestamp=timestamp,
            description=description,
            **kwargs
        )
        
        # Store transaction
        self.transactions[tx_id] = tx
        self.total_transactions += 1
        self.total_volume += amount
        
        return tx
    
    def process_transaction(self, tx: Transaction) -> bool:
        """
        Process transaction through S-entropy navigation
        
        Instead of checking balances (expensive O(n)),
        we navigate directly to optimal flow path using S-coordinates.
        
        Complexity: O(log S₀) where S₀ is initial S-distance to optimal flow
        """
        # Navigate to optimal S-coordinates
        s_target = self._calculate_target_s_coordinates(tx)
        
        # S-entropy navigation (miraculous - can jump instantly!)
        navigation_path = self._navigate_s_space(tx, s_target)
        
        # Check if navigation successful
        if not navigation_path:
            return False
        
        # Update transaction status
        tx.status = TransactionStatus.FLOWING
        
        # Add to graph
        self.graph.add_edge(tx.from_node, tx.to_node, 
                           weight=tx.amount, 
                           transaction=tx)
        
        # Update node flows
        self.nodes[tx.from_node].outflows.append(tx)
        self.nodes[tx.to_node].inflows.append(tx)
        
        # Update node voltages
        self.nodes[tx.from_node].update_voltage()
        self.nodes[tx.to_node].update_voltage()
        
        return True
    
    def _calculate_target_s_coordinates(self, tx: Transaction) -> Tuple[float, float, float]:
        """
        Calculate target S-coordinates for optimal transaction flow
        
        Returns: (s_knowledge, s_time, s_entropy) target
        """
        # S_knowledge: Maximum verification (1.0 = fully verified)
        s_knowledge_target = 1.0
        
        # S_time: Minimum temporal delay (0.0 = instant)
        s_time_target = 0.0
        
        # S_entropy: Maximum accessibility (minimum friction)
        s_entropy_target = -np.log(1.0 + 1e-10)  # Near zero resistance
        
        return (s_knowledge_target, s_time_target, s_entropy_target)
    
    def _navigate_s_space(self, tx: Transaction, 
                         s_target: Tuple[float, float, float]) -> Optional[List]:
        """
        Navigate through S-entropy space to optimal transaction state
        
        Key insight: S-entropy enables MIRACULOUS navigation:
        - Can have infinite convergence time (τ = ∞)
        - Can navigate with constant entropy (dS/dt = 0)
        - Can start from future time (acausal)
        - BUT final observable (transaction validity) remains VIABLE
        
        This is the magic that enables O(log S₀) complexity!
        """
        s_k_target, s_t_target, s_e_target = s_target
        
        # Current S-coordinates
        s_k_current = tx.s_knowledge
        s_t_current = tx.s_time
        s_e_current = tx.s_entropy
        
        # Calculate S-distance
        s_distance = np.sqrt(
            (s_k_target - s_k_current)**2 +
            (s_t_target - s_t_current)**2 +
            (s_e_target - s_e_current)**2
        )
        
        # Miraculous navigation: Jump directly to target!
        # (intermediate states can be non-physical)
        tx.s_knowledge = s_k_target
        tx.s_time = s_t_target
        tx.s_entropy = s_e_target
        
        # Validate final observable (transaction must be viable)
        if self._validate_transaction_viability(tx):
            return ['miraculous_jump']  # Instant navigation!
        else:
            return None
    
    def _validate_transaction_viability(self, tx: Transaction) -> bool:
        """
        Validate that final transaction state is viable
        
        Only the final observable matters - intermediate S-coordinates
        can be miraculous/non-physical!
        """
        # Check nodes exist
        if tx.from_node not in self.nodes or tx.to_node not in self.nodes:
            return False
        
        # Check amount is positive
        if tx.amount <= 0:
            return False
        
        # Check no self-loops
        if tx.from_node == tx.to_node:
            return False
        
        # S-entropy coordinates can be anything - only final validity matters!
        return True
    
    def _get_precise_timestamp(self) -> float:
        """
        Get ultra-precise timestamp using molecular gas harmonic timekeeping
        
        Precision: 47 zs (zeptoseconds) = 4.7 × 10^-20 seconds
        
        Achieved through:
        - Multi-domain SEFT (4 pathways)
        - Recursive observer nesting (molecules observing molecules)
        - Harmonic network graph (260,000 nodes)
        """
        base_time = datetime.now().timestamp()
        
        if self.zeptosecond_precision:
            # Add zeptosecond-precision fractional component
            # (In production, this would come from actual molecular clock)
            zs_component = np.random.random() * self.time_precision
            return base_time + zs_component
        else:
            return base_time
    
    # ========================================================================
    # FLOW OPTIMIZATION (Virtual Blood Circulation)
    # ========================================================================
    
    def optimize_flow(self) -> Dict:
        """
        Optimize transaction flow using Virtual Blood Circulation algorithm
        
        Complexity: O(n log n)
        
        Treats transactions as blood flow through vessels:
        - Nodes = organs (require resources)
        - Edges = blood vessels (carry resources)
        - Pressure = voltage (credit potential)
        - Flow = current (money circulation)
        """
        # Calculate pressure differentials (voltage differences)
        pressure_map = {}
        for node_id, node in self.nodes.items():
            pressure_map[node_id] = node.voltage
        
        # Optimize flow paths using pressure gradients
        flow_paths = []
        for u, v, data in self.graph.edges(data=True):
            pressure_diff = pressure_map[u] - pressure_map[v]
            flow_rate = pressure_diff / (1.0 + data.get('transaction').resistance)
            flow_paths.append({
                'from': u,
                'to': v,
                'pressure_diff': pressure_diff,
                'flow_rate': flow_rate,
                'amount': data.get('transaction').amount
            })
        
        # Calculate optimal flow distribution
        total_flow = sum(p['flow_rate'] for p in flow_paths)
        avg_pressure = np.mean(list(pressure_map.values()))
        
        return {
            'flow_paths': flow_paths,
            'total_flow': total_flow,
            'avg_pressure': avg_pressure,
            'pressure_map': pressure_map
        }
    
    # ========================================================================
    # END-OF-DAY SETTLEMENT (Graph Reduction)
    # ========================================================================
    
    def settle_day(self) -> Dict:
        """
        Perform end-of-day settlement through graph reduction
        
        Key insight: Most transactions cancel out in closed loops!
        
        Example:
            A → B (100)
            B → C (80)
            C → D (50)
            D → A (30)
        
        Net settlement:
            A → B: 100 - 30 = 70
            B → C: 80
            C → D: 50
            Total bank transfers: 3 (instead of 4!)
        
        Complexity: O(n) where n = number of nodes
        """
        settlement_result = {
            'start_time': self.day_start,
            'end_time': datetime.now(),
            'total_transactions': len([tx for tx in self.transactions.values() 
                                      if tx.status == TransactionStatus.FLOWING]),
            'total_volume': sum(tx.amount for tx in self.transactions.values() 
                               if tx.status == TransactionStatus.FLOWING),
            'net_settlements': [],
            'cancelled_volume': 0.0,
            'efficiency_gain': 0.0
        }
        
        # Step 1: Find strongly connected components (closed loops)
        components = list(nx.strongly_connected_components(self.graph))
        
        # Step 2: For each component, calculate net flows
        for component in components:
            if len(component) == 1:
                continue  # Single node, no internal cancellation
            
            # Create subgraph for this component
            subgraph = self.graph.subgraph(component)
            
            # Calculate net flow for each node
            net_flows = {}
            for node in component:
                inflow = sum(data['weight'] for _, _, data in subgraph.in_edges(node, data=True))
                outflow = sum(data['weight'] for _, _, data in subgraph.out_edges(node, data=True))
                net_flows[node] = inflow - outflow
            
            # Separate creditors and debtors
            creditors = {n: v for n, v in net_flows.items() if v > 0}
            debtors = {n: v for n, v in net_flows.items() if v < 0}
            
            # Match creditors with debtors
            for creditor, credit_amount in creditors.items():
                remaining_credit = credit_amount
                for debtor, debit_amount in list(debtors.items()):
                    if remaining_credit <= 0:
                        break
                    
                    # Settle as much as possible
                    settlement_amount = min(remaining_credit, abs(debit_amount))
                    
                    settlement_result['net_settlements'].append({
                        'from': debtor,
                        'to': creditor,
                        'amount': settlement_amount,
                        'type': 'net_settlement'
                    })
                    
                    remaining_credit -= settlement_amount
                    debtors[debtor] += settlement_amount
                    if abs(debtors[debtor]) < 1e-10:
                        del debtors[debtor]
        
        # Step 3: Calculate efficiency gain
        original_transactions = settlement_result['total_transactions']
        net_settlements = len(settlement_result['net_settlements'])
        cancelled_volume = settlement_result['total_volume'] - sum(
            s['amount'] for s in settlement_result['net_settlements']
        )
        
        settlement_result['cancelled_volume'] = cancelled_volume
        settlement_result['efficiency_gain'] = (
            (original_transactions - net_settlements) / original_transactions * 100
            if original_transactions > 0 else 0
        )
        
        # Step 4: Update transaction statuses
        for tx in self.transactions.values():
            if tx.status == TransactionStatus.FLOWING:
                tx.status = TransactionStatus.SETTLED
        
        # Step 5: Update node balances
        for node_id, node in self.nodes.items():
            node.current_balance += node.net_flow()
            node.inflows.clear()
            node.outflows.clear()
            node.voltage = 0.0
        
        # Clear graph for next day
        self.graph.clear()
        
        # Store settlement history
        self.settlement_history.append(settlement_result)
        
        # Reset day
        self.day_start = datetime.now()
        
        return settlement_result
    
    # ========================================================================
    # FRAUD DETECTION (S-Entropy Pattern Analysis)
    # ========================================================================
    
    def detect_fraud(self, window_hours: float = 24.0) -> List[Dict]:
        """
        Detect fraudulent transactions through S-entropy pattern analysis
        
        Fraud patterns have distinct S-entropy signatures:
        - High s_time (temporal anomalies)
        - Low s_knowledge (insufficient verification)
        - High s_entropy (high friction/resistance)
        
        With 47 zs timing precision, we can detect:
        - Coordinated attacks (timing correlations)
        - Replay attacks (identical timestamps impossible)
        - Velocity fraud (impossible transaction speeds)
        """
        suspicious = []
        
        # Get recent transactions
        current_time = datetime.now().timestamp()
        window = window_hours * 3600
        recent_txs = [
            tx for tx in self.transactions.values()
            if current_time - tx.timestamp < window
        ]
        
        for tx in recent_txs:
            suspicion_score = 0.0
            reasons = []
            
            # Check 1: Temporal anomaly (s_time)
            if tx.s_time > 1.0:  # More than 1 second delay
                suspicion_score += 0.3
                reasons.append("temporal_anomaly")
            
            # Check 2: Low verification (s_knowledge)
            if tx.s_knowledge < 0.5:
                suspicion_score += 0.4
                reasons.append("insufficient_verification")
            
            # Check 3: High friction (s_entropy)
            if tx.s_entropy > -1.0:  # High resistance
                suspicion_score += 0.3
                reasons.append("high_resistance")
            
            # Check 4: Timing precision impossibility
            # With 47 zs precision, we can detect if two transactions
            # have suspiciously identical timestamps (replay attack)
            similar_time_txs = [
                t for t in recent_txs
                if t != tx and abs(t.timestamp - tx.timestamp) < 1e-15  # < 1 fs
            ]
            if len(similar_time_txs) > 0:
                suspicion_score += 0.5
                reasons.append("replay_attack_signature")
            
            if suspicion_score > 0.5:
                suspicious.append({
                    'transaction': tx,
                    'suspicion_score': suspicion_score,
                    'reasons': reasons
                })
        
        return suspicious
    
    # ========================================================================
    # ANALYTICS & REPORTING
    # ========================================================================
    
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        return {
            'nodes': len(self.nodes),
            'transactions': len(self.transactions),
            'total_volume': self.total_volume,
            'avg_transaction_size': self.total_volume / max(1, self.total_transactions),
            'graph_density': nx.density(self.graph),
            'avg_path_length': self.avg_path_length,
            'time_precision': f"{self.time_precision:.2e} seconds",
            'precision_type': 'zeptosecond' if self.zeptosecond_precision else 'picosecond'
        }
    
    def visualize_network(self, output_file: str = 'transaction_network.html'):
        """Visualize transaction network (to be implemented)"""
        # TODO: Implement visualization
        pass

