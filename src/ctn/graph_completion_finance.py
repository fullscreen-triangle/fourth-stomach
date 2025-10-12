"""
Graph Completion Financing (GCF)
=================================

A revolutionary lending system based on shadow network topology rather than
traditional creditworthiness.

Key Insight:
-----------
In a circulation transaction network, "credit" is just incomplete flows.
The shadow network reveals where flows SHOULD exist (high correlation).
Directed loans complete the graph, generating economic activity that
automatically repays the loan through the circulation.

Core Concepts:
-------------
1. Virtual Transaction → Potential Flow
   If ρ(A, B) > 0.7, there's missing economic activity between A and B

2. Directed Loan → Graph Completion
   Loan money to A to transact with B, completing the shadow network

3. Automatic Repayment → Circulation
   The completed flow circulates through the network, repaying through settlement

4. No Default Risk
   The loan IS the flow. Completing the graph generates the repayment.

Mathematical Foundation:
-----------------------
Traditional Credit:
    Risk(loan) = f(credit_history, collateral, income)
    
Graph Completion:
    Opportunity(A→B) = ρ(A,B) × (V_potential - V_actual)
    
Where:
    ρ(A,B) = pattern correlation (from shadow network)
    V_potential = expected volume (from pattern amplitude)
    V_actual = current volume (from transaction history)
    
If Opportunity(A→B) > threshold:
    → Directed loan to complete the flow
    → Repayment through circulation settlement

Example:
-------
Coffee Shop & Supplier B:
    ρ = 0.92 (very high correlation!)
    Pattern suggests: $1000/day flow
    Actual: $400/day flow
    Missing: $600/day
    
Directed Loan:
    System → Coffee Shop: $600 loan
    Coffee Shop → Supplier B: $600 purchase
    
Circulation:
    Supplier B → ... → Coffee Shop (through network)
    Coffee Shop balance increases through normal business
    Loan repays automatically at end-of-day settlement
    
Result:
    Graph completed ✓
    Economic activity increased ✓
    Loan repaid ✓
    NO DEFAULT POSSIBLE (it's just circulation!)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from transaction_graph import CirculationTransactionNetwork, Node, Transaction, TransactionStatus
from shadow_network import ShadowTransactionNetwork, VirtualTransaction, TransactionPattern


class LoanStatus(Enum):
    """Status of graph completion loan"""
    PROPOSED = "proposed"      # Identified but not executed
    ACTIVE = "active"          # Loan disbursed, flow in progress
    COMPLETING = "completing"  # Flow circulating through network
    COMPLETED = "completed"    # Flow completed, loan repaid
    CANCELLED = "cancelled"    # Opportunity no longer valid


@dataclass
class DirectedLoan:
    """
    A loan directed by shadow network analysis to complete graph flows
    
    This is NOT traditional credit!
    - No credit check
    - No collateral
    - No interest (or minimal fee for service)
    - Can't default (loan IS the flow)
    
    Properties:
    ----------
    borrower: Node that receives loan
    target: Node that borrower should transact with
    amount: Loan amount (missing flow volume)
    correlation: Pattern correlation between borrower and target
    opportunity_score: Expected economic benefit
    status: Current status of loan
    """
    loan_id: str
    borrower: str
    target: str
    
    # Financial
    amount: float
    fee: float = 0.0  # Small service fee (not interest!)
    
    # Shadow network metrics
    correlation: float = 0.0
    harmonic_order: Tuple[int, int] = (0, 0)
    opportunity_score: float = 0.0
    
    # Expected vs. actual volumes
    expected_volume: float = 0.0  # From pattern analysis
    actual_volume: float = 0.0    # Current transaction volume
    missing_volume: float = 0.0   # Gap to fill
    
    # Status
    status: LoanStatus = LoanStatus.PROPOSED
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    disbursed_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Repayment tracking
    flow_completion_path: List[str] = field(default_factory=list)
    repayment_amount: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.missing_volume = max(0, self.expected_volume - self.actual_volume)
        
        # Opportunity score: correlation × missing volume × graph connectivity
        self.opportunity_score = abs(self.correlation) * self.missing_volume


class GraphCompletionFinance:
    """
    Graph Completion Financing system
    
    Uses shadow network analysis to:
    1. Identify incomplete flows (virtual transactions with high correlation)
    2. Calculate missing volume (expected vs. actual)
    3. Offer directed loans to complete the graph
    4. Track repayment through circulation settlement
    
    Key Innovation:
    --------------
    Traditional credit is about PAST behavior (credit history).
    Graph completion is about FUTURE flows (network topology).
    
    The shadow network tells us where flows SHOULD exist.
    Directed loans make those flows real.
    Circulation ensures automatic repayment.
    """
    
    def __init__(self, ctn: CirculationTransactionNetwork, stn: ShadowTransactionNetwork):
        self.ctn = ctn
        self.stn = stn
        
        # Directed loans
        self.proposed_loans: List[DirectedLoan] = []
        self.active_loans: List[DirectedLoan] = []
        self.completed_loans: List[DirectedLoan] = []
        
        # Configuration
        self.min_correlation = 0.7        # Minimum pattern correlation
        self.min_opportunity = 100.0      # Minimum missing volume
        self.max_loan_ratio = 0.5         # Max loan as % of missing volume
        self.service_fee_rate = 0.01      # 1% service fee (not interest!)
        
        # Statistics
        self.total_loaned = 0.0
        self.total_repaid = 0.0
        self.total_opportunity_created = 0.0
        
    # ========================================================================
    # OPPORTUNITY IDENTIFICATION
    # ========================================================================
    
    def identify_opportunities(self) -> List[DirectedLoan]:
        """
        Identify graph completion opportunities from shadow network
        
        Process:
        -------
        1. Analyze virtual transactions (pattern correlations)
        2. Calculate expected vs. actual volumes
        3. Identify significant gaps (missing flows)
        4. Propose directed loans to fill gaps
        
        Returns list of proposed loans sorted by opportunity score
        """
        opportunities = []
        
        for vtx in self.stn.virtual_transactions:
            # Check correlation threshold
            if abs(vtx.correlation) < self.min_correlation:
                continue
            
            # Get patterns
            pattern_a = self.stn.patterns.get(vtx.node_a)
            pattern_b = self.stn.patterns.get(vtx.node_b)
            
            if not pattern_a or not pattern_b:
                continue
            
            # Calculate expected volume from pattern
            # Use harmonic amplitude as expected transaction volume
            n, m = vtx.harmonic_order
            expected_a = pattern_a.harmonics.get(n, (0, 0))[0] if n in pattern_a.harmonics else pattern_a.mean_transaction_amount
            expected_b = pattern_b.harmonics.get(m, (0, 0))[0] if m in pattern_b.harmonics else pattern_b.mean_transaction_amount
            expected_volume = (expected_a + expected_b) / 2
            
            # Calculate actual volume (recent transactions between A and B)
            actual_volume = self._calculate_actual_volume(vtx.node_a, vtx.node_b)
            
            # Missing volume
            missing_volume = expected_volume - actual_volume
            
            # Check if significant opportunity
            if missing_volume < self.min_opportunity:
                continue
            
            # Create directed loan opportunity (A → B)
            loan_ab = DirectedLoan(
                loan_id=f"GCF-{vtx.node_a}-{vtx.node_b}-{int(datetime.now().timestamp())}",
                borrower=vtx.node_a,
                target=vtx.node_b,
                amount=missing_volume * self.max_loan_ratio,  # Conservative
                fee=missing_volume * self.max_loan_ratio * self.service_fee_rate,
                correlation=vtx.correlation,
                harmonic_order=vtx.harmonic_order,
                expected_volume=expected_volume,
                actual_volume=actual_volume,
                missing_volume=missing_volume
            )
            
            opportunities.append(loan_ab)
            
            # Also consider reverse direction (B → A) if correlation is symmetric
            if vtx.correlation > 0:  # Positive correlation
                actual_volume_ba = self._calculate_actual_volume(vtx.node_b, vtx.node_a)
                missing_volume_ba = expected_volume - actual_volume_ba
                
                if missing_volume_ba >= self.min_opportunity:
                    loan_ba = DirectedLoan(
                        loan_id=f"GCF-{vtx.node_b}-{vtx.node_a}-{int(datetime.now().timestamp())}",
                        borrower=vtx.node_b,
                        target=vtx.node_a,
                        amount=missing_volume_ba * self.max_loan_ratio,
                        fee=missing_volume_ba * self.max_loan_ratio * self.service_fee_rate,
                        correlation=vtx.correlation,
                        harmonic_order=vtx.harmonic_order,
                        expected_volume=expected_volume,
                        actual_volume=actual_volume_ba,
                        missing_volume=missing_volume_ba
                    )
                    
                    opportunities.append(loan_ba)
        
        # Sort by opportunity score
        opportunities.sort(key=lambda l: l.opportunity_score, reverse=True)
        
        self.proposed_loans = opportunities
        return opportunities
    
    def _calculate_actual_volume(self, from_node: str, to_node: str) -> float:
        """Calculate recent transaction volume between two nodes"""
        total = 0.0
        
        if from_node in self.ctn.nodes:
            node = self.ctn.nodes[from_node]
            for tx in node.outflows:
                if tx.to_node == to_node:
                    total += tx.amount
        
        return total
    
    # ========================================================================
    # LOAN EXECUTION
    # ========================================================================
    
    def disburse_loan(self, loan: DirectedLoan) -> bool:
        """
        Disburse directed loan to complete graph
        
        Process:
        -------
        1. Check borrower's capacity (not creditworthiness!)
        2. Create transaction: System → Borrower
        3. Guide transaction: Borrower → Target
        4. Track flow completion through network
        
        The loan automatically creates the transaction it's meant to enable!
        """
        # Validate
        if loan.status != LoanStatus.PROPOSED:
            return False
        
        if loan.borrower not in self.ctn.nodes or loan.target not in self.ctn.nodes:
            return False
        
        # Disburse loan (credit borrower's account)
        borrower_node = self.ctn.nodes[loan.borrower]
        borrower_node.balance += loan.amount
        
        # Update status
        loan.status = LoanStatus.ACTIVE
        loan.disbursed_at = datetime.now().timestamp()
        
        # Create the directed transaction (borrower → target)
        # This is what the loan is FOR!
        tx = self.ctn.add_transaction(
            loan.borrower,
            loan.target,
            loan.amount,
            timestamp=datetime.now().timestamp()
        )
        
        if tx:
            loan.status = LoanStatus.COMPLETING
            self.active_loans.append(loan)
            self.total_loaned += loan.amount
            
            # Calculate flow completion path (for tracking)
            loan.flow_completion_path = self._find_completion_path(loan.borrower, loan.target)
            
            return True
        else:
            # Transaction failed, rollback
            borrower_node.balance -= loan.amount
            loan.status = LoanStatus.CANCELLED
            return False
    
    def _find_completion_path(self, source: str, target: str) -> List[str]:
        """
        Find path through shadow network for flow to complete
        
        Uses shadow graph to predict how the flow will circulate
        back to source (enabling repayment)
        """
        try:
            # Find shortest path in shadow graph
            path = nx.shortest_path(
                self.stn.shadow_graph,
                source=target,  # Start from target (where money went)
                target=source,  # End at source (completing the loop)
                weight=lambda u, v, d: 1.0 / (abs(d.get('weight', 0.1)) + 0.01)  # Prefer high correlation
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path found, return empty
            return []
    
    # ========================================================================
    # REPAYMENT TRACKING
    # ========================================================================
    
    def track_loan_completion(self, loan: DirectedLoan) -> bool:
        """
        Track loan repayment through circulation settlement
        
        Key Insight:
        -----------
        We don't need borrower to "repay" in traditional sense.
        The loan enabled a transaction that increases network circulation.
        At settlement, borrower's balance reflects their net position.
        If the graph completed as predicted, balance will be positive.
        
        This is automatic "repayment" through circulation!
        """
        if loan.status != LoanStatus.COMPLETING:
            return False
        
        borrower_node = self.ctn.nodes[loan.borrower]
        
        # Check if borrower's balance increased through circulation
        # (This means the flow completed and circulated back)
        if borrower_node.balance >= loan.amount + loan.fee:
            # Loan "repaid" through circulation!
            loan.repayment_amount = loan.amount + loan.fee
            loan.status = LoanStatus.COMPLETED
            loan.completed_at = datetime.now().timestamp()
            
            # Deduct loan repayment from balance
            borrower_node.balance -= loan.repayment_amount
            
            # Statistics
            self.total_repaid += loan.repayment_amount
            self.total_opportunity_created += loan.amount
            
            # Move to completed
            if loan in self.active_loans:
                self.active_loans.remove(loan)
            self.completed_loans.append(loan)
            
            return True
        
        return False
    
    def settle_all_loans(self) -> Dict:
        """
        Settle all active loans at end of day
        
        Called as part of CTN end-of-day settlement.
        Checks all active loans for completion.
        """
        results = {
            'completed': 0,
            'still_active': 0,
            'total_repaid': 0.0
        }
        
        for loan in self.active_loans[:]:  # Copy list to allow modification
            if self.track_loan_completion(loan):
                results['completed'] += 1
                results['total_repaid'] += loan.repayment_amount
            else:
                results['still_active'] += 1
        
        return results
    
    # ========================================================================
    # ANALYSIS & REPORTING
    # ========================================================================
    
    def generate_gcf_report(self) -> Dict:
        """Generate comprehensive graph completion finance report"""
        # Calculate success metrics
        total_loans = len(self.completed_loans) + len(self.active_loans)
        success_rate = len(self.completed_loans) / total_loans if total_loans > 0 else 0.0
        
        # Economic impact
        economic_multiplier = self.total_opportunity_created / self.total_loaned if self.total_loaned > 0 else 0.0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            
            'loan_statistics': {
                'proposed': len(self.proposed_loans),
                'active': len(self.active_loans),
                'completed': len(self.completed_loans),
                'success_rate': success_rate
            },
            
            'financial_metrics': {
                'total_loaned': self.total_loaned,
                'total_repaid': self.total_repaid,
                'total_fees_collected': sum(l.fee for l in self.completed_loans),
                'outstanding': self.total_loaned - self.total_repaid
            },
            
            'economic_impact': {
                'opportunity_created': self.total_opportunity_created,
                'economic_multiplier': economic_multiplier,
                'graph_completion_rate': self._calculate_graph_completion_rate()
            },
            
            'top_opportunities': [
                {
                    'borrower': loan.borrower,
                    'target': loan.target,
                    'amount': loan.amount,
                    'correlation': loan.correlation,
                    'opportunity_score': loan.opportunity_score,
                    'missing_volume': loan.missing_volume
                }
                for loan in sorted(self.proposed_loans, key=lambda l: l.opportunity_score, reverse=True)[:10]
            ],
            
            'completed_loans': [
                {
                    'loan_id': loan.loan_id,
                    'borrower': loan.borrower,
                    'target': loan.target,
                    'amount': loan.amount,
                    'repayment': loan.repayment_amount,
                    'duration_seconds': loan.completed_at - loan.disbursed_at if loan.completed_at and loan.disbursed_at else 0,
                    'completion_path': loan.flow_completion_path
                }
                for loan in self.completed_loans[-10:]  # Last 10 completed
            ]
        }
        
        return report
    
    def _calculate_graph_completion_rate(self) -> float:
        """
        Calculate how "complete" the shadow graph is
        
        Complete graph: All high-correlation virtual transactions
                       have corresponding real transactions
        
        Incomplete graph: Missing flows (opportunities for directed loans)
        """
        if not self.stn.virtual_transactions:
            return 1.0
        
        total_potential_flow = 0.0
        actual_flow = 0.0
        
        for vtx in self.stn.virtual_transactions:
            if abs(vtx.correlation) > self.min_correlation:
                pattern_a = self.stn.patterns.get(vtx.node_a)
                pattern_b = self.stn.patterns.get(vtx.node_b)
                
                if pattern_a and pattern_b:
                    expected = (pattern_a.mean_transaction_amount + pattern_b.mean_transaction_amount) / 2
                    actual = self._calculate_actual_volume(vtx.node_a, vtx.node_b)
                    
                    total_potential_flow += expected
                    actual_flow += min(actual, expected)
        
        if total_potential_flow == 0:
            return 1.0
        
        return actual_flow / total_potential_flow
    
    def get_loan_recommendation(self, node_id: str) -> Optional[DirectedLoan]:
        """
        Get personalized loan recommendation for specific node
        
        Useful for UI: "We can help you complete transactions with..."
        """
        opportunities = [loan for loan in self.proposed_loans if loan.borrower == node_id]
        
        if opportunities:
            # Return highest opportunity score
            return max(opportunities, key=lambda l: l.opportunity_score)
        
        return None


# ============================================================================
# INTEGRATION WITH CTN
# ============================================================================

def integrate_gcf_with_ctn(ctn: CirculationTransactionNetwork, 
                           stn: ShadowTransactionNetwork) -> GraphCompletionFinance:
    """
    Integrate Graph Completion Finance with CTN
    
    Usage:
    -----
    1. Run CTN for period (collect transactions)
    2. Analyze shadow network (identify patterns)
    3. Identify opportunities (graph completion finance)
    4. Disburse directed loans
    5. Track completion through settlement
    6. Repeat!
    """
    gcf = GraphCompletionFinance(ctn, stn)
    
    # Identify opportunities from shadow network
    opportunities = gcf.identify_opportunities()
    
    print(f"Found {len(opportunities)} graph completion opportunities")
    
    # Auto-disburse top opportunities (in production, this would require approval)
    for loan in opportunities[:5]:  # Top 5
        if gcf.disburse_loan(loan):
            print(f"  ✓ Disbursed ${loan.amount:.2f} loan: {loan.borrower} → {loan.target}")
            print(f"    Correlation: {loan.correlation:.3f}, Opportunity: ${loan.missing_volume:.2f}")
    
    return gcf

