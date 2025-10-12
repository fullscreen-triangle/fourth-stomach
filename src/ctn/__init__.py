"""
Circulation Transaction Network (CTN)
=====================================

A revolutionary payment system operating at trans-Planckian precision.

Key components:
- CirculationTransactionNetwork: Main transaction processing engine
- ShadowTransactionNetwork: Market intelligence through pattern analysis
- GraphCompletionFinance: Topology-based lending (no traditional credit!)
- Transaction: Individual money flow unit
- VirtualTransaction: Shadow connection between pattern-correlated nodes
- DirectedLoan: Graph completion financing
- Node: Network participant (person, business, bank)
- TransactionStatus: Transaction lifecycle states
- TransactionPattern: Frequency signature of transaction behavior
"""

from .transaction_graph import (
    CirculationTransactionNetwork,
    Transaction,
    Node,
    TransactionStatus
)

from .shadow_network import (
    ShadowTransactionNetwork,
    VirtualTransaction,
    TransactionPattern
)

from .graph_completion_finance import (
    GraphCompletionFinance,
    DirectedLoan,
    LoanStatus,
    integrate_gcf_with_ctn
)

__version__ = "0.2.0"
__all__ = [
    "CirculationTransactionNetwork",
    "ShadowTransactionNetwork",
    "GraphCompletionFinance",
    "Transaction",
    "VirtualTransaction",
    "DirectedLoan",
    "TransactionPattern",
    "Node",
    "TransactionStatus",
    "LoanStatus",
    "integrate_gcf_with_ctn"
]

