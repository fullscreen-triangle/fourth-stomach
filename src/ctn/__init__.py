"""
Circulation Transaction Network (CTN)
=====================================

A revolutionary payment system operating at trans-Planckian precision.

Key components:
- CirculationTransactionNetwork: Main transaction processing engine
- Transaction: Individual money flow unit
- Node: Network participant (person, business, bank)
- TransactionStatus: Transaction lifecycle states
"""

from .transaction_graph import (
    CirculationTransactionNetwork,
    Transaction,
    Node,
    TransactionStatus
)

__version__ = "0.1.0"
__all__ = [
    "CirculationTransactionNetwork",
    "Transaction", 
    "Node",
    "TransactionStatus"
]

