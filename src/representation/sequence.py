"""
Transaction-to-Sequence Transformation
Converts financial transactions into directional sequences for semantic analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum
import numpy as np


class Direction(Enum):
    """Cardinal directions representing transaction properties."""
    NORTH = "↑"  # Large amount
    SOUTH = "↓"  # Small amount
    EAST = "→"   # Frequent pattern
    WEST = "←"   # Rare pattern
    UP = "⊙"     # High profit
    DOWN = "⊗"   # Low profit


@dataclass
class Transaction:
    """Financial transaction."""
    from_entity: str
    to_entity: str
    amount: float
    timestamp: float
    profit: float = 0.0


class SequenceEncoder:
    """
    Encodes transactions as directional sequences.
    Based on: financial-representation.tex Section on Sequence Representation
    """
    
    def __init__(self, 
                 amount_threshold: float = 1000.0,
                 frequency_percentile: float = 0.5,
                 profit_threshold: float = 0.0):
        """
        Initialize encoder with thresholds.
        
        Args:
            amount_threshold: Separates large/small amounts
            frequency_percentile: Separates frequent/rare (0-1)
            profit_threshold: Separates high/low profit
        """
        self.amount_threshold = amount_threshold
        self.frequency_percentile = frequency_percentile
        self.profit_threshold = profit_threshold
        self.pattern_frequencies = {}
    
    def encode_transaction(self, txn: Transaction, frequency_rank: float = 0.5) -> List[Direction]:
        """
        Encode single transaction into 3 directions (amount, frequency, profit).
        
        Args:
            txn: Transaction to encode
            frequency_rank: Normalized frequency rank [0,1]
        
        Returns:
            List of 3 directions encoding the transaction
        """
        directions = []
        
        # Amount dimension
        if txn.amount >= self.amount_threshold:
            directions.append(Direction.NORTH)
        else:
            directions.append(Direction.SOUTH)
        
        # Frequency dimension
        if frequency_rank >= self.frequency_percentile:
            directions.append(Direction.EAST)
        else:
            directions.append(Direction.WEST)
        
        # Profit dimension
        if txn.profit >= self.profit_threshold:
            directions.append(Direction.UP)
        else:
            directions.append(Direction.DOWN)
        
        return directions
    
    def encode_stream(self, transactions: List[Transaction]) -> List[List[Direction]]:
        """
        Encode transaction stream into sequence.
        
        Args:
            transactions: List of transactions
        
        Returns:
            Sequence of directional encodings
        """
        # Calculate pattern frequencies
        patterns = [f"{t.from_entity}->{t.to_entity}" for t in transactions]
        unique_patterns = list(set(patterns))
        pattern_counts = {p: patterns.count(p) for p in unique_patterns}
        max_count = max(pattern_counts.values()) if pattern_counts else 1
        
        # Encode each transaction
        sequence = []
        for txn in transactions:
            pattern = f"{txn.from_entity}->{txn.to_entity}"
            frequency_rank = pattern_counts.get(pattern, 0) / max_count
            directions = self.encode_transaction(txn, frequency_rank)
            sequence.append(directions)
        
        return sequence
    
    def to_string(self, sequence: List[List[Direction]]) -> str:
        """Convert sequence to readable string."""
        return " ".join([
            "".join([d.value for d in dirs]) 
            for dirs in sequence
        ])
    
    def sequence_to_vector(self, directions: List[Direction]) -> np.ndarray:
        """
        Convert direction sequence to numeric vector.
        Each direction maps to a 6D unit vector.
        """
        direction_vectors = {
            Direction.NORTH: np.array([1, 0, 0, 0, 0, 0]),
            Direction.SOUTH: np.array([0, 1, 0, 0, 0, 0]),
            Direction.EAST: np.array([0, 0, 1, 0, 0, 0]),
            Direction.WEST: np.array([0, 0, 0, 1, 0, 0]),
            Direction.UP: np.array([0, 0, 0, 0, 1, 0]),
            Direction.DOWN: np.array([0, 0, 0, 0, 0, 1]),
        }
        
        # Concatenate direction vectors
        vectors = [direction_vectors[d] for d in directions]
        return np.concatenate(vectors)
