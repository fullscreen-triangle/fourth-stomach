"""
Multi-Modal Financial Representation Module

Implements circuit, sequence, gas molecular, and shadow network representations
with transformations between them.
"""

from .circuit import FinancialCircuit, Resistor, Capacitor, Inductor
from .sequence import SequenceEncoder, Direction, Transaction
from .gas_molecules import FinancialGasSystem, Molecule
from .semantic import SemanticAmplifier, FinancialLanguageModel
from .shadow import ShadowNetwork, RepresentationTransformer
from .moon_landing import ChessWithMiracles, FinancialPosition, Intervention, SValue

__all__ = [
    # Circuit
    'FinancialCircuit',
    'Resistor',
    'Capacitor',
    'Inductor',
    
    # Sequence
    'SequenceEncoder',
    'Direction',
    'Transaction',
    
    # Gas
    'FinancialGasSystem',
    'Molecule',
    
    # Semantic
    'SemanticAmplifier',
    'FinancialLanguageModel',
    
    # Shadow
    'ShadowNetwork',
    'RepresentationTransformer',
    
    # Navigation
    'ChessWithMiracles',
    'FinancialPosition',
    'Intervention',
    'SValue',
]

