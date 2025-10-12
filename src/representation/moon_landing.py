"""
Chess with Miracles - Stochastic Navigation
Meta-information guided financial decision making.
"""

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class FinancialPosition:
    """Current state of financial network as 'chess position'."""
    active_nodes: List[str]
    active_edges: List[Tuple[str, str]]
    total_capital: float
    semantic_coordinates: np.ndarray  # Position in S-space (3D)


@dataclass
class Intervention:
    """Possible financial intervention ('move')."""
    action_type: str  # e.g., "loan", "investment", "trade"
    target_nodes: List[str]
    amount: float
    expected_return: float
    description: str


@dataclass
class SValue:
    """S-entropy coordinates for intervention."""
    s_temporal: float  # Time dimension
    s_information: float  # Information dimension
    s_entropy: float  # Entropy dimension
    
    def is_miraculous(self) -> bool:
        """Check if any dimension exceeds 1.0 (miracle)."""
        return any(s > 1.0 for s in [self.s_temporal, self.s_information, self.s_entropy])
    
    def viability_score(self) -> float:
        """Calculate overall viability."""
        return (self.s_temporal + self.s_information + self.s_entropy) / 3.0


class ChessWithMiracles:
    """
    Financial navigation using meta-information from unplayed moves.
    Based on: financial-representation.tex Chess with Miracles section
    """
    
    def __init__(self, viability_threshold: float = 0.8):
        """
        Initialize navigator.
        
        Args:
            viability_threshold: Minimum viability to accept intervention
        """
        self.viability_threshold = viability_threshold
        self.move_history: List[Tuple[Intervention, SValue]] = []
    
    def estimate_time_cost(self, intervention: Intervention) -> float:
        """
        Estimate time cost of intervention.
        
        Returns:
            s_temporal value (higher = faster)
        """
        # Simple model: inversely related to number of nodes involved
        base_time = len(intervention.target_nodes) * 2.0  # hours
        
        # Amount affects complexity
        if intervention.amount > 100000:
            base_time *= 1.5
        
        # s_temporal = 1 / time_cost
        s_temporal = 10.0 / (base_time + 1.0)
        
        return s_temporal
    
    def estimate_information_gain(self, intervention: Intervention, 
                                  current_position: FinancialPosition) -> float:
        """
        Estimate information gain from intervention.
        
        Returns:
            s_information value
        """
        # Information gain from exploring new nodes
        new_nodes = set(intervention.target_nodes) - set(current_position.active_nodes)
        information_gain = len(new_nodes) * 0.3
        
        # Expected return provides information
        if intervention.expected_return > 0:
            information_gain += np.log1p(intervention.expected_return)
        
        # Normalize
        s_information = np.clip(information_gain, 0.0, 3.0)
        
        return s_information
    
    def estimate_entropy_impact(self, intervention: Intervention,
                                current_position: FinancialPosition) -> float:
        """
        Estimate impact on system entropy.
        
        Returns:
            s_entropy value
        """
        # Entropy reduction from creating new connections
        potential_new_edges = len(intervention.target_nodes) * (len(current_position.active_nodes) - 1)
        current_edges = len(current_position.active_edges)
        
        # Relative entropy change
        entropy_impact = potential_new_edges / (current_edges + 1.0)
        
        # Normalize
        s_entropy = np.clip(entropy_impact, 0.0, 2.0)
        
        return s_entropy
    
    def evaluate_intervention(self, 
                            intervention: Intervention,
                            current_position: FinancialPosition) -> SValue:
        """
        Calculate S-values for intervention.
        
        Args:
            intervention: Proposed intervention
            current_position: Current system state
        
        Returns:
            S-value coordinates
        """
        s_t = self.estimate_time_cost(intervention)
        s_i = self.estimate_information_gain(intervention, current_position)
        s_e = self.estimate_entropy_impact(intervention, current_position)
        
        return SValue(s_temporal=s_t, s_information=s_i, s_entropy=s_e)
    
    def extract_meta_information(self, 
                                evaluated_interventions: List[Tuple[Intervention, SValue]]) -> Dict[str, Any]:
        """
        Extract comparative meta-information from all potential moves.
        
        Args:
            evaluated_interventions: List of (intervention, s_value) pairs
        
        Returns:
            Meta-information dictionary
        """
        if not evaluated_interventions:
            return {}
        
        s_values_array = np.array([
            [s.s_temporal, s.s_information, s.s_entropy]
            for _, s in evaluated_interventions
        ])
        
        meta_info = {
            'mean_s': np.mean(s_values_array, axis=0),
            'std_s': np.std(s_values_array, axis=0),
            'max_s': np.max(s_values_array, axis=0),
            'min_s': np.min(s_values_array, axis=0),
            'miracle_count': sum(1 for _, s in evaluated_interventions if s.is_miraculous()),
            'viable_count': sum(1 for _, s in evaluated_interventions 
                              if s.viability_score() > self.viability_threshold),
            'total_interventions': len(evaluated_interventions),
        }
        
        # Identify miracle dimensions
        meta_info['miracle_dimensions'] = []
        for dim, name in enumerate(['temporal', 'information', 'entropy']):
            if meta_info['max_s'][dim] > 1.0:
                meta_info['miracle_dimensions'].append(name)
        
        return meta_info
    
    def select_intervention(self,
                          interventions: List[Intervention],
                          current_position: FinancialPosition,
                          strategy: str = 'viability') -> Tuple[Intervention, SValue, Dict]:
        """
        Chess with Miracles algorithm: Select optimal intervention.
        
        Args:
            interventions: List of potential interventions
            current_position: Current system state
            strategy: Selection strategy ('viability', 'miracle', 'balanced')
        
        Returns:
            (selected_intervention, s_value, meta_information)
        """
        # Evaluate all interventions
        evaluated = [
            (intervention, self.evaluate_intervention(intervention, current_position))
            for intervention in interventions
        ]
        
        # Extract meta-information from alternatives
        meta_info = self.extract_meta_information(evaluated)
        
        # Select based on strategy
        if strategy == 'viability':
            # Choose most viable
            best_idx = np.argmax([s.viability_score() for _, s in evaluated])
        
        elif strategy == 'miracle':
            # Prefer miraculous moves
            miraculous = [(i, (interv, s)) for i, (interv, s) in enumerate(evaluated) if s.is_miraculous()]
            if miraculous:
                # Among miraculous, choose highest viability
                best_idx = max(miraculous, key=lambda x: x[1][1].viability_score())[0]
            else:
                # Fallback to viability
                best_idx = np.argmax([s.viability_score() for _, s in evaluated])
        
        elif strategy == 'balanced':
            # Balance viability and miracle potential
            scores = [
                s.viability_score() + (0.5 if s.is_miraculous() else 0.0)
                for _, s in evaluated
            ]
            best_idx = np.argmax(scores)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        selected_intervention, selected_s = evaluated[best_idx]
        
        # Record in history
        self.move_history.append((selected_intervention, selected_s))
        
        return selected_intervention, selected_s, meta_info
    
    def stochastic_jump(self,
                       interventions: List[Intervention],
                       current_position: FinancialPosition,
                       temperature: float = 1.0) -> Tuple[Intervention, SValue]:
        """
        Stochastic selection enabling exploration.
        
        Args:
            interventions: Available interventions
            current_position: Current state
            temperature: Controls randomness (higher = more random)
        
        Returns:
            (selected_intervention, s_value)
        """
        # Evaluate all
        evaluated = [
            (intervention, self.evaluate_intervention(intervention, current_position))
            for intervention in interventions
        ]
        
        # Softmax probabilities based on viability
        viabilities = np.array([s.viability_score() for _, s in evaluated])
        exp_viabilities = np.exp(viabilities / temperature)
        probabilities = exp_viabilities / np.sum(exp_viabilities)
        
        # Sample
        selected_idx = np.random.choice(len(evaluated), p=probabilities)
        
        return evaluated[selected_idx]
    
    def navigate_trajectory(self,
                           intervention_generator,
                           initial_position: FinancialPosition,
                           max_steps: int = 10,
                           strategy: str = 'balanced') -> List[Tuple[Intervention, SValue]]:
        """
        Navigate through financial state space using Chess with Miracles.
        
        Args:
            intervention_generator: Function generating interventions at each step
            initial_position: Starting position
            max_steps: Maximum navigation steps
            strategy: Selection strategy
        
        Returns:
            Trajectory of (intervention, s_value) pairs
        """
        trajectory = []
        current_position = initial_position
        
        for step in range(max_steps):
            # Generate possible interventions
            interventions = intervention_generator(current_position)
            
            if not interventions:
                break
            
            # Select intervention
            selected, s_value, meta_info = self.select_intervention(
                interventions, current_position, strategy
            )
            
            trajectory.append((selected, s_value))
            
            # Update position (simplified - would need full simulation)
            # For now, just add target nodes
            current_position.active_nodes.extend(selected.target_nodes)
            current_position.active_nodes = list(set(current_position.active_nodes))
            
            # Check if reached viable state
            if s_value.viability_score() > 0.95:
                break
        
        return trajectory
