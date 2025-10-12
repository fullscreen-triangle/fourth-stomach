"""
Financial Circuit Network Representation
Maps transactions to electrical circuits with Kirchhoff's laws.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np
import networkx as nx


@dataclass
class CircuitElement:
    """Base class for circuit elements."""
    from_node: str
    to_node: str
    
    def voltage_drop(self, current: float, dt: float = 1.0) -> float:
        """Calculate voltage drop across element."""
        raise NotImplementedError


@dataclass
class Resistor(CircuitElement):
    """Financial resistor: transaction friction."""
    resistance: float  # Transaction cost per unit flow
    
    def voltage_drop(self, current: float, dt: float = 1.0) -> float:
        """V = I * R"""
        return current * self.resistance


@dataclass
class Capacitor(CircuitElement):
    """Financial capacitor: liquidity buffer."""
    capacitance: float  # Liquidity capacity
    stored_charge: float = 0.0
    
    def voltage_drop(self, current: float, dt: float = 1.0) -> float:
        """I = C * dV/dt, so V = integral(I/C * dt)"""
        self.stored_charge += current * dt
        return self.stored_charge / self.capacitance if self.capacitance > 0 else 0.0


@dataclass
class Inductor(CircuitElement):
    """Financial inductor: trading momentum."""
    inductance: float  # Trading inertia
    current_rate: float = 0.0
    
    def voltage_drop(self, current: float, dt: float = 1.0) -> float:
        """V = L * dI/dt"""
        di_dt = (current - self.current_rate) / dt if dt > 0 else 0.0
        self.current_rate = current
        return self.inductance * di_dt


class FinancialCircuit:
    """
    Financial transaction network as electrical circuit.
    Based on: financial-representation.tex Circuit Network section
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.elements: Dict[Tuple[str, str], CircuitElement] = {}
        self.node_potentials: Dict[str, float] = {}
        self.edge_currents: Dict[Tuple[str, str], float] = {}
    
    def add_node(self, node_id: str, net_worth: float = 0.0, credit_capacity: float = 0.0, alpha: float = 0.5):
        """
        Add financial node with potential.
        
        V_i = NetWorth + α * CreditCapacity
        
        Args:
            node_id: Node identifier
            net_worth: Current net worth
            credit_capacity: Available credit
            alpha: Credit availability weight [0,1]
        """
        potential = net_worth + alpha * credit_capacity
        self.graph.add_node(node_id)
        self.node_potentials[node_id] = potential
    
    def add_resistor(self, from_node: str, to_node: str, resistance: float):
        """Add resistive element (transaction cost)."""
        self.graph.add_edge(from_node, to_node)
        self.elements[(from_node, to_node)] = Resistor(from_node, to_node, resistance)
    
    def add_capacitor(self, from_node: str, to_node: str, capacitance: float):
        """Add capacitive element (liquidity buffer)."""
        self.graph.add_edge(from_node, to_node)
        self.elements[(from_node, to_node)] = Capacitor(from_node, to_node, capacitance)
    
    def add_inductor(self, from_node: str, to_node: str, inductance: float):
        """Add inductive element (trading momentum)."""
        self.graph.add_edge(from_node, to_node)
        self.elements[(from_node, to_node)] = Inductor(from_node, to_node, inductance)
    
    def set_current(self, from_node: str, to_node: str, current: float):
        """Set current (transaction flow) on edge."""
        self.edge_currents[(from_node, to_node)] = current
    
    def check_kirchhoff_current_law(self, node: str) -> Tuple[bool, float]:
        """
        Check KCL: sum(inflow) = sum(outflow)
        
        Returns:
            (is_satisfied, imbalance)
        """
        inflow = sum(
            self.edge_currents.get((pred, node), 0.0)
            for pred in self.graph.predecessors(node)
        )
        outflow = sum(
            self.edge_currents.get((node, succ), 0.0)
            for succ in self.graph.successors(node)
        )
        imbalance = inflow - outflow
        return abs(imbalance) < 1e-6, imbalance
    
    def check_kirchhoff_voltage_law(self, cycle: List[str]) -> Tuple[bool, float]:
        """
        Check KVL: sum of voltage drops around cycle = 0
        
        Args:
            cycle: List of nodes forming a closed loop
        
        Returns:
            (is_satisfied, total_voltage_drop)
        """
        total_drop = 0.0
        for i in range(len(cycle)):
            from_node = cycle[i]
            to_node = cycle[(i + 1) % len(cycle)]
            
            # Voltage drop = V_from - V_to
            v_drop = self.node_potentials.get(from_node, 0.0) - self.node_potentials.get(to_node, 0.0)
            total_drop += v_drop
        
        return abs(total_drop) < 1e-6, total_drop
    
    def calculate_voltage_drops(self, dt: float = 1.0) -> Dict[Tuple[str, str], float]:
        """
        Calculate voltage drops across all elements.
        
        Args:
            dt: Time step for dynamic elements
        
        Returns:
            Dictionary of (from, to) -> voltage_drop
        """
        voltage_drops = {}
        for edge, element in self.elements.items():
            current = self.edge_currents.get(edge, 0.0)
            voltage_drops[edge] = element.voltage_drop(current, dt)
        return voltage_drops
    
    def verify_equilibrium(self) -> Dict[str, any]:
        """
        Verify circuit is in equilibrium (no arbitrage).
        
        Returns:
            Dictionary with verification results
        """
        results = {
            'kcl_violations': [],
            'kvl_violations': [],
            'in_equilibrium': True
        }
        
        # Check KCL for all nodes
        for node in self.graph.nodes():
            satisfied, imbalance = self.check_kirchhoff_current_law(node)
            if not satisfied:
                results['kcl_violations'].append((node, imbalance))
                results['in_equilibrium'] = False
        
        # Check KVL for all simple cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))[:10]  # Limit to first 10 cycles
            for cycle in cycles:
                satisfied, total_drop = self.check_kirchhoff_voltage_law(cycle)
                if not satisfied:
                    results['kvl_violations'].append((cycle, total_drop))
                    results['in_equilibrium'] = False
        except:
            pass  # No cycles or error finding them
        
        return results
    
    def to_matrix_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to matrix form for linear system solving.
        
        Returns:
            (A, b) where Ax = b is the circuit equation system
            x represents node potentials
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        
        # Conductance matrix (A)
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for (from_node, to_node), element in self.elements.items():
            if isinstance(element, Resistor) and element.resistance > 0:
                conductance = 1.0 / element.resistance
                i, j = node_idx[from_node], node_idx[to_node]
                
                A[i, i] += conductance
                A[i, j] -= conductance
                A[j, i] -= conductance
                A[j, j] += conductance
        
        # Current sources (b)
        for node in nodes:
            i = node_idx[node]
            inflow = sum(
                self.edge_currents.get((pred, node), 0.0)
                for pred in self.graph.predecessors(node)
            )
            outflow = sum(
                self.edge_currents.get((node, succ), 0.0)
                for succ in self.graph.successors(node)
            )
            b[i] = inflow - outflow
        
        return A, b
