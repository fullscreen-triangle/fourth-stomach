"""
Demo: Multi-Modal Financial Representation
Demonstrates all representation modes and transformations.
"""

import numpy as np
from sequence import Transaction, SequenceEncoder, Direction
from circuit import FinancialCircuit
from gas_molecules import FinancialGasSystem
from semantic import SemanticAmplifier, FinancialLanguageModel
from shadow import ShadowNetwork, RepresentationTransformer
from moon_landing import ChessWithMiracles, FinancialPosition, Intervention


def demo_sequence_representation():
    """Demo: Transaction to Sequence encoding."""
    print("=" * 60)
    print("DEMO 1: Sequence Representation")
    print("=" * 60)
    
    # Create transactions
    transactions = [
        Transaction("Alice", "Bob", 1500.0, 1.0, profit=100.0),
        Transaction("Bob", "Charlie", 800.0, 2.0, profit=-50.0),
        Transaction("Charlie", "Alice", 2000.0, 3.0, profit=200.0),
        Transaction("Alice", "Bob", 1600.0, 4.0, profit=120.0),
    ]
    
    # Encode as sequence
    encoder = SequenceEncoder()
    sequence = encoder.encode_stream(transactions)
    
    print(f"\nEncoded {len(transactions)} transactions into directional sequence:")
    print(encoder.to_string(sequence))
    
    print("\nDetailed encoding:")
    for i, (txn, dirs) in enumerate(zip(transactions, sequence)):
        print(f"  {txn.from_entity} → {txn.to_entity} (${txn.amount:.0f}): {[d.value for d in dirs]}")
    
    return sequence


def demo_circuit_representation():
    """Demo: Financial circuit with Kirchhoff's laws."""
    print("\n" + "=" * 60)
    print("DEMO 2: Circuit Representation")
    print("=" * 60)
    
    # Create circuit
    circuit = FinancialCircuit()
    
    # Add nodes with potentials
    circuit.add_node("Alice", net_worth=10000, credit_capacity=5000, alpha=0.5)
    circuit.add_node("Bob", net_worth=8000, credit_capacity=3000, alpha=0.5)
    circuit.add_node("Charlie", net_worth=12000, credit_capacity=4000, alpha=0.5)
    
    # Add circuit elements
    circuit.add_resistor("Alice", "Bob", resistance=0.05)  # 5% transaction cost
    circuit.add_capacitor("Bob", "Charlie", capacitance=1000.0)  # $1000 liquidity
    circuit.add_inductor("Charlie", "Alice", inductance=0.1)  # Trading inertia
    
    # Set currents
    circuit.set_current("Alice", "Bob", current=100.0)
    circuit.set_current("Bob", "Charlie", current=100.0)
    circuit.set_current("Charlie", "Alice", current=100.0)
    
    print("\nCircuit nodes and potentials:")
    for node, potential in circuit.node_potentials.items():
        print(f"  {node}: V = ${potential:.2f}")
    
    print("\nCircuit elements:")
    for edge, element in circuit.elements.items():
        print(f"  {edge[0]} → {edge[1]}: {element.__class__.__name__}")
    
    # Check equilibrium
    results = circuit.verify_equilibrium()
    print(f"\nCircuit in equilibrium: {results['in_equilibrium']}")
    print(f"  KCL violations: {len(results['kcl_violations'])}")
    print(f"  KVL violations: {len(results['kvl_violations'])}")
    
    return circuit


def demo_gas_representation():
    """Demo: Gas molecular system with harmonic interference."""
    print("\n" + "=" * 60)
    print("DEMO 3: Gas Molecular Representation")
    print("=" * 60)
    
    # Create gas system
    gas_system = FinancialGasSystem(market_temperature=1.0)
    
    # Add molecules (financial entities)
    np.random.seed(42)
    entities = ["Alice", "Bob", "Charlie", "David", "Eve"]
    
    for i, entity in enumerate(entities):
        position = np.array([i * 1.0, 0.0, 0.0])
        velocity = np.array([0.1, 0.0, 0.0])
        amplitude = np.random.rand() + 0.5
        frequency = np.random.rand() * 2.0 + 1.0  # 1-3 Hz
        
        gas_system.add_molecule(entity, position, velocity, amplitude, frequency)
    
    print(f"\nCreated gas system with {len(gas_system.molecules)} molecules:")
    for mol_id, mol in gas_system.molecules.items():
        print(f"  {mol_id}: ω = {mol.frequency:.3f} Hz, A = {mol.wavefunction_amplitude:.3f}")
    
    # Detect harmonic coincidences
    print("\nHarmonic coincidences detected:")
    coincidence_found = False
    for i, mol_i in enumerate(entities):
        for mol_j in entities[i+1:]:
            found, n, m, strength = gas_system.detect_harmonic_coincidence(mol_i, mol_j)
            if found:
                coincidence_found = True
                print(f"  {mol_i} ↔ {mol_j}: {n}ω_i ≈ {m}ω_j (strength: {strength:.3f})")
    
    if not coincidence_found:
        print("  (None with current random frequencies)")
    
    # Build correlation network
    correlations = gas_system.build_correlation_network(epsilon_tol=0.1, correlation_threshold=0.5)
    print(f"\nCorrelation network: {len(correlations)} edges")
    
    return gas_system


def demo_semantic_amplification():
    """Demo: Semantic distance amplification."""
    print("\n" + "=" * 60)
    print("DEMO 4: Semantic Distance Amplification")
    print("=" * 60)
    
    amplifier = SemanticAmplifier()
    
    print(f"\nAmplification factors:")
    print(f"  Layer 1 (Directional): γ₁ = {amplifier.gamma_1:.1f}")
    print(f"  Layer 2 (Positional): γ₂ = {amplifier.gamma_2:.1f}")
    print(f"  Layer 3 (Frequency): γ₃ = {amplifier.gamma_3:.1f}")
    print(f"  Layer 4 (Compression): γ₄ = {amplifier.gamma_4:.1f}")
    print(f"  Total: Γ = {amplifier.total_gamma:.0f}")
    
    # Encode two similar patterns
    pattern1 = [Direction.NORTH, Direction.EAST, Direction.UP]
    pattern2 = [Direction.NORTH, Direction.EAST, Direction.DOWN]
    
    emb1 = amplifier.encode_pattern(pattern1, position=0, sequence_length=10, frequency_rank=0.8)
    emb2 = amplifier.encode_pattern(pattern2, position=1, sequence_length=10, frequency_rank=0.7)
    
    distance = amplifier.semantic_distance(emb1, emb2)
    
    print(f"\nPattern 1: {[d.value for d in pattern1]}")
    print(f"Pattern 2: {[d.value for d in pattern2]}")
    print(f"Semantic distance (amplified): {distance:.2f}")
    print(f"Embedding dimension: {len(emb1)}")


def demo_llm_learning():
    """Demo: LLM-style continuous learning."""
    print("\n" + "=" * 60)
    print("DEMO 5: Financial Language Model")
    print("=" * 60)
    
    # Create language model
    llm = FinancialLanguageModel(embedding_dim=32, learning_rate=0.01)
    
    # Generate training stream
    np.random.seed(42)
    directions = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST, Direction.UP, Direction.DOWN]
    
    stream = []
    for _ in range(50):
        sequence_length = np.random.randint(2, 5)
        sequence = [np.random.choice(directions) for _ in range(sequence_length)]
        stream.append(sequence)
    
    print(f"Training on stream of {len(stream)} sequences...")
    
    # Train
    llm.continuous_learn(stream, window_size=5)
    
    # Test prediction
    test_context = [Direction.NORTH, Direction.EAST]
    predicted, probs = llm.predict_next(test_context)
    
    print(f"\nPrediction test:")
    print(f"  Context: {[d.value for d in test_context]}")
    print(f"  Predicted next: {predicted.value}")
    print(f"  Probability: {probs[llm.dir_to_idx[predicted]]:.3f}")


def demo_shadow_network():
    """Demo: Shadow network and miracle circuits."""
    print("\n" + "=" * 60)
    print("DEMO 6: Shadow Network & Miracle Circuits")
    print("=" * 60)
    
    shadow = ShadowNetwork(s_threshold=1.0)
    
    # Test miracle detection
    test_cases = [
        ("Alice", "Bob", 0.85, (1.2, 0.8, 0.9)),  # Miracle in temporal
        ("Bob", "Charlie", 0.75, (0.9, 1.5, 0.7)),  # Miracle in information
        ("Charlie", "David", 0.60, (0.8, 0.7, 0.9)),  # No miracle
    ]
    
    print("\nMiracle circuit detection:")
    for from_node, to_node, corr, s_vals in test_cases:
        is_miracle = shadow.is_miracle_circuit(s_vals)
        shadow.add_shadow_edge(from_node, to_node, corr, s_vals)
        
        print(f"  {from_node} → {to_node}: S=({s_vals[0]:.1f}, {s_vals[1]:.1f}, {s_vals[2]:.1f})")
        print(f"    {'✓ MIRACLE' if is_miracle else '✗ Normal'}")
    
    print(f"\nShadow network edges: {len(shadow.miracle_edges)}")
    for edge, corr in shadow.miracle_edges.items():
        print(f"  {edge[0]} → {edge[1]}: ρ = {corr:.2f}")


def demo_transformation_cycle():
    """Demo: Full transformation cycle T→C→S→G→C'→T'."""
    print("\n" + "=" * 60)
    print("DEMO 7: Representational Equivalence")
    print("=" * 60)
    
    transformer = RepresentationTransformer()
    
    # Create transactions
    transactions = [
        Transaction("Alice", "Bob", 1500.0, 1.0, profit=100.0),
        Transaction("Bob", "Charlie", 800.0, 2.0, profit=-50.0),
        Transaction("Charlie", "Alice", 2000.0, 3.0, profit=200.0),
    ]
    
    print(f"\nOriginal transactions: {len(transactions)}")
    
    # Full cycle
    results = transformer.full_cycle_transform(transactions)
    
    print(f"\nTransformation cycle:")
    print(f"  T → S → G → C → S' complete")
    print(f"  Information preservation: {results['information_preservation']:.1%}")
    print(f"  Information loss: {results['information_loss']:.1%}")
    print(f"  Passes threshold (<5%): {'✓' if results['passes_threshold'] else '✗'}")
    
    print(f"\nIntermediate representations:")
    print(f"  Gas molecules: {len(results['gas_system'].molecules)}")
    print(f"  Circuit nodes: {len(results['circuit'].graph.nodes())}")
    print(f"  Circuit edges: {len(results['circuit'].graph.edges())}")


def demo_chess_with_miracles():
    """Demo: Chess with Miracles navigation."""
    print("\n" + "=" * 60)
    print("DEMO 8: Chess with Miracles Navigation")
    print("=" * 60)
    
    # Create navigator
    navigator = ChessWithMiracles(viability_threshold=0.8)
    
    # Current position
    position = FinancialPosition(
        active_nodes=["Alice", "Bob"],
        active_edges=[("Alice", "Bob")],
        total_capital=10000.0,
        semantic_coordinates=np.array([0.5, 0.5, 0.5])
    )
    
    # Possible interventions
    interventions = [
        Intervention("loan", ["Charlie"], 1000.0, 150.0, "Small loan to Charlie"),
        Intervention("investment", ["David", "Eve"], 5000.0, 800.0, "Investment in D&E partnership"),
        Intervention("trade", ["Bob"], 500.0, 50.0, "Small trade with Bob"),
    ]
    
    print(f"\nCurrent position: {len(position.active_nodes)} nodes, ${position.total_capital:.0f}")
    print(f"Available interventions: {len(interventions)}")
    
    # Evaluate all
    for intervention in interventions:
        s_value = navigator.evaluate_intervention(intervention, position)
        print(f"\n  {intervention.action_type.upper()}: {intervention.description}")
        print(f"    S-values: ({s_value.s_temporal:.2f}, {s_value.s_information:.2f}, {s_value.s_entropy:.2f})")
        print(f"    Viability: {s_value.viability_score():.2f}")
        print(f"    {'⚡ MIRACULOUS' if s_value.is_miraculous() else '  Normal'}")
    
    # Select best
    selected, s_value, meta_info = navigator.select_intervention(
        interventions, position, strategy='viability'
    )
    
    print(f"\n{'─' * 60}")
    print(f"SELECTED: {selected.action_type} - {selected.description}")
    print(f"  S-values: ({s_value.s_temporal:.2f}, {s_value.s_information:.2f}, {s_value.s_entropy:.2f})")
    print(f"  Viability: {s_value.viability_score():.2f}")
    
    print(f"\nMeta-information from alternatives:")
    print(f"  Miracle count: {meta_info['miracle_count']}/{meta_info['total_interventions']}")
    print(f"  Viable count: {meta_info['viable_count']}/{meta_info['total_interventions']}")
    if meta_info['miracle_dimensions']:
        print(f"  Miracle dimensions: {', '.join(meta_info['miracle_dimensions'])}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Multi-Modal Financial Representation" + " " * 12 + "║")
    print("║" + " " * 20 + "DEMONSTRATION SUITE" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        demo_sequence_representation()
        demo_circuit_representation()
        demo_gas_representation()
        demo_semantic_amplification()
        demo_llm_learning()
        demo_shadow_network()
        demo_transformation_cycle()
        demo_chess_with_miracles()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully! ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

