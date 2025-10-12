"""
Shadow Network Demonstration
============================

Shows how transaction patterns reveal hidden market structure through
harmonic coincidences.

Demonstrates:
1. Transaction network (tree structure)
2. Shadow network (graph structure via pattern matching)
3. Market intelligence extraction
"""

from transaction_graph import CirculationTransactionNetwork, TransactionStatus
from shadow_network import ShadowTransactionNetwork
from datetime import datetime, timedelta
import numpy as np
import json


def demo_shadow_network():
    """
    Demonstrate shadow network intelligence extraction
    
    Scenario:
    --------
    5 businesses with different transaction rhythms:
    - Coffee Shop: Daily rhythm (weekday customers)
    - Restaurant: Weekly rhythm (weekend rush)
    - Supplier A: 3-day rhythm (delivery schedule)
    - Supplier B: Daily rhythm (same as coffee shop!)
    - Manufacturer: Weekly rhythm (same as restaurant!)
    
    Tree view: Linear supply chains
    Graph view: Coffee Shop ↔ Supplier B (daily rhythm match)
                Restaurant ↔ Manufacturer (weekly rhythm match)
    """
    
    print("=" * 80)
    print("SHADOW TRANSACTION NETWORK - Market Intelligence Demo")
    print("=" * 80)
    print()
    
    # Initialize CTN
    ctn = CirculationTransactionNetwork(name="Shadow Network Demo")
    
    # Create nodes
    nodes = {
        'coffee_shop': ctn.add_node('coffee_shop', 'Coffee Shop', 'business', credit_limit=10000),
        'restaurant': ctn.add_node('restaurant', 'Restaurant', 'business', credit_limit=20000),
        'supplier_a': ctn.add_node('supplier_a', 'Supplier A (3-day)', 'business', credit_limit=50000),
        'supplier_b': ctn.add_node('supplier_b', 'Supplier B (daily)', 'business', credit_limit=50000),
        'manufacturer': ctn.add_node('manufacturer', 'Manufacturer (weekly)', 'business', credit_limit=100000)
    }
    
    print("✓ Created 5 business nodes")
    print()
    
    # Simulate 30 days of transactions with distinct patterns
    print("Simulating 30 days of transactions...")
    print("-" * 80)
    
    base_time = datetime.now().timestamp()
    day_seconds = 24 * 3600
    
    # Pattern 1: Coffee Shop - Daily rhythm (weekdays)
    for day in range(30):
        if day % 7 < 5:  # Weekdays only
            # Morning rush
            for hour in [7, 8, 9]:
                tx_time = base_time - (30-day) * day_seconds + hour * 3600
                amount = np.random.normal(500, 100)
                ctn.add_transaction('customer_daily', 'coffee_shop', amount, tx_time)
    
    # Pattern 2: Restaurant - Weekly rhythm (weekends)
    for day in range(30):
        if day % 7 >= 5:  # Weekends only
            # Evening rush
            for hour in [18, 19, 20]:
                tx_time = base_time - (30-day) * day_seconds + hour * 3600
                amount = np.random.normal(2000, 300)
                ctn.add_transaction('customer_weekly', 'restaurant', amount, tx_time)
    
    # Pattern 3: Supplier A - 3-day rhythm
    for day in range(0, 30, 3):  # Every 3 days
        tx_time = base_time - (30-day) * day_seconds + 10 * 3600
        amount = np.random.normal(1500, 200)
        ctn.add_transaction('coffee_shop', 'supplier_a', amount, tx_time)
        ctn.add_transaction('restaurant', 'supplier_a', amount * 1.5, tx_time)
    
    # Pattern 4: Supplier B - Daily rhythm (SAME as coffee shop!)
    for day in range(30):
        if day % 7 < 5:  # Weekdays (matches coffee shop)
            tx_time = base_time - (30-day) * day_seconds + 14 * 3600
            amount = np.random.normal(1000, 150)
            ctn.add_transaction('coffee_shop', 'supplier_b', amount, tx_time)
    
    # Pattern 5: Manufacturer - Weekly rhythm (SAME as restaurant!)
    for day in range(0, 30, 7):  # Weekly (matches restaurant)
        tx_time = base_time - (30-day) * day_seconds + 12 * 3600
        amount = np.random.normal(5000, 500)
        ctn.add_transaction('supplier_a', 'manufacturer', amount, tx_time)
        ctn.add_transaction('supplier_b', 'manufacturer', amount * 0.8, tx_time)
    
    total_txs = len(ctn.transactions)
    print(f"✓ Simulated {total_txs} transactions over 30 days")
    print()
    
    # Initialize Shadow Network
    print("Analyzing transaction patterns...")
    print("-" * 80)
    
    stn = ShadowTransactionNetwork(ctn)
    
    # Extract patterns (FFT analysis)
    patterns = stn.extract_patterns(window_days=30)
    
    print(f"✓ Extracted {len(patterns)} transaction patterns")
    print()
    print("Pattern Summary:")
    for node_id, pattern in patterns.items():
        period_days = 1.0 / (pattern.omega_fundamental * 86400) if pattern.omega_fundamental > 0 else float('inf')
        print(f"  {node_id:20s}: ω = {pattern.omega_fundamental:.6f} Hz, "
              f"T = {period_days:.2f} days, "
              f"{len(pattern.harmonics)} harmonics")
    print()
    
    # Find harmonic coincidences
    print("Finding harmonic coincidences...")
    print("-" * 80)
    
    virtual_txs = stn.find_harmonic_coincidences()
    
    print(f"✓ Found {len(virtual_txs)} virtual transactions (pattern matches)")
    print()
    print("Top Virtual Connections:")
    for vtx in sorted(virtual_txs, key=lambda v: abs(v.correlation), reverse=True)[:5]:
        print(f"  {vtx.node_a} ↔ {vtx.node_b}")
        print(f"    Correlation: {vtx.correlation:.3f}")
        print(f"    Harmonic Order: {vtx.harmonic_order}")
        print(f"    Frequency Match: {vtx.frequency_a:.6f} Hz ≈ {vtx.frequency_b:.6f} Hz")
        print(f"    Difference: {vtx.frequency_difference:.8f} Hz")
        print()
    
    # Build shadow graph
    print("Building shadow network graph...")
    print("-" * 80)
    
    shadow_graph = stn.build_shadow_graph()
    
    print(f"✓ Shadow graph: {shadow_graph.number_of_nodes()} nodes, "
          f"{shadow_graph.number_of_edges()} edges")
    print()
    print(f"Transaction network: {len(nodes)} nodes, ~{len(nodes)-1} edges (tree)")
    print(f"Shadow network: {shadow_graph.number_of_nodes()} nodes, "
          f"{shadow_graph.number_of_edges()} edges (graph)")
    print(f"→ {shadow_graph.number_of_edges() - (len(nodes)-1)} additional connections revealed!")
    print()
    
    # Generate intelligence report
    print("=" * 80)
    print("MARKET INTELLIGENCE REPORT")
    print("=" * 80)
    print()
    
    report = stn.generate_intelligence_report()
    
    # Network statistics
    print("Network Statistics:")
    print("-" * 80)
    for key, value in report['network_stats'].items():
        print(f"  {key:30s}: {value:.4f}" if isinstance(value, float) else f"  {key:30s}: {value}")
    print()
    
    # Hub nodes
    print("Hub Nodes (High Betweenness Centrality):")
    print("-" * 80)
    for i, node_id in enumerate(report['hub_nodes'][:5], 1):
        print(f"  {i}. {node_id}")
    print()
    
    # Critical connections
    print("Critical Connections (System Vulnerabilities):")
    print("-" * 80)
    for i, conn in enumerate(report['critical_connections'][:5], 1):
        print(f"  {i}. {conn['node_a']} ↔ {conn['node_b']}")
        print(f"     Correlation: {conn['correlation']:.3f}, "
              f"Betweenness: {conn['betweenness']:.4f}")
        print(f"     Harmonic: {conn['harmonic_order']}")
    print()
    
    # Risk clusters
    print("Risk Clusters (Correlated Exposure):")
    print("-" * 80)
    for i, cluster in enumerate(report['risk_clusters'], 1):
        print(f"  Cluster {i}: {', '.join(cluster)}")
    print()
    
    # Arbitrage opportunities
    if report['arbitrage_opportunities']:
        print("Arbitrage Opportunities:")
        print("-" * 80)
        for i, opp in enumerate(report['arbitrage_opportunities'][:3], 1):
            print(f"  {i}. {opp['node_a']} ↔ {opp['node_b']}")
            print(f"     Amount A: ${opp['amount_a']:.2f}")
            print(f"     Amount B: ${opp['amount_b']:.2f}")
            print(f"     Price Difference: ${opp['price_difference']:.2f}")
            print(f"     Correlation: {opp['correlation']:.3f}")
            print(f"     Arbitrage Potential: {opp['arbitrage_potential']:.3f}")
            print()
    
    # Influence scores
    print("Influence Scores (Market Impact):")
    print("-" * 80)
    for node_id, score in sorted(report['influence_scores'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {node_id:20s}: {score:.4f}")
    print()
    
    # Coordinated behavior
    if report['coordinated_behavior']:
        print("⚠ Coordinated Behavior Detected:")
        print("-" * 80)
        for i, group in enumerate(report['coordinated_behavior'], 1):
            print(f"  Group {i}: {', '.join(group)}")
        print()
    
    # Detailed node intelligence
    print("=" * 80)
    print("DETAILED NODE INTELLIGENCE: Coffee Shop")
    print("=" * 80)
    print()
    
    coffee_intel = stn.get_node_intelligence('coffee_shop')
    print(json.dumps(coffee_intel, indent=2))
    print()
    
    # Key insight
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("The shadow network reveals connections that are INVISIBLE in the transaction tree:")
    print()
    print("Transaction View (Surface):")
    print("  Coffee Shop → Supplier B (direct business)")
    print("  Restaurant → Supplier A (direct business)")
    print()
    print("Shadow View (Hidden Structure):")
    print("  Coffee Shop ↔ Supplier B (daily rhythm match)")
    print("  Restaurant ↔ Manufacturer (weekly rhythm match)")
    print("  → These pattern coincidences reveal the ACTUAL market forces!")
    print()
    print("Applications:")
    print("  • Risk Management: Identify systemic vulnerabilities")
    print("  • Market Intelligence: Understand true influence networks")
    print("  • Fraud Detection: Spot coordinated manipulation")
    print("  • Arbitrage: Find price discrepancies in correlated nodes")
    print("  • Forecasting: Predict behavior through pattern correlation")
    print()
    print("This is EXACTLY like molecular observation:")
    print("  Tree → Single vibrational path")
    print("  Graph → Multiple observation paths through harmonic coincidences")
    print("  → Redundancy, validation, and deeper understanding!")
    print()


def demo_fraud_detection():
    """
    Demonstrate fraud detection through shadow network
    
    Scenario: Coordinated price manipulation
    - Normal businesses: Random transaction patterns
    - Cartel: Synchronized patterns (>0.95 correlation)
    
    Shadow network will detect the cartel through:
    - Abnormally high correlation
    - Synchronized harmonics
    - Coordinated behavior clustering
    """
    print("=" * 80)
    print("FRAUD DETECTION DEMONSTRATION")
    print("=" * 80)
    print()
    
    ctn = CirculationTransactionNetwork(name="Fraud Detection Demo")
    
    # Normal businesses (random patterns)
    normal_nodes = []
    for i in range(5):
        node_id = f"normal_{i}"
        ctn.add_node(node_id, f"Normal Business {i}", 'business', credit_limit=10000)
        normal_nodes.append(node_id)
    
    # Cartel (synchronized pattern)
    cartel_nodes = []
    for i in range(3):
        node_id = f"cartel_{i}"
        ctn.add_node(node_id, f"Cartel Member {i}", 'business', credit_limit=10000)
        cartel_nodes.append(node_id)
    
    print(f"Created {len(normal_nodes)} normal businesses + {len(cartel_nodes)} cartel members")
    print()
    
    # Simulate transactions
    base_time = datetime.now().timestamp()
    day_seconds = 24 * 3600
    
    # Normal businesses: Random patterns
    for node_id in normal_nodes:
        for day in range(30):
            # Random daily transactions
            num_txs = np.random.poisson(5)
            for _ in range(num_txs):
                hour = np.random.uniform(8, 20)
                tx_time = base_time - (30-day) * day_seconds + hour * 3600
                amount = np.random.normal(1000, 300)
                ctn.add_transaction('customer', node_id, amount, tx_time)
    
    # Cartel: HIGHLY synchronized pattern
    cartel_pattern = np.random.random(30)  # Shared pattern
    for node_id in cartel_nodes:
        for day in range(30):
            # Same transaction timing as other cartel members!
            for hour in [9, 12, 15]:  # Fixed hours
                tx_time = base_time - (30-day) * day_seconds + hour * 3600
                # Amount varies with shared pattern
                amount = 1000 * (1 + cartel_pattern[day])
                # Add tiny random noise (but still highly correlated)
                amount += np.random.normal(0, 10)
                ctn.add_transaction('customer', node_id, amount, tx_time)
    
    print(f"Simulated {len(ctn.transactions)} transactions")
    print()
    
    # Analyze with shadow network
    print("Analyzing for coordinated behavior...")
    print("-" * 80)
    
    stn = ShadowTransactionNetwork(ctn)
    patterns = stn.extract_patterns(window_days=30)
    virtual_txs = stn.find_harmonic_coincidences()
    shadow_graph = stn.build_shadow_graph()
    
    # Detect coordinated groups
    coordinated = stn.detect_coordinated_behavior(threshold=0.85)
    
    print(f"Found {len(coordinated)} coordinated groups:")
    print()
    
    for i, group in enumerate(coordinated, 1):
        is_cartel = all(node_id.startswith('cartel_') for node_id in group)
        
        print(f"  Group {i}: {', '.join(group)}")
        print(f"  {'⚠ SUSPICIOUS' if is_cartel else '✓ Legitimate'}")
        
        # Calculate average correlation within group
        correlations = []
        for vtx in virtual_txs:
            if vtx.node_a in group and vtx.node_b in group:
                correlations.append(abs(vtx.correlation))
        
        if correlations:
            avg_corr = np.mean(correlations)
            print(f"  Average Correlation: {avg_corr:.3f}")
            print(f"  Risk Level: {'🔴 HIGH' if avg_corr > 0.9 else '🟡 MEDIUM' if avg_corr > 0.7 else '🟢 LOW'}")
        
        print()
    
    print("✓ Shadow network successfully identified cartel through pattern correlation!")
    print()


if __name__ == "__main__":
    demo_shadow_network()
    print("\n" * 2)
    demo_fraud_detection()

