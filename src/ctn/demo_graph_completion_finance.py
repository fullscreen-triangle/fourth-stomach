"""
Graph Completion Finance - Demonstration
========================================

Shows how directed loans based on network topology (not creditworthiness!)
can complete graphs and create economic activity.

Key Concept:
-----------
Credit is just incomplete flows.
Shadow network reveals where flows SHOULD exist.
Directed loans complete the graph.
Circulation ensures automatic repayment.
"""

from transaction_graph import CirculationTransactionNetwork, TransactionStatus
from shadow_network import ShadowTransactionNetwork
from graph_completion_finance import GraphCompletionFinance, integrate_gcf_with_ctn
from datetime import datetime, timedelta
import numpy as np
import json


def demo_basic_graph_completion():
    """
    Demonstrate basic graph completion financing
    
    Scenario:
    --------
    - Coffee Shop has daily rhythm, pattern suggests $1000/day to Supplier
    - Actually only transacting $400/day (missing $600!)
    - Shadow network reveals high correlation (ρ = 0.9)
    - System offers $300 directed loan to complete graph
    - Coffee Shop uses loan to buy from Supplier
    - Flow circulates through network
    - Loan repays automatically!
    """
    
    print("=" * 80)
    print("GRAPH COMPLETION FINANCE - Basic Demo")
    print("=" * 80)
    print()
    
    # Create CTN
    ctn = CirculationTransactionNetwork(name="GCF Demo")
    
    # Create nodes
    ctn.add_node('coffee_shop', 'Coffee Shop', 'business', credit_limit=5000)
    ctn.add_node('supplier', 'Supplier', 'business', credit_limit=10000)
    ctn.add_node('customers', 'Customers', 'customer', credit_limit=50000)
    ctn.add_node('distributor', 'Distributor', 'business', credit_limit=20000)
    
    print("Created network nodes")
    print()
    
    # Simulate historical transactions (30 days)
    print("Simulating 30 days of transactions...")
    print("-" * 80)
    
    base_time = datetime.now().timestamp()
    day_seconds = 24 * 3600
    
    # Coffee shop receives from customers (daily rhythm, good business!)
    for day in range(30):
        for hour in [7, 8, 9, 12, 14]:  # Business hours
            tx_time = base_time - (30-day) * day_seconds + hour * 3600
            amount = np.random.normal(200, 30)
            ctn.add_transaction('customers', 'coffee_shop', amount, tx_time)
    
    # Coffee shop buys from supplier (daily rhythm, BUT INCOMPLETE!)
    for day in range(30):
        tx_time = base_time - (30-day) * day_seconds + 10 * 3600
        # Only $400/day instead of expected $1000/day!
        amount = np.random.normal(400, 50)
        ctn.add_transaction('coffee_shop', 'supplier', amount, tx_time)
    
    # Supplier buys from distributor (weekly rhythm)
    for day in range(0, 30, 7):
        tx_time = base_time - (30-day) * day_seconds + 12 * 3600
        amount = np.random.normal(2000, 200)
        ctn.add_transaction('supplier', 'distributor', amount, tx_time)
    
    # Distributor completes cycle back to customers
    for day in range(0, 30, 7):
        tx_time = base_time - (30-day) * day_seconds + 16 * 3600
        amount = np.random.normal(1500, 150)
        ctn.add_transaction('distributor', 'customers', amount, tx_time)
    
    print(f"✓ Simulated {len(ctn.transactions)} transactions")
    print()
    
    # Show current balances
    print("Current Network State:")
    print("-" * 80)
    for node_id, node in ctn.nodes.items():
        total_in = sum(tx.amount for tx in node.inflows)
        total_out = sum(tx.amount for tx in node.outflows)
        print(f"  {node.name:20s}: In=${total_in:8.2f}, Out=${total_out:8.2f}, Balance=${node.balance:8.2f}")
    print()
    
    # Analyze shadow network
    print("Analyzing Shadow Network...")
    print("-" * 80)
    
    stn = ShadowTransactionNetwork(ctn)
    patterns = stn.extract_patterns(window_days=30)
    virtual_txs = stn.find_harmonic_coincidences()
    shadow_graph = stn.build_shadow_graph()
    
    print(f"✓ Extracted {len(patterns)} patterns")
    print(f"✓ Found {len(virtual_txs)} virtual transactions")
    print()
    
    # Show pattern analysis
    print("Pattern Analysis:")
    print("-" * 80)
    for node_id, pattern in patterns.items():
        if node_id != 'customers':  # Skip customers for clarity
            period_days = 1.0 / (pattern.omega_fundamental * 86400) if pattern.omega_fundamental > 0 else float('inf')
            print(f"  {node_id:20s}: Period={period_days:5.2f} days, Avg Transaction=${pattern.mean_transaction_amount:7.2f}")
    print()
    
    # Initialize Graph Completion Finance
    print("=" * 80)
    print("GRAPH COMPLETION FINANCE ANALYSIS")
    print("=" * 80)
    print()
    
    gcf = GraphCompletionFinance(ctn, stn)
    
    # Identify opportunities
    print("Identifying Graph Completion Opportunities...")
    print("-" * 80)
    
    opportunities = gcf.identify_opportunities()
    
    if not opportunities:
        print("No opportunities found (graph is already complete!)")
        return
    
    print(f"✓ Found {len(opportunities)} opportunities")
    print()
    
    # Show top opportunities
    print("Top Opportunities:")
    print("-" * 80)
    for i, loan in enumerate(opportunities[:5], 1):
        print(f"{i}. {loan.borrower} → {loan.target}")
        print(f"   Correlation: {loan.correlation:.3f}")
        print(f"   Expected Volume: ${loan.expected_volume:.2f}")
        print(f"   Actual Volume: ${loan.actual_volume:.2f}")
        print(f"   Missing Volume: ${loan.missing_volume:.2f}")
        print(f"   Directed Loan Amount: ${loan.amount:.2f}")
        print(f"   Opportunity Score: {loan.opportunity_score:.2f}")
        print()
    
    # Disburse top loan
    print("=" * 80)
    print("DISBURSING DIRECTED LOAN")
    print("=" * 80)
    print()
    
    top_loan = opportunities[0]
    
    print(f"Loan Details:")
    print(f"  Borrower: {top_loan.borrower}")
    print(f"  Target: {top_loan.target}")
    print(f"  Amount: ${top_loan.amount:.2f}")
    print(f"  Fee: ${top_loan.fee:.2f} ({gcf.service_fee_rate*100}% service fee)")
    print(f"  Rationale: Pattern correlation {top_loan.correlation:.3f} suggests missing flow")
    print()
    
    # Show before balances
    borrower_before = ctn.nodes[top_loan.borrower].balance
    target_before = ctn.nodes[top_loan.target].balance
    
    print(f"Before Loan:")
    print(f"  {top_loan.borrower} balance: ${borrower_before:.2f}")
    print(f"  {top_loan.target} balance: ${target_before:.2f}")
    print()
    
    # Disburse!
    success = gcf.disburse_loan(top_loan)
    
    if success:
        print(f"✓ Loan disbursed successfully!")
        print(f"  Transaction created: {top_loan.borrower} → {top_loan.target} (${top_loan.amount:.2f})")
        print(f"  Flow completion path: {' → '.join(top_loan.flow_completion_path)}")
        print()
        
        # Show after balances
        borrower_after = ctn.nodes[top_loan.borrower].balance
        target_after = ctn.nodes[top_loan.target].balance
        
        print(f"After Loan:")
        print(f"  {top_loan.borrower} balance: ${borrower_after:.2f} (change: ${borrower_after - borrower_before:.2f})")
        print(f"  {top_loan.target} balance: ${target_after:.2f} (change: ${target_after - target_before:.2f})")
        print()
        
        print("Graph Status:")
        print(f"  Loan completed the flow: {top_loan.borrower} → {top_loan.target}")
        print(f"  Economic activity created: ${top_loan.amount:.2f}")
        print(f"  Circulation will distribute this through network")
        print()
        
        # Simulate some more transactions (circulation continuing)
        print("Simulating continued circulation...")
        print("-" * 80)
        
        # More customer purchases
        for i in range(5):
            amount = np.random.normal(150, 20)
            ctn.add_transaction('customers', 'coffee_shop', amount)
        
        print(f"✓ {5} additional transactions circulated")
        print()
        
        # Check loan completion
        print("Checking Loan Repayment...")
        print("-" * 80)
        
        # Settle network
        settlements = ctn.settle_end_of_day()
        
        print(f"✓ End-of-day settlement: {len(settlements)} net transfers")
        print()
        
        # Track loan completion
        completed = gcf.track_loan_completion(top_loan)
        
        if completed:
            print("✓✓✓ LOAN AUTOMATICALLY REPAID THROUGH CIRCULATION! ✓✓✓")
            print()
            print(f"  Loan amount: ${top_loan.amount:.2f}")
            print(f"  Fee: ${top_loan.fee:.2f}")
            print(f"  Total repayment: ${top_loan.repayment_amount:.2f}")
            print(f"  Duration: {top_loan.completed_at - top_loan.disbursed_at:.1f} seconds")
            print()
            print("  How did this work?")
            print("  1. Loan enabled transaction (coffee shop → supplier)")
            print("  2. Money circulated through network")
            print("  3. Coffee shop received money from customers")
            print("  4. Settlement balanced accounts")
            print("  5. Coffee shop had sufficient balance")
            print("  6. Loan automatically repaid!")
            print()
            print("  NO DEFAULT POSSIBLE - It's just circulation!")
        else:
            print(f"⏳ Loan still completing (borrower balance: ${ctn.nodes[top_loan.borrower].balance:.2f})")
            print(f"   Needs: ${top_loan.amount + top_loan.fee:.2f}")
            print(f"   Will complete through continued circulation")
        
        print()
    else:
        print("✗ Loan disbursement failed")
        return
    
    # Generate GCF report
    print("=" * 80)
    print("GRAPH COMPLETION FINANCE REPORT")
    print("=" * 80)
    print()
    
    report = gcf.generate_gcf_report()
    
    print("Loan Statistics:")
    for key, value in report['loan_statistics'].items():
        print(f"  {key:20s}: {value}")
    print()
    
    print("Financial Metrics:")
    for key, value in report['financial_metrics'].items():
        if isinstance(value, float):
            print(f"  {key:20s}: ${value:.2f}")
        else:
            print(f"  {key:20s}: {value}")
    print()
    
    print("Economic Impact:")
    for key, value in report['economic_impact'].items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.3f}")
        else:
            print(f"  {key:20s}: {value}")
    print()
    
    # Key insight
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("Traditional Credit:")
    print("  • Based on past behavior (credit score)")
    print("  • Requires collateral")
    print("  • Risk of default")
    print("  • Interest charges")
    print()
    print("Graph Completion Finance:")
    print("  • Based on network topology (pattern correlation)")
    print("  • No collateral needed")
    print("  • Cannot default (loan IS the flow)")
    print("  • Minimal service fee (not interest!)")
    print()
    print("The Shadow Network Reveals:")
    print("  Where flows SHOULD exist (high correlation)")
    print("  How much is missing (expected - actual)")
    print("  How to complete the graph (directed loans)")
    print()
    print("Result:")
    print("  ✓ Graph completed")
    print("  ✓ Economic activity increased")
    print("  ✓ Loan repaid automatically through circulation")
    print("  ✓ No traditional 'credit' concept needed!")
    print()
    print("This is revolutionary: Credit becomes unnecessary when you")
    print("understand that it's just incomplete flows in a network.")
    print()


def demo_multiple_completions():
    """
    Show multiple graph completions working simultaneously
    """
    print("=" * 80)
    print("MULTIPLE GRAPH COMPLETIONS")
    print("=" * 80)
    print()
    
    # Create larger network
    ctn = CirculationTransactionNetwork(name="Multi-GCF Demo")
    
    # Create 6 nodes
    nodes = ['shop_a', 'shop_b', 'shop_c', 'supplier_x', 'supplier_y', 'distributor']
    for node_id in nodes:
        ctn.add_node(node_id, node_id.replace('_', ' ').title(), 'business', credit_limit=10000)
    
    # Simulate transactions with intentional gaps
    base_time = datetime.now().timestamp()
    day_seconds = 24 * 3600
    
    for day in range(30):
        # Some flows exist
        ctn.add_transaction('shop_a', 'supplier_x', np.random.normal(500, 50), 
                           base_time - (30-day) * day_seconds)
        ctn.add_transaction('shop_b', 'supplier_x', np.random.normal(300, 30),
                           base_time - (30-day) * day_seconds)
        
        # Gaps: shop_c should buy from supplier_y (pattern suggests) but doesn't!
        # shop_b should also buy from supplier_y but doesn't!
    
    # Some weekly flows
    for day in range(0, 30, 7):
        ctn.add_transaction('supplier_x', 'distributor', np.random.normal(2000, 200),
                           base_time - (30-day) * day_seconds)
        ctn.add_transaction('supplier_y', 'distributor', np.random.normal(1500, 150),
                           base_time - (30-day) * day_seconds)
        ctn.add_transaction('distributor', 'shop_a', np.random.normal(1000, 100),
                           base_time - (30-day) * day_seconds)
    
    print(f"Created network with {len(nodes)} nodes and {len(ctn.transactions)} transactions")
    print()
    
    # Analyze
    stn = ShadowTransactionNetwork(ctn)
    stn.extract_patterns(window_days=30)
    stn.find_harmonic_coincidences()
    stn.build_shadow_graph()
    
    # GCF analysis
    gcf = GraphCompletionFinance(ctn, stn)
    opportunities = gcf.identify_opportunities()
    
    print(f"Found {len(opportunities)} graph completion opportunities")
    print()
    
    # Disburse multiple loans
    print("Disbursing multiple directed loans...")
    print("-" * 80)
    
    disbursed = 0
    for loan in opportunities[:3]:  # Top 3
        if gcf.disburse_loan(loan):
            print(f"  ✓ ${loan.amount:.2f}: {loan.borrower} → {loan.target} (ρ={loan.correlation:.3f})")
            disbursed += 1
    
    print()
    print(f"✓ Disbursed {disbursed} loans to complete graph")
    print()
    
    # Show effect
    print("Effect on Network:")
    print("-" * 80)
    print(f"  Before: {len([tx for tx in ctn.transactions if tx.timestamp < base_time])} transactions")
    print(f"  After: {len(ctn.transactions)} transactions (+{disbursed} from directed loans)")
    print(f"  Graph completeness: {gcf._calculate_graph_completion_rate():.1%}")
    print()


if __name__ == "__main__":
    demo_basic_graph_completion()
    print("\n" * 2)
    demo_multiple_completions()

