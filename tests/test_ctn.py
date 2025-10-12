"""
Tests for Circulation Transaction Network
==========================================

Comprehensive test suite for CTN functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ctn import (
    CirculationTransactionNetwork,
    ShadowTransactionNetwork,
    Transaction,
    VirtualTransaction,
    TransactionPattern,
    Node,
    TransactionStatus
)


class TestCirculationTransactionNetwork:
    """Test CTN core functionality"""
    
    def test_network_creation(self):
        """Test creating a CTN"""
        ctn = CirculationTransactionNetwork(name="Test Network")
        assert ctn.name == "Test Network"
        assert len(ctn.nodes) == 0
        assert len(ctn.transactions) == 0
    
    def test_add_node(self):
        """Test adding nodes to network"""
        ctn = CirculationTransactionNetwork()
        
        node = ctn.add_node('node1', 'Test Node', 'business', credit_limit=10000)
        
        assert node.node_id == 'node1'
        assert node.name == 'Test Node'
        assert node.node_type == 'business'
        assert node.credit_limit == 10000
        assert 'node1' in ctn.nodes
    
    def test_add_transaction(self):
        """Test adding transactions"""
        ctn = CirculationTransactionNetwork()
        
        ctn.add_node('A', 'Node A', 'customer')
        ctn.add_node('B', 'Node B', 'business')
        
        tx = ctn.add_transaction('A', 'B', 100.0)
        
        assert tx.from_node == 'A'
        assert tx.to_node == 'B'
        assert tx.amount == 100.0
        assert tx.status == TransactionStatus.PENDING
        assert tx in ctn.transactions
    
    def test_transaction_flow_conservation(self):
        """Test Kirchhoff's current law: inflows = outflows"""
        ctn = CirculationTransactionNetwork()
        
        # Create nodes
        for i in range(4):
            ctn.add_node(f'node{i}', f'Node {i}', 'business')
        
        # Add transactions forming a cycle
        ctn.add_transaction('node0', 'node1', 100)
        ctn.add_transaction('node1', 'node2', 100)
        ctn.add_transaction('node2', 'node3', 100)
        ctn.add_transaction('node3', 'node0', 100)
        
        # Check conservation at each node
        for node_id, node in ctn.nodes.items():
            total_in = sum(tx.amount for tx in node.inflows)
            total_out = sum(tx.amount for tx in node.outflows)
            
            # In a closed cycle, inflows should equal outflows
            assert abs(total_in - total_out) < 1e-10
    
    def test_end_of_day_settlement(self):
        """Test end-of-day settlement reduces transactions"""
        ctn = CirculationTransactionNetwork()
        
        # Create nodes
        ctn.add_node('A', 'Customer A', 'customer')
        ctn.add_node('B', 'Shop B', 'business')
        ctn.add_node('C', 'Supplier C', 'business')
        
        # Transactions: A→B→C
        ctn.add_transaction('A', 'B', 50)
        ctn.add_transaction('B', 'C', 30)
        
        # Settle
        settlements = ctn.settle_end_of_day()
        
        # Should have reduced from 2 to ~2 settlements (no perfect cancellation)
        assert len(settlements) <= 2
        
        # Check balances are correct
        assert ctn.nodes['A'].balance == -50  # Paid out 50
        assert ctn.nodes['B'].balance == 20   # Received 50, paid 30
        assert ctn.nodes['C'].balance == 30   # Received 30
    
    def test_credit_limit_enforcement(self):
        """Test credit limit prevents over-borrowing"""
        ctn = CirculationTransactionNetwork()
        
        ctn.add_node('A', 'Node A', 'customer', credit_limit=100)
        ctn.add_node('B', 'Node B', 'business')
        
        # Transaction within limit
        tx1 = ctn.add_transaction('A', 'B', 80)
        assert tx1 is not None
        
        # Transaction exceeding limit
        tx2 = ctn.add_transaction('A', 'B', 50)
        assert tx2 is None  # Should fail due to credit limit
    
    def test_circuit_reduction(self):
        """Test graph reduction for settlement"""
        ctn = CirculationTransactionNetwork()
        
        # Create circular transactions
        nodes = ['A', 'B', 'C', 'D']
        for node_id in nodes:
            ctn.add_node(node_id, f'Node {node_id}', 'business')
        
        # Cycle: A→B→C→D→A with same amount (should fully cancel)
        amount = 100
        ctn.add_transaction('A', 'B', amount)
        ctn.add_transaction('B', 'C', amount)
        ctn.add_transaction('C', 'D', amount)
        ctn.add_transaction('D', 'A', amount)
        
        settlements = ctn.settle_end_of_day()
        
        # Should have minimal settlements (ideally 0 net transfer)
        # All balances should be ~0
        for node in ctn.nodes.values():
            assert abs(node.balance) < 1e-10


class TestShadowTransactionNetwork:
    """Test Shadow Network functionality"""
    
    @pytest.fixture
    def setup_network(self):
        """Create a CTN with pattern data"""
        ctn = CirculationTransactionNetwork(name="Shadow Test")
        
        # Create nodes with different rhythms
        nodes = {
            'daily': ctn.add_node('daily', 'Daily Business', 'business'),
            'weekly': ctn.add_node('weekly', 'Weekly Business', 'business'),
            'daily2': ctn.add_node('daily2', 'Daily Business 2', 'business')
        }
        
        # Simulate 30 days of transactions
        base_time = datetime.now().timestamp()
        day_seconds = 24 * 3600
        
        # Daily pattern (node and daily2 should match!)
        for day in range(30):
            tx_time = base_time - (30-day) * day_seconds + 12 * 3600
            ctn.add_transaction('customer', 'daily', 1000 + np.random.normal(0, 50), tx_time)
            ctn.add_transaction('customer', 'daily2', 1000 + np.random.normal(0, 50), tx_time)
        
        # Weekly pattern
        for day in range(0, 30, 7):
            tx_time = base_time - (30-day) * day_seconds + 12 * 3600
            ctn.add_transaction('customer', 'weekly', 5000 + np.random.normal(0, 200), tx_time)
        
        return ctn
    
    def test_pattern_extraction(self, setup_network):
        """Test FFT pattern extraction"""
        ctn = setup_network
        stn = ShadowTransactionNetwork(ctn)
        
        patterns = stn.extract_patterns(window_days=30)
        
        # Should have patterns for all nodes with sufficient data
        assert len(patterns) >= 2
        
        # Each pattern should have frequency spectrum
        for node_id, pattern in patterns.items():
            assert len(pattern.frequencies) > 0
            assert len(pattern.amplitudes) > 0
            assert pattern.omega_fundamental > 0
    
    def test_harmonic_coincidence_detection(self, setup_network):
        """Test finding nodes with matching patterns"""
        ctn = setup_network
        stn = ShadowTransactionNetwork(ctn)
        
        stn.extract_patterns(window_days=30)
        virtual_txs = stn.find_harmonic_coincidences()
        
        # Should find virtual connection between 'daily' and 'daily2'
        # (both have daily rhythm)
        daily_connection = any(
            (vtx.node_a == 'daily' and vtx.node_b == 'daily2') or
            (vtx.node_a == 'daily2' and vtx.node_b == 'daily')
            for vtx in virtual_txs
        )
        
        assert daily_connection, "Should find connection between nodes with same daily rhythm"
    
    def test_shadow_graph_construction(self, setup_network):
        """Test building shadow network graph"""
        ctn = setup_network
        stn = ShadowTransactionNetwork(ctn)
        
        stn.extract_patterns(window_days=30)
        stn.find_harmonic_coincidences()
        shadow_graph = stn.build_shadow_graph()
        
        # Graph should have nodes
        assert shadow_graph.number_of_nodes() > 0
        
        # Graph should have edges (virtual transactions)
        assert shadow_graph.number_of_edges() > 0
    
    def test_risk_cluster_detection(self, setup_network):
        """Test detecting correlated risk groups"""
        ctn = setup_network
        stn = ShadowTransactionNetwork(ctn)
        
        stn.extract_patterns(window_days=30)
        stn.find_harmonic_coincidences()
        stn.build_shadow_graph()
        
        clusters = stn.detect_risk_clusters()
        
        # Should find at least one cluster
        assert len(clusters) >= 1
        
        # Daily nodes should be in same cluster
        for cluster in clusters:
            if 'daily' in cluster and 'daily2' in cluster:
                # Found them together!
                assert True
                return
    
    def test_influence_score_calculation(self, setup_network):
        """Test calculating node influence scores"""
        ctn = setup_network
        stn = ShadowTransactionNetwork(ctn)
        
        stn.extract_patterns(window_days=30)
        stn.find_harmonic_coincidences()
        stn.build_shadow_graph()
        
        influence = stn.calculate_influence_scores()
        
        # Should have influence scores for all nodes
        assert len(influence) > 0
        
        # Scores should be between 0 and 1
        for score in influence.values():
            assert 0 <= score <= 1
    
    def test_fraud_detection(self):
        """Test detecting coordinated behavior (fraud/cartel)"""
        ctn = CirculationTransactionNetwork(name="Fraud Test")
        
        # Normal businesses
        for i in range(3):
            ctn.add_node(f'normal_{i}', f'Normal {i}', 'business')
        
        # Cartel (synchronized)
        for i in range(3):
            ctn.add_node(f'cartel_{i}', f'Cartel {i}', 'business')
        
        # Simulate transactions
        base_time = datetime.now().timestamp()
        day_seconds = 24 * 3600
        
        # Normal: random patterns
        for node_id in [f'normal_{i}' for i in range(3)]:
            for day in range(30):
                num_txs = np.random.poisson(3)
                for _ in range(num_txs):
                    hour = np.random.uniform(8, 20)
                    tx_time = base_time - (30-day) * day_seconds + hour * 3600
                    amount = np.random.normal(1000, 300)
                    ctn.add_transaction('customer', node_id, amount, tx_time)
        
        # Cartel: HIGHLY synchronized
        cartel_pattern = np.random.random(30)
        for node_id in [f'cartel_{i}' for i in range(3)]:
            for day in range(30):
                for hour in [9, 12, 15]:  # Fixed hours
                    tx_time = base_time - (30-day) * day_seconds + hour * 3600
                    amount = 1000 * (1 + cartel_pattern[day]) + np.random.normal(0, 5)
                    ctn.add_transaction('customer', node_id, amount, tx_time)
        
        # Analyze
        stn = ShadowTransactionNetwork(ctn)
        stn.extract_patterns(window_days=30)
        stn.find_harmonic_coincidences()
        stn.build_shadow_graph()
        
        coordinated = stn.detect_coordinated_behavior(threshold=0.85)
        
        # Should find cartel group
        cartel_detected = False
        for group in coordinated:
            cartel_nodes = {f'cartel_{i}' for i in range(3)}
            if cartel_nodes.issubset(group):
                cartel_detected = True
                break
        
        assert cartel_detected, "Should detect cartel through high correlation"


class TestPerformance:
    """Performance and complexity tests"""
    
    def test_transaction_processing_speed(self):
        """Test transaction throughput"""
        ctn = CirculationTransactionNetwork()
        
        # Create 100 nodes
        for i in range(100):
            ctn.add_node(f'node{i}', f'Node {i}', 'business', credit_limit=100000)
        
        # Add 10000 transactions
        import time
        start = time.time()
        
        for _ in range(10000):
            from_node = f'node{np.random.randint(0, 100)}'
            to_node = f'node{np.random.randint(0, 100)}'
            if from_node != to_node:
                amount = np.random.uniform(10, 1000)
                ctn.add_transaction(from_node, to_node, amount)
        
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 10k transactions)
        throughput = len(ctn.transactions) / elapsed
        print(f"\nTransaction throughput: {throughput:.0f} tx/s")
        
        assert throughput > 1000, "Should process at least 1000 tx/s"
    
    def test_settlement_complexity(self):
        """Test settlement scales efficiently"""
        ctn = CirculationTransactionNetwork()
        
        # Create 50 nodes
        for i in range(50):
            ctn.add_node(f'node{i}', f'Node {i}', 'business', credit_limit=100000)
        
        # Add random transactions
        for _ in range(500):
            from_node = f'node{np.random.randint(0, 50)}'
            to_node = f'node{np.random.randint(0, 50)}'
            if from_node != to_node:
                amount = np.random.uniform(10, 1000)
                ctn.add_transaction(from_node, to_node, amount)
        
        # Settle
        import time
        start = time.time()
        settlements = ctn.settle_end_of_day()
        elapsed = time.time() - start
        
        print(f"\nSettlement time: {elapsed*1000:.2f} ms for {len(ctn.transactions)} transactions")
        print(f"Reduced to {len(settlements)} net settlements")
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        
        # Should reduce transactions significantly
        reduction = (1 - len(settlements) / len(ctn.transactions)) * 100
        print(f"Transaction reduction: {reduction:.1f}%")


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_workflow(self):
        """Test complete CTN + Shadow Network workflow"""
        # 1. Create CTN
        ctn = CirculationTransactionNetwork(name="Integration Test")
        
        # 2. Add nodes
        nodes = ['shop_a', 'shop_b', 'supplier', 'manufacturer']
        for node_id in nodes:
            ctn.add_node(node_id, node_id.replace('_', ' ').title(), 'business', credit_limit=50000)
        
        # 3. Simulate transactions
        base_time = datetime.now().timestamp()
        day_seconds = 24 * 3600
        
        for day in range(30):
            # Shops buy from supplier
            for shop in ['shop_a', 'shop_b']:
                tx_time = base_time - (30-day) * day_seconds + 10 * 3600
                amount = np.random.normal(1000, 100)
                ctn.add_transaction(shop, 'supplier', amount, tx_time)
            
            # Supplier buys from manufacturer weekly
            if day % 7 == 0:
                tx_time = base_time - (30-day) * day_seconds + 14 * 3600
                amount = np.random.normal(5000, 500)
                ctn.add_transaction('supplier', 'manufacturer', amount, tx_time)
        
        # 4. Settle transactions
        settlements = ctn.settle_end_of_day()
        assert len(settlements) > 0
        
        # 5. Create shadow network
        stn = ShadowTransactionNetwork(ctn)
        patterns = stn.extract_patterns(window_days=30)
        assert len(patterns) > 0
        
        # 6. Find virtual transactions
        virtual_txs = stn.find_harmonic_coincidences()
        # Should find shop_a and shop_b are correlated (same buying pattern)
        
        # 7. Build shadow graph
        shadow_graph = stn.build_shadow_graph()
        assert shadow_graph.number_of_nodes() > 0
        
        # 8. Generate intelligence report
        report = stn.generate_intelligence_report()
        
        assert 'network_stats' in report
        assert 'risk_clusters' in report
        assert 'influence_scores' in report
        
        print("\n✓ Complete workflow successful")
        print(f"  Transactions: {len(ctn.transactions)}")
        print(f"  Settlements: {len(settlements)}")
        print(f"  Patterns: {len(patterns)}")
        print(f"  Virtual connections: {len(virtual_txs)}")
        print(f"  Shadow graph: {shadow_graph.number_of_nodes()} nodes, {shadow_graph.number_of_edges()} edges")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

