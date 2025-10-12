"""
CTN Visualization Tools
=======================

Visualization utilities for Circulation Transaction Network and Shadow Network.

Features:
- Transaction flow visualization (circuit diagrams)
- Shadow network graph visualization
- Pattern spectrum analysis
- Time series visualization
- Real-time monitoring dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from transaction_graph import CirculationTransactionNetwork, Node, Transaction
from shadow_network import ShadowTransactionNetwork, TransactionPattern, VirtualTransaction


# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


class CTNVisualizer:
    """Visualization tools for CTN"""
    
    def __init__(self, ctn: CirculationTransactionNetwork):
        self.ctn = ctn
        
    def plot_transaction_flow(self, filename: Optional[str] = None):
        """
        Visualize transaction network as circuit diagram
        
        Nodes = Circuit elements
        Edges = Transaction flows (currents)
        Colors = Node types (business, bank, etc.)
        Widths = Transaction volumes
        """
        G = nx.DiGraph()
        
        # Add nodes
        node_colors = {
            'customer': '#3498db',  # Blue
            'business': '#2ecc71',  # Green
            'bank': '#e74c3c',      # Red
            'supplier': '#f39c12',  # Orange
        }
        
        for node_id, node in self.ctn.nodes.items():
            G.add_node(
                node_id,
                color=node_colors.get(node.node_type, '#95a5a6'),
                size=node.credit_limit / 1000
            )
        
        # Add edges (aggregate transaction volumes)
        edge_volumes = {}
        for tx in self.ctn.transactions:
            edge = (tx.from_node, tx.to_node)
            edge_volumes[edge] = edge_volumes.get(edge, 0) + tx.amount
        
        for (from_node, to_node), volume in edge_volumes.items():
            G.add_edge(from_node, to_node, weight=volume)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(14, 10))
        
        # Draw nodes
        node_colors_list = [G.nodes[node]['color'] for node in G.nodes()]
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors_list,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=2,
            edgecolors='black'
        )
        
        # Draw edges
        edge_widths = [G[u][v]['weight'] / 500 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color='#34495e',
            arrows=True,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            font_color='white'
        )
        
        plt.title("Transaction Flow Network\n(Node size = Credit limit, Edge width = Transaction volume)",
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_balance_history(self, node_id: str, filename: Optional[str] = None):
        """Plot balance history over time for a node"""
        if node_id not in self.ctn.nodes:
            print(f"Node {node_id} not found")
            return
        
        node = self.ctn.nodes[node_id]
        
        # Collect all transactions
        all_txs = sorted(node.inflows + node.outflows, key=lambda tx: tx.timestamp)
        
        if not all_txs:
            print(f"No transactions for node {node_id}")
            return
        
        # Calculate balance over time
        times = []
        balances = []
        current_balance = 0
        
        for tx in all_txs:
            times.append(datetime.fromtimestamp(tx.timestamp))
            if tx in node.inflows:
                current_balance += tx.amount
            else:
                current_balance -= tx.amount
            balances.append(current_balance)
        
        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(times, balances, linewidth=2, color='#3498db')
        plt.fill_between(times, balances, alpha=0.3, color='#3498db')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero balance')
        plt.axhline(y=node.credit_limit, color='green', linestyle='--', alpha=0.5, 
                    label=f'Credit limit (${node.credit_limit:,.0f})')
        
        plt.title(f"Balance History: {node.name}", fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Balance ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class ShadowNetworkVisualizer:
    """Visualization tools for Shadow Transaction Network"""
    
    def __init__(self, stn: ShadowTransactionNetwork):
        self.stn = stn
        
    def plot_pattern_spectrum(self, node_id: str, filename: Optional[str] = None):
        """
        Plot frequency spectrum for a node's transaction pattern
        
        Shows:
        - Fundamental frequency (dominant rhythm)
        - Harmonics (sub-patterns)
        - Amplitudes (pattern strength)
        """
        if node_id not in self.stn.patterns:
            print(f"Pattern not found for node {node_id}")
            return
        
        pattern = self.stn.patterns[node_id]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Frequency spectrum
        ax1 = axes[0]
        ax1.plot(pattern.frequencies, pattern.amplitudes, linewidth=1, color='#3498db', alpha=0.7)
        ax1.fill_between(pattern.frequencies, pattern.amplitudes, alpha=0.3, color='#3498db')
        
        # Mark fundamental
        fund_idx = np.abs(pattern.frequencies - pattern.omega_fundamental).argmin()
        ax1.plot(pattern.omega_fundamental, pattern.amplitudes[fund_idx], 
                'ro', markersize=10, label='Fundamental')
        
        # Mark harmonics
        for n, (amp, phase) in pattern.harmonics.items():
            freq = n * pattern.omega_fundamental
            freq_idx = np.abs(pattern.frequencies - freq).argmin()
            ax1.plot(freq, amp, 'go', markersize=8, alpha=0.7)
        
        ax1.set_xlabel("Frequency (Hz)", fontsize=12)
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.set_title(f"Transaction Pattern Spectrum: {node_id}\n"
                      f"Fundamental: {pattern.omega_fundamental:.6f} Hz "
                      f"(Period: {1.0/(pattern.omega_fundamental * 86400):.2f} days)",
                      fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Time series
        ax2 = axes[1]
        time_days = pattern.timestamps / 86400  # Convert to days
        ax2.plot(time_days, pattern.time_series, linewidth=2, color='#2ecc71')
        ax2.fill_between(time_days, pattern.time_series, alpha=0.3, color='#2ecc71')
        
        ax2.set_xlabel("Time (days)", fontsize=12)
        ax2.set_ylabel("Transaction Flow ($)", fontsize=12)
        ax2.set_title("Transaction Time Series", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_shadow_graph(self, filename: Optional[str] = None, 
                          min_correlation: float = 0.5):
        """
        Plot shadow network graph showing pattern correlations
        
        Nodes = Businesses/entities
        Edges = Virtual transactions (pattern correlations)
        Colors = Communities/risk clusters
        Widths = Correlation strength
        """
        if len(self.stn.shadow_graph.nodes) == 0:
            print("Shadow graph is empty")
            return
        
        # Filter to significant correlations
        G_filtered = nx.Graph()
        for node in self.stn.shadow_graph.nodes():
            G_filtered.add_node(node)
        
        for u, v, data in self.stn.shadow_graph.edges(data=True):
            if abs(data.get('weight', 0)) >= min_correlation:
                G_filtered.add_edge(u, v, **data)
        
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(G_filtered, weight='weight'))
        
        # Assign colors to communities
        node_colors = {}
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = community_colors[i]
        
        # Layout
        pos = nx.spring_layout(G_filtered, k=3, iterations=100, weight='weight')
        
        # Draw
        plt.figure(figsize=(16, 12))
        
        # Draw nodes
        node_color_list = [node_colors.get(node, [0.5, 0.5, 0.5, 1.0]) for node in G_filtered.nodes()]
        
        # Node sizes based on degree centrality
        degree_cent = nx.degree_centrality(G_filtered)
        node_sizes = [5000 * degree_cent.get(node, 0.1) for node in G_filtered.nodes()]
        
        nx.draw_networkx_nodes(
            G_filtered, pos,
            node_color=node_color_list,
            node_size=node_sizes,
            alpha=0.9,
            linewidths=2,
            edgecolors='black'
        )
        
        # Draw edges
        edge_widths = []
        edge_colors = []
        for u, v in G_filtered.edges():
            corr = G_filtered[u][v].get('weight', 0)
            edge_widths.append(abs(corr) * 5)
            edge_colors.append('#2ecc71' if corr > 0.7 else '#f39c12' if corr > 0.5 else '#95a5a6')
        
        nx.draw_networkx_edges(
            G_filtered, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color=edge_colors
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G_filtered, pos,
            font_size=9,
            font_weight='bold',
            font_color='white'
        )
        
        # Add legend for communities
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=community_colors[i], markersize=10,
                      label=f'Cluster {i+1} ({len(c)} nodes)')
            for i, c in enumerate(communities[:5])  # Top 5 clusters
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title("Shadow Transaction Network\n"
                  "(Pattern Correlations Revealing Hidden Market Structure)\n"
                  f"Green=High correlation (>0.7), Orange=Medium (>0.5)",
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_correlation_matrix(self, filename: Optional[str] = None):
        """Plot heatmap of pattern correlations between all nodes"""
        if not self.stn.patterns:
            print("No patterns available")
            return
        
        node_ids = sorted(self.stn.patterns.keys())
        n = len(node_ids)
        
        # Build correlation matrix
        corr_matrix = np.zeros((n, n))
        for i, node_a in enumerate(node_ids):
            for j, node_b in enumerate(node_ids):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Find virtual transaction between these nodes
                    vtx = next((v for v in self.stn.virtual_transactions 
                               if (v.node_a == node_a and v.node_b == node_b) or
                                  (v.node_a == node_b and v.node_b == node_a)), None)
                    if vtx:
                        corr_matrix[i, j] = vtx.correlation
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            xticklabels=node_ids,
            yticklabels=node_ids,
            cmap='RdYlGn',
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'}
        )
        
        plt.title("Transaction Pattern Correlation Matrix\n"
                  "(Reveals Hidden Relationships)",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_influence_network(self, filename: Optional[str] = None):
        """
        Plot influence network using PageRank
        
        Node size = Influence score
        Color = Community
        Shows who influences whom in the market
        """
        if len(self.stn.shadow_graph.nodes) == 0:
            print("Shadow graph is empty")
            return
        
        influence = self.stn.calculate_influence_scores()
        
        # Layout
        pos = nx.spring_layout(self.stn.shadow_graph, k=2, iterations=100)
        
        # Draw
        plt.figure(figsize=(16, 12))
        
        # Node sizes from influence
        node_sizes = [10000 * influence.get(node, 0.01) for node in self.stn.shadow_graph.nodes()]
        
        # Node colors from betweenness
        betweenness = nx.betweenness_centrality(self.stn.shadow_graph, weight='weight')
        node_colors = [betweenness.get(node, 0) for node in self.stn.shadow_graph.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            self.stn.shadow_graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='YlOrRd',
            alpha=0.9,
            linewidths=2,
            edgecolors='black'
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.stn.shadow_graph, pos,
            alpha=0.3,
            edge_color='#34495e'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.stn.shadow_graph, pos,
            font_size=9,
            font_weight='bold',
            font_color='white'
        )
        
        # Colorbar
        plt.colorbar(nodes, label='Betweenness Centrality (Market Control)')
        
        # Highlight top influencers
        top_5 = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (node_id, score) in enumerate(top_5, 1):
            x, y = pos[node_id]
            plt.annotate(f"#{i} Influencer\n{score:.3f}",
                        xy=(x, y), xytext=(10, 10 + i*15),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.title("Market Influence Network\n"
                  "(Node size = Influence score, Color = Centrality)",
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_interactive_dashboard(self, filename: str = "shadow_network_dashboard.html"):
        """
        Create interactive Plotly dashboard
        
        Contains:
        - Shadow network graph (interactive)
        - Pattern spectra
        - Correlation matrix
        - Time series
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Shadow Network Graph",
                "Pattern Correlations",
                "Transaction Patterns",
                "Influence Scores"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Shadow network graph
        if len(self.stn.shadow_graph.nodes) > 0:
            pos = nx.spring_layout(self.stn.shadow_graph, k=2, iterations=50)
            
            # Add edges
            edge_traces = []
            for u, v, data in self.stn.shadow_graph.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=abs(data.get('weight', 0))*3, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
                fig.add_trace(edge_trace, row=1, col=1)
            
            # Add nodes
            node_x = [pos[node][0] for node in self.stn.shadow_graph.nodes()]
            node_y = [pos[node][1] for node in self.stn.shadow_graph.nodes()]
            node_text = list(self.stn.shadow_graph.nodes())
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='top center',
                marker=dict(size=15, color='#3498db'),
                showlegend=False
            )
            fig.add_trace(node_trace, row=1, col=1)
        
        # 2. Correlation heatmap
        if self.stn.patterns:
            node_ids = sorted(self.stn.patterns.keys())
            n = len(node_ids)
            corr_matrix = np.zeros((n, n))
            
            for i, node_a in enumerate(node_ids):
                for j, node_b in enumerate(node_ids):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        vtx = next((v for v in self.stn.virtual_transactions 
                                   if (v.node_a == node_a and v.node_b == node_b) or
                                      (v.node_a == node_b and v.node_b == node_a)), None)
                        if vtx:
                            corr_matrix[i, j] = vtx.correlation
            
            heatmap = go.Heatmap(
                z=corr_matrix,
                x=node_ids,
                y=node_ids,
                colorscale='RdYlGn',
                zmid=0,
                showscale=False
            )
            fig.add_trace(heatmap, row=1, col=2)
        
        # 3. Pattern time series (first node)
        if self.stn.patterns:
            first_node = list(self.stn.patterns.keys())[0]
            pattern = self.stn.patterns[first_node]
            
            time_days = pattern.timestamps / 86400
            pattern_trace = go.Scatter(
                x=time_days,
                y=pattern.time_series,
                mode='lines',
                name=first_node,
                line=dict(color='#2ecc71', width=2),
                fill='tozeroy',
                showlegend=False
            )
            fig.add_trace(pattern_trace, row=2, col=1)
        
        # 4. Influence scores
        if len(self.stn.shadow_graph.nodes) > 0:
            influence = self.stn.calculate_influence_scores()
            sorted_influence = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:10]
            
            bar_trace = go.Bar(
                x=[node for node, _ in sorted_influence],
                y=[score for _, score in sorted_influence],
                marker=dict(color='#e74c3c'),
                showlegend=False
            )
            fig.add_trace(bar_trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Shadow Transaction Network - Interactive Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save
        fig.write_html(filename)
        print(f"✓ Interactive dashboard saved to {filename}")


def generate_all_visualizations(ctn: CirculationTransactionNetwork, 
                                 stn: ShadowTransactionNetwork,
                                 output_dir: str = "visualizations"):
    """Generate complete visualization suite"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    print("-" * 80)
    
    # CTN visualizations
    ctn_viz = CTNVisualizer(ctn)
    print("1. Transaction flow network...")
    ctn_viz.plot_transaction_flow(f"{output_dir}/transaction_flow.png")
    
    # Balance histories for first few nodes
    for i, node_id in enumerate(list(ctn.nodes.keys())[:3], 2):
        print(f"{i}. Balance history: {node_id}...")
        ctn_viz.plot_balance_history(node_id, f"{output_dir}/balance_{node_id}.png")
    
    # Shadow network visualizations
    stn_viz = ShadowNetworkVisualizer(stn)
    
    # Pattern spectra for first few nodes
    for i, node_id in enumerate(list(stn.patterns.keys())[:3], 5):
        print(f"{i}. Pattern spectrum: {node_id}...")
        stn_viz.plot_pattern_spectrum(node_id, f"{output_dir}/spectrum_{node_id}.png")
    
    print("8. Shadow network graph...")
    stn_viz.plot_shadow_graph(f"{output_dir}/shadow_network.png")
    
    print("9. Correlation matrix...")
    stn_viz.plot_correlation_matrix(f"{output_dir}/correlation_matrix.png")
    
    print("10. Influence network...")
    stn_viz.plot_influence_network(f"{output_dir}/influence_network.png")
    
    print("11. Interactive dashboard...")
    stn_viz.create_interactive_dashboard(f"{output_dir}/interactive_dashboard.html")
    
    print()
    print(f"✓ All visualizations saved to {output_dir}/")

