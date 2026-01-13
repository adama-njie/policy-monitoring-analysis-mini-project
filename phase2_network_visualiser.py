"""
Phase 2: Network Visualization for Document Relationships
==========================================================
Create interactive network graphs showing policy document connections.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import networkx as nx

class DocumentNetworkVisualizer:
    """
    Create network visualizations of document relationships.
    """
    
    def __init__(self, comparison_dir: str):
        """
        Initialize visualizer.
        
        Args:
            comparison_dir: Directory containing comparison results
        """
        self.comparison_dir = Path(comparison_dir)
        
        # Load similarity matrix
        print("Loading similarity matrix...")
        self.similarity_df = pd.read_csv(
            self.comparison_dir / "similarity_matrix.csv",
            index_col=0
        )
        
        # Load comparison report - with fallback if JSON is corrupted
        print("Loading comparison report...")
        try:
            with open(self.comparison_dir / "comparison_report.json", 'r') as f:
                self.report = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: JSON file corrupted. Creating minimal report from similarity matrix.")
            # Create minimal report from similarity matrix
            self.report = self._create_minimal_report_from_similarity()
        
        print(f"✓ Loaded data for {len(self.similarity_df)} documents")
    
    def _create_minimal_report_from_similarity(self) -> Dict:
        """Create a minimal report structure from similarity matrix if JSON is corrupted."""
        # Extract document info from similarity matrix
        doc_names = list(self.similarity_df.index)
        
        # Create similar pairs
        similar_pairs = []
        for i in range(len(self.similarity_df)):
            for j in range(i + 1, len(self.similarity_df)):
                similar_pairs.append({
                    'doc1': doc_names[i],
                    'doc2': doc_names[j],
                    'similarity': float(self.similarity_df.iloc[i, j]),
                    'doc1_type': 'unknown',
                    'doc2_type': 'unknown'
                })
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Create conflicts (low similarity)
        conflicts = [p for p in similar_pairs if p['similarity'] < 0.3]
        
        return {
            'top_similar_pairs': similar_pairs[:30],
            'potential_conflicts': conflicts[:30]
        }
    
    def create_network_graph(self, similarity_threshold: float = 0.5) -> nx.Graph:
        """
        Create NetworkX graph from similarity matrix.
        
        Args:
            similarity_threshold: Minimum similarity to create an edge
            
        Returns:
            NetworkX graph
        """
        print(f"\nCreating network graph (threshold: {similarity_threshold})...")
        
        G = nx.Graph()
        
        # Add nodes
        for doc_name in self.similarity_df.index:
            G.add_node(doc_name)
        
        # Add edges based on similarity threshold
        edge_count = 0
        for i in range(len(self.similarity_df)):
            for j in range(i + 1, len(self.similarity_df)):
                similarity = self.similarity_df.iloc[i, j]
                
                if similarity >= similarity_threshold:
                    G.add_edge(
                        self.similarity_df.index[i],
                        self.similarity_df.index[j],
                        weight=float(similarity)
                    )
                    edge_count += 1
        
        print(f"✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def visualize_interactive_network(self, G: nx.Graph, output_path: str = None):
        """
        Create interactive network visualization using Plotly.
        
        Args:
            G: NetworkX graph
            output_path: Path to save HTML file
        """
        print("\nCreating interactive network visualization...")
        
        # Compute layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            connections = len(list(G.neighbors(node)))
            node_text.append(f"{node}<br>Connections: {connections}")
            node_size.append(10 + connections * 3)  # Size based on connections
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n.split('.')[0][:20] for n in G.nodes()],  # Short labels
            hovertext=node_text,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    x=1.02
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='Policy Document Network<br>Node size = number of connections',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Interactive network saved to {output_path}")
        
        return fig
    
    def visualize_similarity_network_by_type(self, output_path: str = None):
        """
        Create network visualization colored by document type.
        
        Args:
            output_path: Path to save HTML file
        """
        print("\nCreating type-colored network visualization...")
        
        # Create graph
        G = self.create_network_graph(similarity_threshold=0.45)
        
        # Get document types from report
        doc_types = {}
        for pair in self.report['top_similar_pairs']:
            doc_types[pair['doc1']] = pair['doc1_type']
            doc_types[pair['doc2']] = pair['doc2_type']
        
        # Assign colors to document types
        unique_types = list(set(doc_types.values()))
        color_map = {dtype: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)] 
                     for i, dtype in enumerate(unique_types)}
        
        # Compute layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create traces for each document type
        traces = []
        
        for doc_type in unique_types:
            # Get nodes of this type
            type_nodes = [node for node in G.nodes() if doc_types.get(node) == doc_type]
            
            if not type_nodes:
                continue
            
            node_x = [pos[node][0] for node in type_nodes]
            node_y = [pos[node][1] for node in type_nodes]
            node_text = [f"{node}<br>Type: {doc_type}" for node in type_nodes]
            
            trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                name=doc_type.capitalize(),
                text=[n.split('.')[0][:15] for n in type_nodes],
                hovertext=node_text,
                textposition="top center",
                textfont=dict(size=7),
                marker=dict(
                    size=15,
                    color=color_map[doc_type],
                    line=dict(width=2, color='white')
                )
            )
            traces.append(trace)
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.3, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + traces,
            layout=go.Layout(
                title=dict(
                    text='Policy Document Network by Type',
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Type-colored network saved to {output_path}")
        
        return fig
    
    def create_coherence_map(self, output_path: str = None):
        """
        Create a coherence map showing policy alignment and conflicts.
        
        Args:
            output_path: Path to save HTML file
        """
        print("\nCreating policy coherence map...")
        
        # Get conflict data
        conflicts = self.report.get('potential_conflicts', [])[:30]  # Top 30 conflicts
        similar_pairs = self.report['top_similar_pairs'][:30]  # Top 30 similar
        
        # Create graph with both types of connections
        G = nx.Graph()
        
        # Add similar connections (green)
        for pair in similar_pairs:
            G.add_edge(
                pair['doc1'],
                pair['doc2'],
                weight=pair['similarity'],
                connection_type='coherent'
            )
        
        # Add conflict connections (red)
        for conflict in conflicts:
            if not G.has_edge(conflict['doc1'], conflict['doc2']):
                G.add_edge(
                    conflict['doc1'],
                    conflict['doc2'],
                    weight=1 - conflict['similarity'],
                    connection_type='conflict'
                )
        
        # Compute layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create edge traces for each type
        coherent_edges_x = []
        coherent_edges_y = []
        conflict_edges_x = []
        conflict_edges_y = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            if edge[2].get('connection_type') == 'coherent':
                coherent_edges_x.extend([x0, x1, None])
                coherent_edges_y.extend([y0, y1, None])
            else:
                conflict_edges_x.extend([x0, x1, None])
                conflict_edges_y.extend([y0, y1, None])
        
        # Coherent edge trace
        coherent_trace = go.Scatter(
            x=coherent_edges_x,
            y=coherent_edges_y,
            line=dict(width=1.5, color='green'),
            hoverinfo='none',
            mode='lines',
            name='Coherent',
            showlegend=True
        )
        
        # Conflict edge trace
        conflict_trace = go.Scatter(
            x=conflict_edges_x,
            y=conflict_edges_y,
            line=dict(width=1.5, color='red', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Conflicting',
            showlegend=True
        )
        
        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [node for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[n.split('.')[0][:15] for n in G.nodes()],
            hovertext=node_text,
            textposition="top center",
            textfont=dict(size=7),
            showlegend=False,
            marker=dict(
                size=12,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[coherent_trace, conflict_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='Policy Coherence Map<br>Green = Coherent | Red = Potential Conflicts',
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Coherence map saved to {output_path}")
        
        return fig


def main():
    """Main execution function."""
    print("="*80)
    print("PHASE 2: NETWORK VISUALIZATION")
    print("="*80)
    
    # Initialize visualizer
    visualizer = DocumentNetworkVisualizer("preprocessed_output/document_comparison")
    
    # Create basic network graph
    G = visualizer.create_network_graph(similarity_threshold=0.5)
    
    # Generate visualizations
    visualizer.visualize_interactive_network(
        G,
        output_path="preprocessed_output/document_comparison/network_interactive.html"
    )
    
    visualizer.visualize_similarity_network_by_type(
        output_path="preprocessed_output/document_comparison/network_by_type.html"
    )
    
    visualizer.create_coherence_map(
        output_path="preprocessed_output/document_comparison/coherence_map.html"
    )
    
    print("\n" + "="*80)
    print("✅ Network Visualizations Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - network_interactive.html (interactive network)")
    print("  - network_by_type.html (colored by document type)")
    print("  - coherence_map.html (shows agreements and conflicts)")
    print("\nOpen these HTML files in your browser to explore!")


if __name__ == "__main__":
    main()