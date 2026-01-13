"""
Phase 3: Entity Network Visualization
======================================
Create network visualizations of entity relationships.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List

class EntityNetworkVisualizer:
    """
    Create network visualizations of entity relationships.
    """
    
    def __init__(self, ner_dir: str, use_cleaned: bool = True):
        """
        Initialize visualizer.
        
        Args:
            ner_dir: Directory containing NER results
            use_cleaned: Whether to use cleaned data (default: True)
        """
        self.ner_dir = Path(ner_dir)
        
        # Load entities - prefer DEEP_CLEANED, then CLEANED, then original
        print("Loading entity data...")
        
        if use_cleaned:
            # Try deep cleaned first
            deep_cleaned_path = self.ner_dir / "all_entities_ULTIMATE_CLEANED.csv" #all_entities_DEEP_CLEANED
            cleaned_path = self.ner_dir / "all_entities_ULTIMATE_CLEANED.csv"
            
            if deep_cleaned_path.exists():
                print("  Using DEEP_CLEANED data")
                self.entities_df = pd.read_csv(deep_cleaned_path)
            elif cleaned_path.exists():
                print("  Using CLEANED data")
                self.entities_df = pd.read_csv(cleaned_path)
            else:
                print("  ⚠️  No cleaned data found, using original")
                self.entities_df = pd.read_csv(self.ner_dir / "all_entities.csv")
        else:
            self.entities_df = pd.read_csv(self.ner_dir / "all_entities.csv")
        
        # Create entities by document structure
        self.entities_by_doc = {}
        for doc_id in self.entities_df['doc_id'].unique():
            doc_entities = self.entities_df[self.entities_df['doc_id'] == doc_id]
            self.entities_by_doc[doc_id] = doc_entities.to_dict('records')
        
        # Load policy summary if available
        policy_file = self.ner_dir / "policy_specific_entities.json"
        if policy_file.exists():
            with open(policy_file, 'r') as f:
                policy_data = json.load(f)
                self.policy_summary = policy_data.get('summary', {})
        else:
            self.policy_summary = {}
        
        print(f"✓ Loaded {len(self.entities_df):,} entities from {len(self.entities_by_doc)} documents")
    
    def create_entity_network(self, entity_type: str = 'ORG', 
                             min_cooccurrence: int = 2,
                             top_n: int = 30) -> nx.Graph:
        """
        Create network graph from entity co-occurrences.
        
        Args:
            entity_type: Type of entity to include
            min_cooccurrence: Minimum co-occurrence count
            top_n: Maximum number of entities to include
            
        Returns:
            NetworkX graph
        """
        print(f"\nCreating {entity_type} entity network...")
        
        # Get top entities
        entities_of_type = self.entities_df[self.entities_df['label'] == entity_type]
        top_entities = entities_of_type['text'].value_counts().head(top_n).index.tolist()
        
        # Build co-occurrence graph
        G = nx.Graph()
        
        # Add nodes
        for entity in top_entities:
            count = len(entities_of_type[entities_of_type['text'] == entity])
            G.add_node(entity, weight=count)
        
        # Add edges based on co-occurrence in documents
        edge_weights = {}
        
        for doc_id, entities in self.entities_by_doc.items():
            doc_entities = [e['text'] for e in entities if e['label'] == entity_type and e['text'] in top_entities]
            
            # Create edges for entities in same document
            for i, ent1 in enumerate(doc_entities):
                for ent2 in doc_entities[i+1:]:
                    edge = tuple(sorted([ent1, ent2]))
                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # Add edges with sufficient co-occurrence
        for (ent1, ent2), weight in edge_weights.items():
            if weight >= min_cooccurrence:
                G.add_edge(ent1, ent2, weight=weight)
        
        print(f"✓ Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def visualize_entity_network(self, G: nx.Graph, entity_type: str,
                                output_path: str = None):
        """
        Create interactive visualization of entity network.
        
        Args:
            G: NetworkX graph
            entity_type: Type of entities
            output_path: Path to save HTML file
        """
        print(f"\nCreating interactive {entity_type} network visualization...")
        
        # Compute layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        # Normalize edge weights for width
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 for w in edge_weights]
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            weight = node[1].get('weight', 1)
            connections = len(list(G.neighbors(node[0])))
            
            node_text.append(f"{node[0]}<br>Mentions: {weight}<br>Connections: {connections}")
            node_size.append(10 + weight * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n[:20] for n in G.nodes()],
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
                    title='Frequency',
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
                    text=f'{entity_type} Entity Network<br>Node size = mention frequency',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Network visualization saved to {output_path}")
        
        return fig
    
    def create_policy_instrument_network(self, output_path: str = None):
        """
        Create network showing relationships between policy instruments and energy carriers.
        
        Args:
            output_path: Path to save HTML file
        """
        print("\nCreating policy instrument network...")
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add policy instrument nodes
        instruments = self.policy_summary['policy_instruments']
        for instrument, count in instruments.items():
            if count >= 3:  # Minimum threshold
                G.add_node(instrument, node_type='instrument', weight=count)
        
        # Add energy carrier nodes
        carriers = self.policy_summary['energy_carriers']
        for carrier, count in carriers.items():
            if count >= 5:  # Minimum threshold
                G.add_node(carrier, node_type='carrier', weight=count)
        
        # Add edges based on co-occurrence in documents
        # (Simplified: connect if they appear in same document)
        for doc_id, entities in self.entities_by_doc.items():
            doc_text = " ".join([e['text'].lower() for e in entities])
            
            # Find instruments and carriers in this doc
            doc_instruments = [i for i in instruments.keys() if i in doc_text]
            doc_carriers = [c for c in carriers.keys() if c in doc_text]
            
            # Connect instruments to carriers
            for inst in doc_instruments:
                for carr in doc_carriers:
                    if G.has_node(inst) and G.has_node(carr):
                        if G.has_edge(inst, carr):
                            G[inst][carr]['weight'] = G[inst][carr].get('weight', 0) + 1
                        else:
                            G.add_edge(inst, carr, weight=1)
        
        # Create visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Separate nodes by type
        instrument_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'instrument']
        carrier_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'carrier']
        
        # Edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Instrument nodes
        inst_x = [pos[n][0] for n in instrument_nodes]
        inst_y = [pos[n][1] for n in instrument_nodes]
        inst_size = [G.nodes[n]['weight'] * 3 for n in instrument_nodes]
        
        inst_trace = go.Scatter(
            x=inst_x, y=inst_y,
            mode='markers+text',
            name='Policy Instruments',
            text=instrument_nodes,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=inst_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Carrier nodes
        carr_x = [pos[n][0] for n in carrier_nodes]
        carr_y = [pos[n][1] for n in carrier_nodes]
        carr_size = [G.nodes[n]['weight'] * 3 for n in carrier_nodes]
        
        carr_trace = go.Scatter(
            x=carr_x, y=carr_y,
            mode='markers+text',
            name='Energy Carriers',
            text=carrier_nodes,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=carr_size,
                color='lightcoral',
                line=dict(width=2, color='darkred')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, inst_trace, carr_trace],
            layout=go.Layout(
                title=dict(
                    text='Policy Instruments ↔ Energy Carriers Network',
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
            print(f"✓ Policy instrument network saved to {output_path}")
        
        return fig


def main():
    """Main execution function."""
    print("="*80)
    print("PHASE 3: ENTITY NETWORK VISUALIZATION")
    print("="*80)
    
    # Initialize visualizer
    visualizer = EntityNetworkVisualizer("preprocessed_output/named_entities")
    
    # Create organization network
    org_graph = visualizer.create_entity_network('ORG', min_cooccurrence=2, top_n=25)
    visualizer.visualize_entity_network(
        org_graph, 'ORG',
        output_path="preprocessed_output/named_entities/org_network.html"
    )
    
    # Create location network
    gpe_graph = visualizer.create_entity_network('GPE', min_cooccurrence=2, top_n=25)
    visualizer.visualize_entity_network(
        gpe_graph, 'GPE',
        output_path="preprocessed_output/named_entities/gpe_network.html"
    )
    
    # Create policy instrument network
    visualizer.create_policy_instrument_network(
        output_path="preprocessed_output/named_entities/policy_instrument_network.html"
    )
    
    print("\n" + "="*80)
    print("✅ Entity Network Visualizations Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - org_network.html (organization relationships)")
    print("  - gpe_network.html (location relationships)")
    print("  - policy_instrument_network.html (instruments ↔ carriers)")
    print("\nOpen these HTML files in your browser to explore!")


if __name__ == "__main__":
    main()