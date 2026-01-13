"""
Phase 4: Topic Modeling with BERTopic
======================================
Discover latent topics in policy documents using BERTopic.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PolicyTopicModeler:
    """
    Topic modeling for policy documents using BERTopic.
    """
    
    def __init__(self, preprocessed_dir: str):
        """
        Initialize topic modeler.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        
        # Load metadata
        print("Loading document metadata...")
        metadata_path = self.preprocessed_dir / "metadata" / "metadata_catalog.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load documents
        print("Loading document texts...")
        self.documents = []
        self.doc_info = []
        text_dir = self.preprocessed_dir / "processed_texts"
        
        for meta in self.metadata:
            doc_id = meta['doc_id']
            text_path = text_dir / f"{doc_id}.txt"
            
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.documents.append(text)
            self.doc_info.append({
                'doc_id': doc_id,
                'filename': meta['filename'],
                'doc_type': meta['document_type'],
                'year': meta['year']
            })
        
        print(f"✓ Loaded {len(self.documents)} documents")
        
        # Create output directory
        self.topic_dir = self.preprocessed_dir / "topic_modeling"
        self.topic_dir.mkdir(exist_ok=True)
        
        self.topic_model = None
        self.topics = None
        self.probs = None
        
    def create_topic_model(self, 
                          n_topics: int = None,
                          min_topic_size: int = 2,
                          embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Create and fit BERTopic model.
        
        Args:
            n_topics: Number of topics (None for automatic)
            min_topic_size: Minimum size for a topic
            embedding_model: Sentence transformer model
        """
        print("\n" + "="*80)
        print("CREATING TOPIC MODEL")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Number of topics: {'Automatic' if n_topics is None else n_topics}")
        print(f"  Min topic size: {min_topic_size}")
        
        # Initialize embedding model
        print("\nInitializing embedding model...")
        sentence_model = SentenceTransformer(embedding_model)
        
        # Initialize UMAP for dimensionality reduction
        print("Configuring UMAP...")
        umap_model = UMAP(
            n_neighbors=3,  # Small corpus, small neighbors
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Initialize HDBSCAN for clustering
        print("Configuring HDBSCAN...")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize CountVectorizer for topic representation
        vectorizer_model = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Create BERTopic model
        print("\nCreating BERTopic model...")
        self.topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            n_gram_range=(1, 3),
            calculate_probabilities=True,
            verbose=True
        )
        
        # Fit model
        print("\nFitting topic model...")
        print("This may take a few minutes...")
        
        self.topics, self.probs = self.topic_model.fit_transform(self.documents)
        
        n_topics_found = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        print(f"\n✓ Topic modeling complete!")
        print(f"  Topics discovered: {n_topics_found}")
        print(f"  Outliers: {sum(1 for t in self.topics if t == -1)}")
        
        return self.topic_model
    
    def get_topic_info(self) -> pd.DataFrame:
        """
        Get detailed information about discovered topics.
        
        Returns:
            DataFrame with topic information
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created yet. Run create_topic_model first.")
        
        topic_info = self.topic_model.get_topic_info()
        
        # Add custom columns
        topic_info['topic_label'] = topic_info.apply(
            lambda row: f"Topic {row['Topic']}: {row['Name'][:50]}" if row['Topic'] != -1 else "Outliers",
            axis=1
        )
        
        return topic_info
    
    def get_document_topics(self) -> pd.DataFrame:
        """
        Get topic assignments for each document.
        
        Returns:
            DataFrame with document-topic mappings
        """
        if self.topics is None:
            raise ValueError("Topics not assigned yet. Run create_topic_model first.")
        
        doc_topics = []
        
        for i, (doc_id, topic, prob) in enumerate(zip(
            [d['doc_id'] for d in self.doc_info],
            self.topics,
            self.probs
        )):
            doc_topics.append({
                'doc_id': doc_id,
                'filename': self.doc_info[i]['filename'],
                'doc_type': self.doc_info[i]['doc_type'],
                'year': self.doc_info[i]['year'],
                'topic': int(topic),
                'probability': float(prob[topic] if topic != -1 else 0)
            })
        
        return pd.DataFrame(doc_topics)
    
    def get_representative_documents(self, topic_id: int, n_docs: int = 3) -> List[Dict]:
        """
        Get most representative documents for a topic.
        
        Args:
            topic_id: Topic ID
            n_docs: Number of documents to return
            
        Returns:
            List of representative documents
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created yet.")
        
        # Get documents for this topic
        topic_docs = [i for i, t in enumerate(self.topics) if t == topic_id]
        
        if not topic_docs:
            return []
        
        # Get probabilities for these documents
        doc_probs = [(i, self.probs[i][topic_id]) for i in topic_docs]
        doc_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        representatives = []
        for idx, prob in doc_probs[:n_docs]:
            representatives.append({
                'doc_id': self.doc_info[idx]['doc_id'],
                'filename': self.doc_info[idx]['filename'],
                'doc_type': self.doc_info[idx]['doc_type'],
                'year': self.doc_info[idx]['year'],
                'probability': float(prob),
                'text_preview': self.documents[idx][:500]
            })
        
        return representatives
    
    def analyze_topics_by_type(self) -> pd.DataFrame:
        """
        Analyze topic distribution across document types.
        
        Returns:
            DataFrame with topic-type distribution
        """
        doc_topics_df = self.get_document_topics()
        
        # Create cross-tabulation
        topic_type_dist = pd.crosstab(
            doc_topics_df['topic'],
            doc_topics_df['doc_type'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        return topic_type_dist
    
    def analyze_topics_over_time(self) -> pd.DataFrame:
        """
        Analyze topic evolution over time.
        
        Returns:
            DataFrame with topic-year distribution
        """
        doc_topics_df = self.get_document_topics()
        
        # Filter out unknown years
        doc_topics_df = doc_topics_df[doc_topics_df['year'] != 'unknown']
        
        # Convert year to numeric
        doc_topics_df['year_num'] = pd.to_numeric(doc_topics_df['year'], errors='coerce')
        doc_topics_df = doc_topics_df.dropna(subset=['year_num'])
        
        # Create cross-tabulation
        topic_time_dist = pd.crosstab(
            doc_topics_df['topic'],
            doc_topics_df['year_num']
        )
        
        return topic_time_dist
    
    def visualize_topics(self, output_path: str = None):
        """
        Create topic visualization.
        
        Args:
            output_path: Path to save figure
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created yet.")
        
        print("\nGenerating topic visualization...")
        
        # Use BERTopic's built-in visualization
        fig = self.topic_model.visualize_topics()
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Topic visualization saved to {output_path}")
        
        return fig
    
    def visualize_topic_hierarchy(self, output_path: str = None):
        """
        Create hierarchical topic visualization.
        
        Args:
            output_path: Path to save figure
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created yet.")
        
        print("\nGenerating topic hierarchy...")
        
        # Use BERTopic's hierarchical visualization
        fig = self.topic_model.visualize_hierarchy()
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Topic hierarchy saved to {output_path}")
        
        return fig
    
    def visualize_barchart(self, top_n: int = 8, output_path: str = None):
        """
        Create topic barchart visualization.
        
        Args:
            top_n: Number of topics to show
            output_path: Path to save figure
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created yet.")
        
        print(f"\nGenerating barchart for top {top_n} topics...")
        
        fig = self.topic_model.visualize_barchart(top_n_topics=top_n)
        
        if output_path:
            fig.write_html(output_path)
            print(f"✓ Barchart saved to {output_path}")
        
        return fig
    
    def visualize_topic_distribution(self, output_path: str = None):
        """
        Visualize topic distribution across documents.
        
        Args:
            output_path: Path to save figure
        """
        print("\nGenerating topic distribution visualization...")
        
        topic_counts = pd.Series(self.topics).value_counts().sort_index()
        
        # Remove outlier topic (-1) for cleaner visualization
        if -1 in topic_counts.index:
            outliers = topic_counts[-1]
            topic_counts = topic_counts[topic_counts.index != -1]
        else:
            outliers = 0
        
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(topic_counts)))
        bars = plt.bar(range(len(topic_counts)), topic_counts.values, color=colors)
        
        plt.xlabel('Topic ID', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title('Topic Distribution Across Documents', fontsize=14, fontweight='bold')
        plt.xticks(range(len(topic_counts)), topic_counts.index)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        # Add outliers note
        if outliers > 0:
            plt.text(0.02, 0.98, f'Outliers: {outliers}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution saved to {output_path}")
        
        plt.close()
    
    def save_results(self, topic_info: pd.DataFrame, doc_topics: pd.DataFrame):
        """
        Save all topic modeling results.
        
        Args:
            topic_info: Topic information DataFrame
            doc_topics: Document-topic assignments DataFrame
        """
        print("\nSaving topic modeling results...")
        
        # Save topic information
        topic_info.to_csv(self.topic_dir / "topic_info.csv", index=False)
        
        # Save document-topic assignments
        doc_topics.to_csv(self.topic_dir / "document_topics.csv", index=False)
        
        # Save detailed topic representations
        topic_representations = {}
        for topic_id in topic_info['Topic'].unique():
            if topic_id == -1:
                continue
            
            topic_words = self.topic_model.get_topic(topic_id)
            representatives = self.get_representative_documents(topic_id, n_docs=3)
            
            topic_representations[int(topic_id)] = {
                'topic_words': topic_words,
                'representative_documents': representatives
            }
        
        with open(self.topic_dir / "topic_representations.json", 'w') as f:
            json.dump(topic_representations, f, indent=2)
        
        # Save model
        self.topic_model.save(str(self.topic_dir / "bertopic_model"))
        
        print(f"✓ Results saved to {self.topic_dir}")


if __name__ == "__main__":
    """Main execution function."""
    print("="*80)
    print("PHASE 4: TOPIC MODELING WITH BERTOPIC")
    print("="*80)
    
    # Initialize modeler
    modeler = PolicyTopicModeler("preprocessed_output")
    
    # Create topic model
    topic_model = modeler.create_topic_model(
        n_topics=None,  # Automatic topic discovery
        min_topic_size=2,  # Minimum 2 documents per topic
        embedding_model='all-MiniLM-L6-v2'
    )
    
    # Get topic information
    topic_info = modeler.get_topic_info()
    doc_topics = modeler.get_document_topics()
    
    # Analyze topics
    print("\n" + "="*80)
    print("TOPIC ANALYSIS")
    print("="*80)
    
    print("\n📊 Topics Discovered:")
    for _, row in topic_info.iterrows():
        if row['Topic'] == -1:
            continue
        print(f"\nTopic {row['Topic']}: {row['Name']}")
        print(f"  Documents: {row['Count']}")
        print(f"  Top words: {', '.join([w for w, _ in topic_model.get_topic(row['Topic'])[:5]])}")
    
    # Get representative documents for each topic
    print("\n📄 Representative Documents per Topic:")
    for topic_id in topic_info['Topic'].unique():
        if topic_id == -1:
            continue
        
        print(f"\nTopic {topic_id}:")
        representatives = modeler.get_representative_documents(topic_id, n_docs=2)
        for i, doc in enumerate(representatives, 1):
            print(f"  {i}. {doc['filename']} (prob: {doc['probability']:.3f})")
    
    # Analyze by document type
    print("\n📁 Topic Distribution by Document Type:")
    topic_type_dist = modeler.analyze_topics_by_type()
    print(topic_type_dist.round(1))
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    modeler.visualize_topics(
        output_path=modeler.topic_dir / "topic_visualization.html"
    )
    
    modeler.visualize_barchart(
        top_n=8,
        output_path=modeler.topic_dir / "topic_barchart.html"
    )
    
    modeler.visualize_topic_hierarchy(
        output_path=modeler.topic_dir / "topic_hierarchy.html"
    )
    
    modeler.visualize_topic_distribution(
        output_path=modeler.topic_dir / "topic_distribution.png"
    )
    
    # Save results
    modeler.save_results(topic_info, doc_topics)
    
    print("\n" + "="*80)
    print("✅ Phase 4 Complete!")
    print("="*80)
    print("\nOutputs saved to: preprocessed_output/topic_modeling/")
    print("  - topic_info.csv")
    print("  - document_topics.csv")
    print("  - topic_representations.json")
    print("  - bertopic_model/ (saved model)")
    print("  - topic_visualization.html")
    print("  - topic_barchart.html")
    print("  - topic_hierarchy.html")
    print("  - topic_distribution.png")
    print("\nNext: Phase 5 - Dashboard")

