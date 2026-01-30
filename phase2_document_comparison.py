"""
Phase 2: Document Comparison & Coherence Analysis
==================================================
Analyze document similarities, policy coherence, and conflicts.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict

class DocumentComparisonAnalyzer:
    """
    Comprehensive document comparison and coherence analysis.
    """
    
    def __init__(self, preprocessed_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents and embeddings
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        
        # Load metadata
        print("Loading document metadata...")
        metadata_path = self.preprocessed_dir / "metadata" / "metadata_catalog.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load embeddings
        print("Loading embeddings...")
        embeddings_path = self.preprocessed_dir / "embeddings" / "embeddings.npy"
        self.chunk_embeddings = np.load(embeddings_path)
        
        with open(self.preprocessed_dir / "embeddings" / "chunk_metadata.json", 'r') as f:
            self.chunk_metadata = json.load(f)
        
        # Load document texts
        print("Loading document texts...")
        self.documents = []
        text_dir = self.preprocessed_dir / "processed_texts"
        for meta in self.metadata:
            doc_id = meta['doc_id']
            text_path = text_dir / f"{doc_id}.txt"
            
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.documents.append({
                'doc_id': doc_id,
                'text': text,
                'metadata': meta
            })
        
        print(f"✓ Loaded {len(self.documents)} documents with {len(self.chunk_embeddings)} chunks")
        
        # Create output directory
        self.comparison_dir = self.preprocessed_dir / "document_comparison"
        self.comparison_dir.mkdir(exist_ok=True)
        
    def compute_document_embeddings(self) -> np.ndarray:
        """
        Compute document-level embeddings by averaging chunk embeddings.
        
        Returns:
            Array of document embeddings (n_docs, embedding_dim)
        """
        print("\nComputing document-level embeddings...")
        
        doc_embeddings = []
        
        for doc in self.documents:
            doc_id = doc['doc_id']
            
            # Find all chunks for this document
            chunk_indices = [i for i, meta in enumerate(self.chunk_metadata) 
                           if meta['doc_id'] == doc_id]
            
            if chunk_indices:
                # Average the chunk embeddings
                doc_embedding = np.mean(self.chunk_embeddings[chunk_indices], axis=0)
                doc_embeddings.append(doc_embedding)
            else:
                print(f"  ⚠️  No chunks found for {doc_id}")
                doc_embeddings.append(np.zeros(self.chunk_embeddings.shape[1]))
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Normalize embeddings for cosine similarity
        doc_embeddings = normalize(doc_embeddings, axis=1)
        
        print(f"✓ Computed embeddings for {len(doc_embeddings)} documents")
        
        return doc_embeddings
    
    def compute_similarity_matrix(self, doc_embeddings: np.ndarray) -> pd.DataFrame:
        """
        Compute pairwise similarity matrix for all documents.
        
        Args:
            doc_embeddings: Document embeddings array
            
        Returns:
            DataFrame with similarity scores
        """
        print("\nComputing similarity matrix...")
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(doc_embeddings)
        
        # Create DataFrame with document names
        doc_names = [doc['metadata']['filename'] for doc in self.documents]
        
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=doc_names,
            columns=doc_names
        )
        
        print(f"✓ Similarity matrix computed: {similarity_matrix.shape}")
        
        return similarity_df
    
    def find_most_similar_pairs(self, similarity_df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """
        Find the most similar document pairs.
        
        Args:
            similarity_df: Similarity matrix
            top_n: Number of top pairs to return
            
        Returns:
            List of similar document pairs with scores
        """
        print(f"\nFinding top {top_n} most similar document pairs...")
        
        # Get upper triangle (avoid duplicates and self-similarity)
        similar_pairs = []
        
        for i in range(len(similarity_df)):
            for j in range(i + 1, len(similarity_df)):
                similar_pairs.append({
                    'doc1': similarity_df.index[i],
                    'doc2': similarity_df.index[j],
                    'similarity': similarity_df.iloc[i, j],
                    'doc1_id': self.documents[i]['doc_id'],
                    'doc2_id': self.documents[j]['doc_id'],
                    'doc1_type': self.documents[i]['metadata']['document_type'],
                    'doc2_type': self.documents[j]['metadata']['document_type'],
                    'doc1_year': self.documents[i]['metadata']['year'],
                    'doc2_year': self.documents[j]['metadata']['year']
                })
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_pairs[:top_n]
    
    def find_least_similar_pairs(self, similarity_df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """
        Find the least similar (most divergent) document pairs.
        
        Args:
            similarity_df: Similarity matrix
            top_n: Number of bottom pairs to return
            
        Returns:
            List of divergent document pairs with scores
        """
        print(f"\nFinding top {top_n} most divergent document pairs...")
        
        divergent_pairs = []
        
        for i in range(len(similarity_df)):
            for j in range(i + 1, len(similarity_df)):
                divergent_pairs.append({
                    'doc1': similarity_df.index[i],
                    'doc2': similarity_df.index[j],
                    'similarity': similarity_df.iloc[i, j],
                    'doc1_id': self.documents[i]['doc_id'],
                    'doc2_id': self.documents[j]['doc_id'],
                    'doc1_type': self.documents[i]['metadata']['document_type'],
                    'doc2_type': self.documents[j]['metadata']['document_type']
                })
        
        # Sort by similarity (ascending for divergence)
        divergent_pairs.sort(key=lambda x: x['similarity'])
        
        return divergent_pairs[:top_n]
    
    def analyze_coherence_by_type(self, similarity_df: pd.DataFrame) -> Dict:
        """
        Analyze policy coherence within and across document types.
        
        Args:
            similarity_df: Similarity matrix
            
        Returns:
            Dictionary with coherence statistics
        """
        print("\nAnalyzing coherence by document type...")
        
        # Group documents by type
        doc_types = defaultdict(list)
        for i, doc in enumerate(self.documents):
            doc_types[doc['metadata']['document_type']].append(i)
        
        coherence_analysis = {
            'within_type': {},
            'across_type': {}
        }
        
        # Within-type coherence
        for doc_type, indices in doc_types.items():
            if len(indices) < 2:
                continue
            
            similarities = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    similarities.append(similarity_df.iloc[idx1, idx2])
            
            coherence_analysis['within_type'][doc_type] = {
                'count': len(indices),
                'avg_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities))
            }
        
        # Across-type coherence
        type_pairs = []
        for type1 in doc_types.keys():
            for type2 in doc_types.keys():
                if type1 < type2:  # Avoid duplicates
                    similarities = []
                    for idx1 in doc_types[type1]:
                        for idx2 in doc_types[type2]:
                            similarities.append(similarity_df.iloc[idx1, idx2])
                    
                    if similarities:
                        coherence_analysis['across_type'][f"{type1}-{type2}"] = {
                            'avg_similarity': float(np.mean(similarities)),
                            'std_similarity': float(np.std(similarities))
                        }
        
        return coherence_analysis
    
    def analyze_temporal_evolution(self, similarity_df: pd.DataFrame) -> Dict:
        """
        Analyze how policies evolve over time.
        
        Args:
            similarity_df: Similarity matrix
            
        Returns:
            Dictionary with temporal analysis
        """
        print("\nAnalyzing temporal policy evolution...")
        
        # Group documents by year
        year_groups = defaultdict(list)
        for i, doc in enumerate(self.documents):
            year = doc['metadata']['year']
            if year != 'unknown':
                year_groups[year].append(i)
        
        temporal_analysis = {
            'year_statistics': {},
            'year_transitions': []
        }
        
        # Statistics per year
        for year, indices in sorted(year_groups.items()):
            if len(indices) > 1:
                similarities = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        similarities.append(similarity_df.iloc[idx1, idx2])
                
                temporal_analysis['year_statistics'][year] = {
                    'count': len(indices),
                    'avg_coherence': float(np.mean(similarities)),
                    'documents': [self.documents[i]['metadata']['filename'] for i in indices]
                }
        
        # Year-to-year transitions
        sorted_years = sorted([y for y in year_groups.keys() if y != 'unknown'])
        for i in range(len(sorted_years) - 1):
            year1, year2 = sorted_years[i], sorted_years[i + 1]
            
            similarities = []
            for idx1 in year_groups[year1]:
                for idx2 in year_groups[year2]:
                    similarities.append(similarity_df.iloc[idx1, idx2])
            
            if similarities:
                temporal_analysis['year_transitions'].append({
                    'from_year': year1,
                    'to_year': year2,
                    'avg_similarity': float(np.mean(similarities)),
                    'interpretation': 'High continuity' if np.mean(similarities) > 0.6 
                                    else 'Moderate change' if np.mean(similarities) > 0.4 
                                    else 'Significant shift'
                })
        
        return temporal_analysis
    
    def detect_potential_conflicts(self, similarity_df: pd.DataFrame, threshold: float = 0.3) -> List[Dict]:
        """
        Detect potential policy conflicts based on low similarity.
        
        Args:
            similarity_df: Similarity matrix
            threshold: Similarity threshold below which documents are considered potentially conflicting
            
        Returns:
            List of potential conflicts
        """
        print(f"\nDetecting potential conflicts (threshold: {threshold})...")
        
        conflicts = []
        
        for i in range(len(similarity_df)):
            for j in range(i + 1, len(similarity_df)):
                similarity = similarity_df.iloc[i, j]
                
                if similarity < threshold:
                    conflicts.append({
                        'doc1': similarity_df.index[i],
                        'doc2': similarity_df.index[j],
                        'similarity': float(similarity),
                        'doc1_type': self.documents[i]['metadata']['document_type'],
                        'doc2_type': self.documents[j]['metadata']['document_type'],
                        'doc1_year': self.documents[i]['metadata']['year'],
                        'doc2_year': self.documents[j]['metadata']['year'],
                        'severity': 'High' if similarity < 0.2 else 'Moderate'
                    })
        
        conflicts.sort(key=lambda x: x['similarity'])
        
        print(f"✓ Found {len(conflicts)} potential conflicts")
        
        return conflicts
    
    def visualize_similarity_heatmap(self, similarity_df: pd.DataFrame, output_path: str = None):
        """
        Create a heatmap visualization of document similarities.
        
        Args:
            similarity_df: Similarity matrix
            output_path: Path to save the figure
        """
        print("\nGenerating similarity heatmap...")
        
        plt.figure(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
        
        # Create heatmap
        sns.heatmap(
            similarity_df,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        
        plt.title('Document Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Documents', fontsize=12)
        plt.ylabel('Documents', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {output_path}")
        
        plt.close()
    
    def visualize_hierarchical_clustering(self, similarity_df: pd.DataFrame, output_path: str = None):
        """
        Create hierarchical clustering dendrogram.
        
        Args:
            similarity_df: Similarity matrix
            output_path: Path to save the figure
        """
        print("\nGenerating hierarchical clustering dendrogram...")
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_df.values
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        plt.figure(figsize=(14, 8))
        
        dendrogram(
            linkage_matrix,
            labels=similarity_df.index,
            leaf_rotation=90,
            leaf_font_size=9
        )
        
        plt.title('Hierarchical Clustering of Policy Documents', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Documents', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dendrogram saved to {output_path}")
        
        plt.close()
    
    def generate_comparison_report(self, similarity_df: pd.DataFrame, 
                                   similar_pairs: List[Dict],
                                   divergent_pairs: List[Dict],
                                   coherence_analysis: Dict,
                                   temporal_analysis: Dict,
                                   conflicts: List[Dict]) -> Dict:
        """
        Generate comprehensive comparison report.
        """
        print("\nGenerating comparison report...")
        
        report = {
            'summary': {
                'total_documents': len(self.documents),
                'avg_similarity': float(np.mean(similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)])),
                'std_similarity': float(np.std(similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)])),
                'most_similar_score': similar_pairs[0]['similarity'] if similar_pairs else 0,
                'least_similar_score': divergent_pairs[0]['similarity'] if divergent_pairs else 0,
                'potential_conflicts': len(conflicts)
            },
            'top_similar_pairs': similar_pairs[:10],
            'top_divergent_pairs': divergent_pairs[:10],
            'coherence_analysis': coherence_analysis,
            'temporal_analysis': temporal_analysis,
            'potential_conflicts': conflicts[:20]
        }
        
        return report
    
    def save_results(self, similarity_df: pd.DataFrame, report: Dict):
        """Save all analysis results."""
        print("\nSaving results...")
        
        # Save similarity matrix
        similarity_df.to_csv(self.comparison_dir / "similarity_matrix.csv")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Save report with type conversion
        serializable_report = convert_to_serializable(report)
        with open(self.comparison_dir / "comparison_report.json", 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        # Save readable summary
        with open(self.comparison_dir / "comparison_summary.txt", 'w') as f:
            f.write("DOCUMENT COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Documents: {report['summary']['total_documents']}\n")
            f.write(f"Average Similarity: {report['summary']['avg_similarity']:.4f}\n")
            f.write(f"Std Deviation: {report['summary']['std_similarity']:.4f}\n")
            f.write(f"Potential Conflicts: {report['summary']['potential_conflicts']}\n\n")
            
            f.write("TOP 10 MOST SIMILAR DOCUMENT PAIRS\n")
            f.write("-" * 80 + "\n")
            for i, pair in enumerate(report['top_similar_pairs'][:10], 1):
                f.write(f"{i}. {pair['doc1']} <-> {pair['doc2']}\n")
                f.write(f"   Similarity: {pair['similarity']:.4f}\n")
                f.write(f"   Types: {pair['doc1_type']} & {pair['doc2_type']}\n\n")
            
            f.write("\nTOP 10 MOST DIVERGENT DOCUMENT PAIRS\n")
            f.write("-" * 80 + "\n")
            for i, pair in enumerate(report['top_divergent_pairs'][:10], 1):
                f.write(f"{i}. {pair['doc1']} <-> {pair['doc2']}\n")
                f.write(f"   Similarity: {pair['similarity']:.4f}\n")
                f.write(f"   Types: {pair['doc1_type']} & {pair['doc2_type']}\n\n")
        
        print(f"✓ Results saved to {self.comparison_dir}")


if __name__ == "__main__":

    """Main execution function."""
    print("="*80)
    print("PHASE 2: DOCUMENT COMPARISON & COHERENCE ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = DocumentComparisonAnalyzer("preprocessed_output")
    
    # Compute document embeddings
    doc_embeddings = analyzer.compute_document_embeddings()
    
    # Compute similarity matrix
    similarity_df = analyzer.compute_similarity_matrix(doc_embeddings)
    
    # Find similar and divergent pairs
    similar_pairs = analyzer.find_most_similar_pairs(similarity_df, top_n=10)
    divergent_pairs = analyzer.find_least_similar_pairs(similarity_df, top_n=10)
    
    # Analyze coherence
    coherence_analysis = analyzer.analyze_coherence_by_type(similarity_df)
    
    # Analyze temporal evolution
    temporal_analysis = analyzer.analyze_temporal_evolution(similarity_df)
    
    # Detect conflicts
    conflicts = analyzer.detect_potential_conflicts(similarity_df, threshold=0.3)
    
    # Generate visualizations
    analyzer.visualize_similarity_heatmap(
        similarity_df,
        output_path=analyzer.comparison_dir / "similarity_heatmap.png"
    )
    
    analyzer.visualize_hierarchical_clustering(
        similarity_df,
        output_path=analyzer.comparison_dir / "clustering_dendrogram.png"
    )
    
    # Generate and save report
    report = analyzer.generate_comparison_report(
        similarity_df,
        similar_pairs,
        divergent_pairs,
        coherence_analysis,
        temporal_analysis,
        conflicts
    )
    
    analyzer.save_results(similarity_df, report)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Average Document Similarity: {report['summary']['avg_similarity']:.4f}")
    print(f"  Most Similar Pair: {similar_pairs[0]['similarity']:.4f}")
    print(f"  Least Similar Pair: {divergent_pairs[0]['similarity']:.4f}")
    print(f"  Potential Conflicts Detected: {len(conflicts)}")
    
    print(f"\n🔗 TOP 5 MOST SIMILAR PAIRS")
    for i, pair in enumerate(similar_pairs[:5], 1):
        print(f"  {i}. {pair['doc1'][:40]}... <-> {pair['doc2'][:40]}...")
        print(f"     Similarity: {pair['similarity']:.4f} | Types: {pair['doc1_type']} & {pair['doc2_type']}")
    
    print(f"\n⚠️  TOP 5 MOST DIVERGENT PAIRS")
    for i, pair in enumerate(divergent_pairs[:5], 1):
        print(f"  {i}. {pair['doc1'][:40]}... <-> {pair['doc2'][:40]}...")
        print(f"     Similarity: {pair['similarity']:.4f} | Types: {pair['doc1_type']} & {pair['doc2_type']}")
    
    print(f"\n📈 COHERENCE BY DOCUMENT TYPE")
    for doc_type, stats in coherence_analysis['within_type'].items():
        print(f"  {doc_type.capitalize()}: {stats['avg_similarity']:.4f} avg similarity ({stats['count']} docs)")
    
    print("\n" + "="*80)
    print("✅ Phase 2 Complete!")
    print("="*80)
    print("\nOutputs saved to: preprocessed_output/document_comparison/")
    print("  - similarity_matrix.csv")
    print("  - comparison_report.json")
    print("  - comparison_summary.txt")
    print("  - similarity_heatmap.png")
    print("  - clustering_dendrogram.png")
    print("\nNext: Phase 3 - Named Entity Recognition")

