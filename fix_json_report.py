"""
Quick Fix: Regenerate comparison_report.json with proper type conversion
========================================================================
Run this to fix the corrupted JSON file.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def regenerate_json_report(comparison_dir: str):
    """Regenerate the JSON report from existing data."""
    comparison_dir = Path(comparison_dir)
    
    print("Loading similarity matrix...")
    similarity_df = pd.read_csv(comparison_dir / "similarity_matrix.csv", index_col=0)
    
    print("Recreating report structure...")
    
    # Get document names
    doc_names = list(similarity_df.index)
    
    # Create similar pairs
    similar_pairs = []
    for i in range(len(similarity_df)):
        for j in range(i + 1, len(similarity_df)):
            similar_pairs.append({
                'doc1': doc_names[i],
                'doc2': doc_names[j],
                'similarity': float(similarity_df.iloc[i, j]),
                'doc1_type': 'unknown',  # We don't have this info anymore
                'doc2_type': 'unknown'
            })
    
    # Sort and get top similar
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    top_similar = similar_pairs[:30]
    
    # Get divergent pairs (bottom of the list)
    divergent_pairs = similar_pairs[-30:]
    divergent_pairs.reverse()
    
    # Create conflicts
    conflicts = [p for p in similar_pairs if p['similarity'] < 0.3]
    
    # Calculate summary statistics
    similarities = [p['similarity'] for p in similar_pairs]
    
    report = {
        'summary': {
            'total_documents': len(doc_names),
            'avg_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'most_similar_score': float(top_similar[0]['similarity']) if top_similar else 0.0,
            'least_similar_score': float(divergent_pairs[0]['similarity']) if divergent_pairs else 0.0,
            'potential_conflicts': len(conflicts)
        },
        'top_similar_pairs': top_similar,
        'top_divergent_pairs': divergent_pairs,
        'potential_conflicts': conflicts[:20],
        'coherence_analysis': {
            'within_type': {},
            'across_type': {}
        },
        'temporal_analysis': {
            'year_statistics': {},
            'year_transitions': []
        }
    }
    
    # Convert to serializable
    report = convert_to_serializable(report)
    
    # Save
    output_path = comparison_dir / "comparison_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Successfully regenerated {output_path}")
    print(f"  Total documents: {report['summary']['total_documents']}")
    print(f"  Avg similarity: {report['summary']['avg_similarity']:.4f}")
    print(f"  Top similarity: {report['summary']['most_similar_score']:.4f}")
    print(f"  Conflicts found: {report['summary']['potential_conflicts']}")

if __name__ == "__main__":
    regenerate_json_report("preprocessed_output/document_comparison")
    print("\n✅ JSON report fixed! You can now run the network visualizer.")