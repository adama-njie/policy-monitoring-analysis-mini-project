"""
Phase 3: Regenerate Visualizations with Cleaned Data
=====================================================
Regenerate all visualizations using cleaned entities.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data(ner_dir):
    """Load cleaned entity data - prefer ULTIMATE, then DEEP, then regular CLEANED."""
    ner_dir = Path(ner_dir)
    
    print("Loading cleaned entity data...")
    
    # Try in order of preference
    ultimate_path = ner_dir / "all_entities_ULTIMATE_CLEANED.csv"
    deep_path = ner_dir / "all_entities_DEEP_CLEANED.csv"
    cleaned_path = ner_dir / "all_entities_CLEANED.csv"
    
    if ultimate_path.exists():
        print("  Using ULTIMATE_CLEANED data")
        entities_df = pd.read_csv(ultimate_path)
        stats_file = "entity_statistics_ULTIMATE_CLEANED.json"
    elif deep_path.exists():
        print("  Using DEEP_CLEANED data")
        entities_df = pd.read_csv(deep_path)
        stats_file = "entity_statistics_DEEP_CLEANED.json"
    elif cleaned_path.exists():
        print("  Using CLEANED data")
        entities_df = pd.read_csv(cleaned_path)
        stats_file = "entity_statistics_CLEANED.json"
    else:
        print("  ⚠️  No cleaned data found! Run cleaning first.")
        return None, None, None
    
    # Load statistics
    with open(ner_dir / stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"✓ Loaded {len(entities_df):,} cleaned entities")
    return entities_df, stats, ner_dir

def visualize_entity_distribution(stats, output_path):
    """Visualize entity type distribution (cleaned)."""
    print("\nGenerating cleaned entity distribution...")
    
    entity_types = stats['entity_types']
    sorted_types = dict(sorted(entity_types.items(), key=lambda x: x[-1], reverse=True))
    
    plt.figure(figsize=(14, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_types)))
    bars = plt.bar(range(len(sorted_types)), sorted_types.values(), color=colors)
    
    plt.xlabel('Entity Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Named Entity Types (Cleaned Data)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_types)), sorted_types.keys(), rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

def visualize_top_entities(stats, entity_type, top_n, output_path):
    """Visualize top N entities of a specific type."""
    print(f"\nGenerating top {top_n} {entity_type} (cleaned)...")
    
    top_entities = stats['top_entities_by_type'].get(entity_type, [])[:top_n]
    
    if not top_entities:
        print(f"No entities of type {entity_type} found")
        return
    
    entities, counts = zip(*top_entities)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(entities)))
    bars = plt.barh(range(len(entities)), counts, color=colors)
    
    plt.yticks(range(len(entities)), entities)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Entity', fontsize=12)
    plt.title(f'Top {top_n} {entity_type} Entities (Cleaned)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

def build_cooccurrence_matrix(entities_df, entity_type, top_n):
    """Build co-occurrence matrix for cleaned entities."""
    print(f"\nBuilding co-occurrence matrix for {entity_type}...")
    
    # Get top entities
    entities_of_type = entities_df[entities_df['label'] == entity_type]
    top_entities = entities_of_type['text'].value_counts().head(top_n).index.tolist()
    
    # Group by document
    doc_entities = entities_df[entities_df['label'] == entity_type].groupby('doc_id')['text'].apply(list)
    
    # Build co-occurrence matrix
    cooccurrence = np.zeros((len(top_entities), len(top_entities)))
    
    for doc_id, entities in doc_entities.items():
        # Filter to top entities
        entities = [e for e in entities if e in top_entities]
        
        # Count co-occurrences
        for i, ent1 in enumerate(top_entities):
            for j, ent2 in enumerate(top_entities):
                if ent1 in entities and ent2 in entities:
                    cooccurrence[i, j] += 1
    
    # Create DataFrame
    cooccurrence_df = pd.DataFrame(
        cooccurrence,
        index=top_entities,
        columns=top_entities
    )
    
    return cooccurrence_df

def visualize_cooccurrence_heatmap(cooccurrence_df, entity_type, output_path):
    """Visualize entity co-occurrence as a heatmap."""
    print(f"\nGenerating co-occurrence heatmap for {entity_type}...")
    
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(
        cooccurrence_df,
        annot=False,
        cmap='YlOrRd',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Co-occurrence Count'}
    )
    
    plt.title(f'{entity_type} Co-occurrence Matrix (Cleaned Data)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Entities', fontsize=12)
    plt.ylabel('Entities', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

def main():
    """Main execution function."""
    print("="*80)
    print("PHASE 3: REGENERATE VISUALIZATIONS WITH CLEANED DATA")
    print("="*80)
    
    # Load cleaned data
    entities_df, stats, ner_dir = load_cleaned_data("preprocessed_output/named_entities")
    
    # Generate all visualizations
    visualize_entity_distribution(
        stats,
        ner_dir / "entity_distribution_CLEANED.png"
    )
    
    visualize_top_entities(
        stats, 'ORG', 15,
        ner_dir / "top_organizations_CLEANED.png"
    )
    
    visualize_top_entities(
        stats, 'GPE', 15,
        ner_dir / "top_locations_CLEANED.png"
    )
    
    # Build and visualize co-occurrence matrices
    org_cooccurrence = build_cooccurrence_matrix(entities_df, 'ORG', 15)
    visualize_cooccurrence_heatmap(
        org_cooccurrence, 'ORG',
        ner_dir / "org_cooccurrence_CLEANED.png"
    )
    
    gpe_cooccurrence = build_cooccurrence_matrix(entities_df, 'GPE', 15)
    visualize_cooccurrence_heatmap(
        gpe_cooccurrence, 'GPE',
        ner_dir / "gpe_cooccurrence_CLEANED.png"
    )
    
    print("\n" + "="*80)
    print("✅ Visualizations Regenerated!")
    print("="*80)
    print("\nCleaned visualization files:")
    print("  - entity_distribution_CLEANED.png")
    print("  - top_organizations_CLEANED.png")
    print("  - top_locations_CLEANED.png")
    print("  - org_cooccurrence_CLEANED.png")
    print("  - gpe_cooccurrence_CLEANED.png")
    print("\nTo regenerate network visualizations:")
    print("  python phase3_entity_network.py")
    print("  (Networks will automatically use cleaned data)")


if __name__ == "__main__":
    main()