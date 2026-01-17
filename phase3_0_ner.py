"""
Phase 3: Named Entity Recognition (NER)
========================================
Extract and analyze entities from policy documents.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PolicyEntityExtractor:
    """
    Extract and analyze named entities from policy documents.
    """
    
    def __init__(self, preprocessed_dir: str, model_name: str = 'en_core_web_sm'):
        """
        Initialize the entity extractor.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
            model_name: spaCy model to use
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        
        # Load spaCy model
        print(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        print(f"✓ Model loaded with {len(self.nlp.pipe_names)} pipeline components")
        
        # Load metadata
        print("Loading document metadata...")
        metadata_path = self.preprocessed_dir / "metadata" / "metadata_catalog.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load documents
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
                'filename': meta['filename'],
                'text': text,
                'metadata': meta
            })
        
        print(f"✓ Loaded {len(self.documents)} documents")
        
        # Create output directory
        self.ner_dir = self.preprocessed_dir / "named_entities"
        self.ner_dir.mkdir(exist_ok=True)
        
        # Storage for extracted entities
        self.all_entities = []
        self.entity_by_doc = {}
        
    def extract_entities(self, batch_size: int = 5):
        """
        Extract entities from all documents.
        
        Args:
            batch_size: Number of documents to process at once
        """
        print(f"\nExtracting entities from {len(self.documents)} documents...")
        
        for doc_info in tqdm(self.documents, desc="Processing documents"):
            doc_id = doc_info['doc_id']
            text = doc_info['text']
            
            # Process with spaCy
            doc = self.nlp(text[:1000000])  # Limit to 1M chars for memory
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'doc_id': doc_id,
                    'filename': doc_info['filename']
                })
            
            self.entity_by_doc[doc_id] = entities
            self.all_entities.extend(entities)
        
        print(f"✓ Extracted {len(self.all_entities)} entities total")
    
    def get_entity_statistics(self) -> Dict:
        """
        Calculate statistics about extracted entities.
        
        Returns:
            Dictionary with entity statistics
        """
        print("\nCalculating entity statistics...")
        
        # Count by type
        entity_types = Counter(ent['label'] for ent in self.all_entities)
        
        # Count unique entities by type
        unique_by_type = defaultdict(set)
        for ent in self.all_entities:
            unique_by_type[ent['label']].add(ent['text'].lower())
        
        unique_counts = {label: len(entities) for label, entities in unique_by_type.items()}
        
        # Top entities by type
        top_entities_by_type = {}
        for label in entity_types.keys():
            entities_of_type = [ent['text'] for ent in self.all_entities if ent['label'] == label]
            top_entities_by_type[label] = Counter(entities_of_type).most_common(10)
        
        # Entities per document
        entities_per_doc = {
            doc_id: len(entities) 
            for doc_id, entities in self.entity_by_doc.items()
        }
        
        stats = {
            'total_entities': len(self.all_entities),
            'entity_types': dict(entity_types),
            'unique_entities_by_type': unique_counts,
            'top_entities_by_type': top_entities_by_type,
            'entities_per_document': entities_per_doc,
            'avg_entities_per_doc': np.mean(list(entities_per_doc.values())),
            'documents_processed': len(self.documents)
        }
        
        return stats
    
    def extract_policy_specific_entities(self) -> Dict:
        """
        Extract policy-specific entities using pattern matching.
        
        Returns:
            Dictionary with policy-specific entities
        """
        print("\nExtracting policy-specific entities...")
        
        policy_entities = {
            'policy_instruments': [],
            'energy_carriers': [],
            'targets': [],
            'years': []
        }
        
        # Define patterns
        policy_patterns = [
            'subsidy', 'subsidies', 'incentive', 'tax credit', 'feed-in tariff',
            'carbon price', 'carbon tax', 'cap and trade', 'emissions trading',
            'regulation', 'directive', 'mandate', 'standard', 'target'
        ]
        
        energy_patterns = [
            'hydrogen', 'ammonia', 'methanol', 'e-fuel', 'biofuel',
            'solar', 'wind', 'renewable', 'fossil fuel', 'natural gas',
            'electricity', 'battery'
        ]
        
        target_patterns = [
            'net zero', 'carbon neutral', 'climate neutral', 'emissions reduction',
            'renewable energy target', 'ghg reduction', 'decarbonization'
        ]
        
        # Search in documents
        for doc_info in self.documents:
            text_lower = doc_info['text'].lower()
            doc_id = doc_info['doc_id']
            
            # Policy instruments
            for pattern in policy_patterns:
                if pattern in text_lower:
                    policy_entities['policy_instruments'].append({
                        'term': pattern,
                        'doc_id': doc_id,
                        'filename': doc_info['filename']
                    })
            
            # Energy carriers
            for pattern in energy_patterns:
                if pattern in text_lower:
                    policy_entities['energy_carriers'].append({
                        'term': pattern,
                        'doc_id': doc_id,
                        'filename': doc_info['filename']
                    })
            
            # Targets
            for pattern in target_patterns:
                if pattern in text_lower:
                    policy_entities['targets'].append({
                        'term': pattern,
                        'doc_id': doc_id,
                        'filename': doc_info['filename']
                    })
        
        # Count occurrences
        policy_summary = {
            'policy_instruments': Counter(e['term'] for e in policy_entities['policy_instruments']),
            'energy_carriers': Counter(e['term'] for e in policy_entities['energy_carriers']),
            'targets': Counter(e['term'] for e in policy_entities['targets'])
        }
        
        return policy_entities, policy_summary
    
    def build_entity_cooccurrence_matrix(self, entity_type: str = 'ORG', 
                                         top_n: int = 20) -> pd.DataFrame:
        """
        Build co-occurrence matrix for entities of a specific type.
        
        Args:
            entity_type: Type of entity (ORG, GPE, PERSON, etc.)
            top_n: Number of top entities to include
            
        Returns:
            DataFrame with co-occurrence counts
        """
        print(f"\nBuilding co-occurrence matrix for {entity_type} entities...")
        
        # Get top entities of this type
        entities_of_type = [ent['text'] for ent in self.all_entities if ent['label'] == entity_type]
        top_entities = [e[0] for e in Counter(entities_of_type).most_common(top_n)]
        
        # Build co-occurrence matrix
        cooccurrence = np.zeros((len(top_entities), len(top_entities)))
        
        for doc_id, entities in self.entity_by_doc.items():
            # Get entities of this type in this document
            doc_entities = [ent['text'] for ent in entities if ent['label'] == entity_type and ent['text'] in top_entities]
            
            # Count co-occurrences
            for i, ent1 in enumerate(top_entities):
                for j, ent2 in enumerate(top_entities):
                    if ent1 in doc_entities and ent2 in doc_entities:
                        cooccurrence[i, j] += 1
        
        # Create DataFrame
        cooccurrence_df = pd.DataFrame(
            cooccurrence,
            index=top_entities,
            columns=top_entities
        )
        
        return cooccurrence_df
    
    def visualize_entity_distribution(self, stats: Dict, output_path: str = None):
        """
        Visualize entity type distribution.
        
        Args:
            stats: Entity statistics dictionary
            output_path: Path to save the figure
        """
        print("\nGenerating entity distribution visualization...")
        
        entity_types = stats['entity_types']
        
        # Sort by count
        sorted_types = dict(sorted(entity_types.items(), key=lambda x: x[-1], reverse=True))
        
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_types)))
        bars = plt.bar(range(len(sorted_types)), sorted_types.values(), color=colors)
        
        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Named Entity Types', fontsize=14, fontweight='bold')
        plt.xticks(range(len(sorted_types)), sorted_types.keys(), rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
        
        plt.close()
    
    def visualize_cooccurrence_heatmap(self, cooccurrence_df: pd.DataFrame, 
                                       entity_type: str, output_path: str = None):
        """
        Visualize entity co-occurrence as a heatmap.
        
        Args:
            cooccurrence_df: Co-occurrence matrix
            entity_type: Type of entities
            output_path: Path to save the figure
        """
        print(f"\nGenerating co-occurrence heatmap for {entity_type}...")
        
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(
            cooccurrence_df,
            annot=False,
            cmap='YlOrRd',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Co-occurrence Count'}
        )
        
        plt.title(f'{entity_type} Co-occurrence Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Entities', fontsize=12)
        plt.ylabel('Entities', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {output_path}")
        
        plt.close()
    
    def visualize_top_entities(self, stats: Dict, entity_type: str, 
                               top_n: int = 15, output_path: str = None):
        """
        Visualize top N entities of a specific type.
        
        Args:
            stats: Entity statistics
            entity_type: Type of entity to visualize
            top_n: Number of top entities
            output_path: Path to save figure
        """
        print(f"\nGenerating top {top_n} {entity_type} visualization...")
        
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
        plt.title(f'Top {top_n} {entity_type} Entities', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count, i, f' {count}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
        
        plt.close()
    
    def save_results(self, stats: Dict, policy_entities: Dict, policy_summary: Dict):
        """
        Save all NER results.
        
        Args:
            stats: Entity statistics
            policy_entities: Policy-specific entities
            policy_summary: Summary of policy entities
        """
        print("\nSaving NER results...")
        
        # Save all entities
        entities_df = pd.DataFrame(self.all_entities)
        entities_df.to_csv(self.ner_dir / "all_entities.csv", index=False)
        
        # Save statistics
        with open(self.ner_dir / "entity_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save policy entities
        with open(self.ner_dir / "policy_specific_entities.json", 'w') as f:
            json.dump({
                'entities': policy_entities,
                'summary': {k: dict(v) for k, v in policy_summary.items()}
            }, f, indent=2)
        
        # Save entities by document
        with open(self.ner_dir / "entities_by_document.json", 'w') as f:
            json.dump(self.entity_by_doc, f, indent=2)
        
        # Create summary report
        with open(self.ner_dir / "ner_summary.txt", 'w') as f:
            f.write("NAMED ENTITY RECOGNITION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Entities Extracted: {stats['total_entities']}\n")
            f.write(f"Documents Processed: {stats['documents_processed']}\n")
            f.write(f"Average Entities per Document: {stats['avg_entities_per_doc']:.1f}\n\n")
            
            f.write("ENTITY TYPES\n")
            f.write("-" * 80 + "\n")
            for entity_type, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {entity_type}: {count} ({stats['unique_entities_by_type'].get(entity_type, 0)} unique)\n")
            
            f.write("\n\nTOP ORGANIZATIONS\n")
            f.write("-" * 80 + "\n")
            for entity, count in stats['top_entities_by_type'].get('ORG', [])[:10]:
                f.write(f"  {entity}: {count}\n")
            
            f.write("\n\nTOP LOCATIONS\n")
            f.write("-" * 80 + "\n")
            for entity, count in stats['top_entities_by_type'].get('GPE', [])[:10]:
                f.write(f"  {entity}: {count}\n")
            
            f.write("\n\nPOLICY INSTRUMENTS\n")
            f.write("-" * 80 + "\n")
            for term, count in policy_summary['policy_instruments'].most_common(10):
                f.write(f"  {term}: {count}\n")
            
            f.write("\n\nENERGY CARRIERS\n")
            f.write("-" * 80 + "\n")
            for term, count in policy_summary['energy_carriers'].most_common(10):
                f.write(f"  {term}: {count}\n")
        
        print(f"✓ Results saved to {self.ner_dir}")


def main():
    """Main execution function."""
    print("="*80)
    print("PHASE 3: NAMED ENTITY RECOGNITION")
    print("="*80)
    
    # Initialize extractor
    extractor = PolicyEntityExtractor("preprocessed_output")
    
    # Extract entities
    extractor.extract_entities()
    
    # Get statistics
    stats = extractor.get_entity_statistics()
    
    # Extract policy-specific entities
    policy_entities, policy_summary = extractor.extract_policy_specific_entities()
    
    # Build co-occurrence matrices for key entity types
    org_cooccurrence = extractor.build_entity_cooccurrence_matrix('ORG', top_n=15)
    gpe_cooccurrence = extractor.build_entity_cooccurrence_matrix('GPE', top_n=15)
    
    # Generate visualizations
    extractor.visualize_entity_distribution(
        stats,
        output_path=extractor.ner_dir / "entity_distribution.png"
    )
    
    extractor.visualize_top_entities(
        stats, 'ORG', top_n=15,
        output_path=extractor.ner_dir / "top_organizations.png"
    )
    
    extractor.visualize_top_entities(
        stats, 'GPE', top_n=15,
        output_path=extractor.ner_dir / "top_locations.png"
    )
    
    extractor.visualize_cooccurrence_heatmap(
        org_cooccurrence, 'ORG',
        output_path=extractor.ner_dir / "org_cooccurrence.png"
    )
    
    extractor.visualize_cooccurrence_heatmap(
        gpe_cooccurrence, 'GPE',
        output_path=extractor.ner_dir / "gpe_cooccurrence.png"
    )
    
    # Save all results
    extractor.save_results(stats, policy_entities, policy_summary)
    
    # Print summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Entities: {stats['total_entities']:,}")
    print(f"  Unique Entity Types: {len(stats['entity_types'])}")
    print(f"  Avg per Document: {stats['avg_entities_per_doc']:.1f}")
    
    print(f"\n🏢 TOP 5 ORGANIZATIONS")
    for entity, count in stats['top_entities_by_type'].get('ORG', [])[:5]:
        print(f"  {entity}: {count}")
    
    print(f"\n🌍 TOP 5 LOCATIONS")
    for entity, count in stats['top_entities_by_type'].get('GPE', [])[:5]:
        print(f"  {entity}: {count}")
    
    print(f"\n🔧 TOP 5 POLICY INSTRUMENTS")
    for term, count in policy_summary['policy_instruments'].most_common(5):
        print(f"  {term}: {count}")
    
    print(f"\n⚡ TOP 5 ENERGY CARRIERS")
    for term, count in policy_summary['energy_carriers'].most_common(5):
        print(f"  {term}: {count}")
    
    print("\n" + "="*80)
    print("✅ Phase 3 Complete!")
    print("="*80)
    print(f"\nOutputs saved to: {extractor.ner_dir}/")
    print("  - all_entities.csv")
    print("  - entity_statistics.json")
    print("  - policy_specific_entities.json")
    print("  - entities_by_document.json")
    print("  - ner_summary.txt")
    print("  - entity_distribution.png")
    print("  - top_organizations.png")
    print("  - top_locations.png")
    print("  - org_cooccurrence.png")
    print("  - gpe_cooccurrence.png")
    print("\nNext: Phase 4 - Topic Modeling with BERTopic")


if __name__ == "__main__":
    main()