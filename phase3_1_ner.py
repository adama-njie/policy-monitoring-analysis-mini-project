"""
Phase 3: Entity Cleaning & Consolidation
=========================================
Clean and consolidate extracted entities to improve quality.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set

class EntityCleaner:
    """
    Clean and consolidate extracted entities.
    """
    
    def __init__(self, ner_dir: str):
        """
        Initialize cleaner.
        
        Args:
            ner_dir: Directory containing NER results
        """
        self.ner_dir = Path(ner_dir)
        
        print("Loading entity data...")
        self.entities_df = pd.read_csv(self.ner_dir / "all_entities.csv")
        
        print(f"✓ Loaded {len(self.entities_df)} entities")
        
        # Define cleaning rules
        self._define_cleaning_rules()
        
    def _define_cleaning_rules(self):
        """Define entity cleaning and consolidation rules."""
        
        # Geographic consolidation (cities → countries)
        self.geo_consolidation = {
            'Paris': 'France',
            'Hamburg': 'Germany',
            'Berlin': 'Germany',
            'London': 'United Kingdom',
            'Brussels': 'Belgium',
            'Copenhagen': 'Denmark',
            'Madrid': 'Spain',
            'Rome': 'Italy',
            'Amsterdam': 'Netherlands',
            'Vienna': 'Austria',
            'Stockholm': 'Sweden',
            'Warsaw': 'Poland',
            'Prague': 'Czech Republic',
            'Lisbon': 'Portugal',
            'Athens': 'Greece',
            'Dublin': 'Ireland',
            'Helsinki': 'Finland',
            'Budapest': 'Hungary',
            'Bucharest': 'Romania',
            'Sofia': 'Bulgaria'
        }
        
        # Organization name normalization
        self.org_normalization = {
            'EU': 'European Union',
            'EC': 'European Commission',
            'EEA': 'European Union',
            'Union': 'European Union',
            'Commission': 'European Commission',
            'Council': 'European Council',
            'the European Parliament': 'European Parliament',
            'UK': 'United Kingdom',
            'US': 'United States',
            'USA': 'United States'
        }
        
        # Acronyms to full names (policy/technical terms)
        self.acronym_expansion = {
            'GHG': 'Greenhouse Gas',
            'CCS': 'Carbon Capture and Storage',
            'CBAM': 'Carbon Border Adjustment Mechanism',
            'SAF': 'Sustainable Aviation Fuel',
            'PtL': 'Power-to-Liquid',
            'PtX': 'Power-to-X',
            'DAC': 'Direct Air Capture',
            'CCUS': 'Carbon Capture Utilization and Storage',
            'ETS': 'Emissions Trading System',
            'RED': 'Renewable Energy Directive',
            'RFNBO': 'Renewable Fuels of Non-Biological Origin'
        }
        
        # False positives to remove by entity type
        self.false_positives = {
            'GPE': {  # Not geopolitical entities
                'Ammonia', 'pp', 'Duwe', 'CCS', 'al.', 'regulations',
                'Figure', 'Table', 'Annex', 'Article', 'Section',
                'CO2', 'GHG', 'et al', 'ibid', 'op. cit.'
            },
            'ORG': {  # Not organizations
                'CO2', 'GHG', 'CCS', 'Hydrogen', 'Ammonia', 'DAC',
                'OJ L', 'ELI', 'MW', 'GW', 'TWh', 'kWh',
                'Annex', 'Article', 'Societal Transitions',
                'pp', 'et al', 'ibid', 'Figure', 'Table'
            },
            'PERSON': {  # Common false positives for names
                'al', 'et al', 'ibid', 'op', 'cit', 'pp'
            }
        }
        
        # Technical terms that should be reclassified
        self.reclassify = {
            'CBAM': {'from': 'ORG', 'to': 'LAW'},
            'Carbon Border Adjustment Mechanism': {'from': 'ORG', 'to': 'LAW'},
            'ETS': {'from': 'ORG', 'to': 'LAW'},
            'Emissions Trading System': {'from': 'ORG', 'to': 'LAW'},
            'Paris Agreement': {'from': 'GPE', 'to': 'LAW'},
            'Kyoto Protocol': {'from': 'GPE', 'to': 'LAW'}
        }
        
        # Common person names that are misclassified
        self.known_persons = {
            'A.E Torkayesh', 'Torkayesh', 'Duwe'
        }
        
    def clean_entities(self) -> pd.DataFrame:
        """
        Clean and consolidate entities.
        
        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning entities...")
        
        df = self.entities_df.copy()
        original_count = len(df)
        
        # Step 1: Remove false positives
        print("  Step 1: Removing false positives...")
        for entity_type, false_pos_set in self.false_positives.items():
            mask = (df['label'] == entity_type) & (df['text'].isin(false_pos_set))
            removed = mask.sum()
            df = df[~mask]
            if removed > 0:
                print(f"    Removed {removed} false {entity_type} entities")
        
        # Step 2: Geographic consolidation
        print("  Step 2: Consolidating geographic entities...")
        geo_mask = df['label'] == 'GPE'
        df.loc[geo_mask, 'text'] = df.loc[geo_mask, 'text'].replace(self.geo_consolidation)
        consolidated = sum(1 for k, v in self.geo_consolidation.items() if k in df[geo_mask]['text'].values)
        print(f"    Consolidated {len(self.geo_consolidation)} city names to countries")
        
        # Step 3: Normalize organization names
        print("  Step 3: Normalizing organization names...")
        org_mask = df['label'] == 'ORG'
        df.loc[org_mask, 'text'] = df.loc[org_mask, 'text'].replace(self.org_normalization)
        print(f"    Normalized {len(self.org_normalization)} organization variants")
        
        # Step 4: Expand acronyms
        print("  Step 4: Expanding technical acronyms...")
        for label in ['ORG', 'PRODUCT', 'GPE']:
            mask = df['label'] == label
            df.loc[mask, 'text'] = df.loc[mask, 'text'].replace(self.acronym_expansion)
        
        # Step 5: Reclassify misclassified entities
        print("  Step 5: Reclassifying entities...")
        for entity_text, reclassification in self.reclassify.items():
            mask = (df['text'] == entity_text) & (df['label'] == reclassification['from'])
            df.loc[mask, 'label'] = reclassification['to']
            if mask.sum() > 0:
                print(f"    Reclassified {entity_text}: {reclassification['from']} → {reclassification['to']}")
        
        # Step 6: Handle known persons
        print("  Step 6: Fixing person entity misclassifications...")
        for person_name in self.known_persons:
            # If misclassified as ORG, fix to PERSON
            mask = (df['text'] == person_name) & (df['label'] == 'ORG')
            if mask.sum() > 0:
                df.loc[mask, 'label'] = 'PERSON'
                print(f"    Fixed: {person_name} (ORG → PERSON)")
        
        # Step 7: Remove very short entities (likely noise)
        print("  Step 7: Removing very short entities...")
        short_mask = df['text'].str.len() <= 1
        removed_short = short_mask.sum()
        df = df[~short_mask]
        if removed_short > 0:
            print(f"    Removed {removed_short} single-character entities")
        
        # Step 8: Clean whitespace and special characters
        df['text'] = df['text'].str.strip()
        
        print(f"\n✓ Cleaning complete:")
        print(f"  Original entities: {original_count:,}")
        print(f"  After cleaning: {len(df):,}")
        print(f"  Removed: {original_count - len(df):,} ({(original_count - len(df))/original_count*100:.1f}%)")
        
        return df
    
    def create_entity_glossary(self, cleaned_df: pd.DataFrame) -> Dict:
        """
        Create a glossary of entities with explanations.
        
        Args:
            cleaned_df: Cleaned entities DataFrame
            
        Returns:
            Dictionary with entity glossary
        """
        print("\nCreating entity glossary...")
        
        glossary = {
            'organizations': {},
            'locations': {},
            'laws': {},
            'technical_terms': {}
        }
        
        # Organizations
        org_entities = cleaned_df[cleaned_df['label'] == 'ORG']['text'].value_counts()
        for org, count in org_entities.head(30).items():
            explanation = self._explain_organization(org)
            glossary['organizations'][org] = {
                'count': int(count),
                'explanation': explanation
            }
        
        # Locations
        gpe_entities = cleaned_df[cleaned_df['label'] == 'GPE']['text'].value_counts()
        for location, count in gpe_entities.head(20).items():
            explanation = self._explain_location(location)
            glossary['locations'][location] = {
                'count': int(count),
                'explanation': explanation
            }
        
        # Laws
        law_entities = cleaned_df[cleaned_df['label'] == 'LAW']['text'].value_counts()
        for law, count in law_entities.head(20).items():
            glossary['laws'][law] = {
                'count': int(count),
                'explanation': 'Policy or legal framework'
            }
        
        # Technical terms (from acronym expansion)
        for acronym, full_name in self.acronym_expansion.items():
            if acronym in cleaned_df['text'].values or full_name in cleaned_df['text'].values:
                count = len(cleaned_df[cleaned_df['text'].isin([acronym, full_name])])
                glossary['technical_terms'][full_name] = {
                    'acronym': acronym,
                    'count': count,
                    'explanation': self._explain_technical_term(full_name)
                }
        
        return glossary
    
    def _explain_organization(self, org: str) -> str:
        """Provide explanation for organization."""
        explanations = {
            'European Union': 'Political and economic union of European countries',
            'European Commission': 'Executive branch of the European Union',
            'European Parliament': 'Legislative body of the European Union',
            'European Council': 'Institution defining EU political direction and priorities',
            'IPCC': 'Intergovernmental Panel on Climate Change',
            'WWF': 'World Wildlife Fund',
            'IEA': 'International Energy Agency',
            'IRENA': 'International Renewable Energy Agency',
            'OECD': 'Organisation for Economic Co-operation and Development'
        }
        return explanations.get(org, 'Organization mentioned in policy documents')
    
    def _explain_location(self, location: str) -> str:
        """Provide explanation for location."""
        if location == 'Member States':
            return 'Countries that are members of the European Union'
        elif location in ['Germany', 'France', 'Spain', 'Italy', 'Poland', 'Netherlands']:
            return f'EU Member State, major player in energy policy'
        elif location == 'China':
            return 'Major global economy and energy consumer'
        elif location == 'United States':
            return 'Major global economy and policy influencer'
        else:
            return 'Geographic location referenced in policy context'
    
    def _explain_technical_term(self, term: str) -> str:
        """Provide explanation for technical term."""
        explanations = {
            'Greenhouse Gas': 'Gases that trap heat in the atmosphere (CO2, CH4, etc.)',
            'Carbon Capture and Storage': 'Technology to capture CO2 emissions and store them',
            'Carbon Border Adjustment Mechanism': 'EU policy to prevent carbon leakage',
            'Sustainable Aviation Fuel': 'Alternative fuel for aviation with lower emissions',
            'Power-to-Liquid': 'Technology converting electricity to liquid fuels',
            'Power-to-X': 'Technologies converting electricity to other energy carriers',
            'Direct Air Capture': 'Technology to remove CO2 directly from atmosphere',
            'Emissions Trading System': 'Market-based approach to controlling emissions',
            'Renewable Fuels of Non-Biological Origin': 'Synthetic fuels from renewable sources'
        }
        return explanations.get(term, 'Technical term in energy/climate policy')
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, 
                                 cleaned_df: pd.DataFrame,
                                 glossary: Dict) -> str:
        """
        Generate comprehensive cleaning report.
        
        Args:
            original_df: Original entities
            cleaned_df: Cleaned entities
            glossary: Entity glossary
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("ENTITY CLEANING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("CLEANING SUMMARY")
        report.append("-" * 80)
        report.append(f"Original entities: {len(original_df):,}")
        report.append(f"After cleaning: {len(cleaned_df):,}")
        report.append(f"Removed: {len(original_df) - len(cleaned_df):,} ({(len(original_df) - len(cleaned_df))/len(original_df)*100:.1f}%)")
        report.append("")
        
        # Changes by entity type
        report.append("CHANGES BY ENTITY TYPE")
        report.append("-" * 80)
        orig_counts = original_df['label'].value_counts()
        clean_counts = cleaned_df['label'].value_counts()
        
        for label in sorted(set(orig_counts.index) | set(clean_counts.index)):
            orig = orig_counts.get(label, 0)
            clean = clean_counts.get(label, 0)
            diff = orig - clean
            if diff != 0:
                report.append(f"  {label}: {orig:,} → {clean:,} (Δ {diff:+,})")
        report.append("")
        
        # Top organizations (cleaned)
        report.append("TOP 15 ORGANIZATIONS (AFTER CLEANING)")
        report.append("-" * 80)
        for org, data in list(glossary['organizations'].items())[:15]:
            report.append(f"  {org}: {data['count']:,}")
            report.append(f"    → {data['explanation']}")
        report.append("")
        
        # Top locations (cleaned)
        report.append("TOP 10 LOCATIONS (AFTER CLEANING)")
        report.append("-" * 80)
        for loc, data in list(glossary['locations'].items())[:10]:
            report.append(f"  {loc}: {data['count']:,}")
            report.append(f"    → {data['explanation']}")
        report.append("")
        
        # Technical terms
        report.append("KEY TECHNICAL TERMS")
        report.append("-" * 80)
        for term, data in glossary['technical_terms'].items():
            report.append(f"  {term} ({data['acronym']}): {data['count']:,}")
            report.append(f"    → {data['explanation']}")
        report.append("")
        
        # Cleaning rules applied
        report.append("CLEANING RULES APPLIED")
        report.append("-" * 80)
        report.append(f"  1. Geographic consolidation: {len(self.geo_consolidation)} city→country mappings")
        report.append(f"  2. Organization normalization: {len(self.org_normalization)} variants standardized")
        report.append(f"  3. Acronym expansion: {len(self.acronym_expansion)} acronyms expanded")
        report.append(f"  4. False positive removal: {sum(len(v) for v in self.false_positives.values())} patterns")
        report.append(f"  5. Entity reclassification: {len(self.reclassify)} entities reclassified")
        report.append(f"  6. Person name fixes: {len(self.known_persons)} known persons")
        report.append("")
        
        return "\n".join(report)
    
    def save_cleaned_results(self, cleaned_df: pd.DataFrame, glossary: Dict, report: str):
        """Save cleaned entities and documentation."""
        print("\nSaving cleaned results...")
        
        # Save cleaned entities
        cleaned_df.to_csv(self.ner_dir / "all_entities_CLEANED.csv", index=False)
        
        # Save glossary
        with open(self.ner_dir / "entity_glossary.json", 'w') as f:
            json.dump(glossary, f, indent=2)
        
        # Save cleaning report
        with open(self.ner_dir / "cleaning_report.txt", 'w') as f:
            f.write(report)
        
        # Update statistics with cleaned data
        stats = self._calculate_cleaned_statistics(cleaned_df)
        with open(self.ner_dir / "entity_statistics_CLEANED.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Cleaned results saved to {self.ner_dir}")
    
    def _calculate_cleaned_statistics(self, cleaned_df: pd.DataFrame) -> Dict:
        """Calculate statistics for cleaned entities."""
        stats = {
            'total_entities': len(cleaned_df),
            'entity_types': cleaned_df['label'].value_counts().to_dict(),
            'unique_entities_by_type': cleaned_df.groupby('label')['text'].nunique().to_dict(),
            'top_entities_by_type': {}
        }
        
        for label in cleaned_df['label'].unique():
            entities = cleaned_df[cleaned_df['label'] == label]['text'].value_counts().head(15)
            stats['top_entities_by_type'][label] = list(entities.items())
        
        return stats


if __name__ == "__main__":
    """Main execution function."""
    print("="*80)
    print("PHASE 3: ENTITY CLEANING & CONSOLIDATION")
    print("="*80)
    
    # Initialize cleaner
    cleaner = EntityCleaner("preprocessed_output/named_entities")
    
    # Load original data
    original_df = cleaner.entities_df.copy()
    
    # Clean entities
    cleaned_df = cleaner.clean_entities()
    
    # Create glossary
    glossary = cleaner.create_entity_glossary(cleaned_df)
    
    # Generate report
    report = cleaner.generate_cleaning_report(original_df, cleaned_df, glossary)
    
    # Print report
    print("\n" + report)
    
    # Save results
    cleaner.save_cleaned_results(cleaned_df, glossary, report)
    
    print("\n" + "="*80)
    print("✅ Entity Cleaning Complete!")
    print("="*80)
    print("\nNew files created:")
    print("  - all_entities_CLEANED.csv")
    print("  - entity_glossary.json")
    print("  - cleaning_report.txt")
    print("  - entity_statistics_CLEANED.json")
    print("\nTo regenerate visualizations with cleaned data:")
    print("  python phase3_ner_regenerate.py")


