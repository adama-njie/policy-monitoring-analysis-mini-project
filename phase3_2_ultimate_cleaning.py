"""
Phase 3: Ultimate Entity Cleaning v3
=====================================
Comprehensive cleaning addressing all identified issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import re

def ultimate_clean_entities(input_csv, output_csv):
    """
    Perform ultimate comprehensive cleaning of entities.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to save cleaned CSV
    """
    print("="*80)
    print("PHASE 3: ULTIMATE ENTITY CLEANING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    original_count = len(df)
    print(f"Original entities: {original_count:,}")
    
    # Normalize text first (remove extra spaces, normalize encoding)
    print("\nStep 0: Normalizing text encoding and whitespace...")
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
    df['text'] = df['text'].str.replace(r'\xa0', ' ', regex=True)  # Non-breaking space
    df['text'] = df['text'].str.replace(r'[\u200b-\u200f]', '', regex=True)  # Zero-width chars
    
    # Step 1: Geographic consolidation - COMPREHENSIVE
    print("\nStep 1: Comprehensive geographic consolidation...")
    geo_consolidation = {
        # UK variants
        'UK': 'United Kingdom',
        'U.K.': 'United Kingdom',
        'United Kingdom': 'United Kingdom',  # Normalize duplicates
        'the United Kingdom': 'United Kingdom',
        'Britain': 'United Kingdom',
        'Great Britain': 'United Kingdom',
        
        # Germany variants (all possible forms)
        'Germany': 'Germany',
        'Deutschland': 'Germany',  # German name
        'Bundesrepublik Deutschland': 'Germany',
        'FRG': 'Germany',
        'West Germany': 'Germany',
        'East Germany': 'Germany',
        
        # France variants
        'France': 'France',
        'Paris': 'France',
        
        # Spain variants
        'Spain': 'Spain',
        'Madrid': 'Spain',
        
        # Other cities to countries
        'Hamburg': 'Germany',
        'Berlin': 'Germany',
        'Munich': 'Germany',
        'Frankfurt': 'Germany',
        'London': 'United Kingdom',
        'Brussels': 'Belgium',
        'Copenhagen': 'Denmark',
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
        'Sofia': 'Bulgaria',
        
        # Switzerland and cities
        'Switzerland': 'Switzerland',
        'Geneva': 'Switzerland',
        'Zurich': 'Switzerland',
        'Bern': 'Switzerland',
        'Basel': 'Switzerland',
        
        # US variants
        'US': 'United States',
        'U.S.': 'United States',
        'USA': 'United States',
        'U.S.A.': 'United States',
        'the United States': 'United States',
        
        # China variants
        'China': 'China',
        'PRC': 'China',
        'People\'s Republic of China': 'China'
    }
    
    geo_mask = df['label'] == 'GPE'
    df.loc[geo_mask, 'text'] = df.loc[geo_mask, 'text'].replace(geo_consolidation)
    print(f"  Applied {len(geo_consolidation)} geographic consolidation rules")
    
    # Step 2: Organization consolidation - COMPREHENSIVE
    print("\nStep 2: Comprehensive organization consolidation...")
    org_consolidation = {
        # European Union variants
        'EU': 'European Union',
        'EEA': 'European Union',
        'Union': 'European Union',
        'the Union': 'European Union',
        'European Union': 'European Union',
        'the European Union': 'European Union',
        'the European': 'European Union',  # Fragment
        
        # European Commission variants
        'EC': 'European Commission',
        'Commission': 'European Commission',
        'the Commission': 'European Commission',
        'European Commission': 'European Commission',
        'the European Commission': 'European Commission',
        
        # European Council
        'Council': 'European Council',
        'the Council': 'European Council',
        'European Council': 'European Council',
        
        # European Parliament
        'Parliament': 'European Parliament',
        'the Parliament': 'European Parliament',
        'European Parliament': 'European Parliament',
        'the European Parliament': 'European Parliament',
        
        # Other organizations
        'IPCC': 'IPCC',
        'WWF': 'WWF',
        'IEA': 'International Energy Agency',
        'IRENA': 'International Renewable Energy Agency',
        'OECD': 'OECD'
    }
    
    org_mask = df['label'] == 'ORG'
    df.loc[org_mask, 'text'] = df.loc[org_mask, 'text'].replace(org_consolidation)
    print(f"  Applied {len(org_consolidation)} organization consolidation rules")
    
    # Step 3: Expand ALL acronyms (apply to all entity types)
    print("\nStep 3: Expanding technical acronyms...")
    acronym_expansion = {
        'GHG': 'Greenhouse Gas',
        'CCS': 'Carbon Capture and Storage',
        'CCU': 'Carbon Capture and Utilization',
        'CCUS': 'Carbon Capture Utilization and Storage',
        'CBAM': 'Carbon Border Adjustment Mechanism',
        'SAF': 'Sustainable Aviation Fuel',
        'PtL': 'Power-to-Liquid',
        'PtX': 'Power-to-X',
        'DAC': 'Direct Air Capture',
        'ETS': 'Emissions Trading System',
        'RED': 'Renewable Energy Directive',
        'RFNBO': 'Renewable Fuels of Non-Biological Origin',
        'LCA': 'Life Cycle Assessment',
        'SMR': 'Steam Methane Reforming'
    }
    
    # Apply to all entity types
    df['text'] = df['text'].replace(acronym_expansion)
    print(f"  Expanded {len(acronym_expansion)} acronyms across all entity types")
    
    # Step 4: Remove false positive GPE entities - COMPREHENSIVE
    print("\nStep 4: Removing false positive GPE entities...")
    false_gpe = {
        # Technical/chemical terms
        'Ammonia', 'Hydrogen', 'Methanol', 'Methane', 'Oxygen', 'Nitrogen',
        'CO2', 'CO 2', 'Carbon Capture and Storage', 'Carbon Capture and Utilization',
        'Greenhouse Gas', 
        
        # Document structure (EXPANDED)
        'Regulation', 'Article', 'Annex', 'Figure', 'Table', 'Section', 
        'Chapter', 'Appendix', 'Part', 'Title',
        # Specific section/table references
        'Section 1', 'Section 2', 'Section 3', 'Section 4', 'Section 5', 
        'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
        'Table 1', 'Table 2', 'Table 3', 'Table 4', 'Table 5',
        'Table 6', 'Table 7', 'Table 8', 'Table 9', 'Table 10',
        'Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure 5',
        'Annex I', 'Annex II', 'Annex III', 'Annex IV',
        'Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5',
        
        # Text artifacts
        'pp', 'al', 'et al', 'et al.', 'ibid', 'op. cit.', 'op', 'cit',
        'Duwe', 'al.', 'et',
        
        # Common words that aren't places
        'Commission', 'Council', 'Parliament', 'Union',
        'renewable', 'fossil', 'energy', 'power', 'gas', 'oil', 'coal',
        'wind', 'solar', 'nuclear', 'hydro',
        
        # Single letters and very short
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        
        # Web/document references
        'www', 'http', 'https', 'doi', 'pdf', 'html', 'com', 'org', 'net'
    }
    
    gpe_mask = (df['label'] == 'GPE') & (df['text'].isin(false_gpe))
    removed_gpe = gpe_mask.sum()
    df = df[~gpe_mask]
    print(f"  Removed {removed_gpe} false GPE entities")
    
    # Step 5: Remove false positive ORG entities - COMPREHENSIVE
    print("\nStep 5: Removing false positive ORG entities...")
    false_org = {
        # Chemical/energy terms (already expanded or not orgs)
        'CO2', 'CO 2', 'Hydrogen', 'Ammonia', 'Methanol', 'Methane', 'Oxygen',
        'Carbon Capture and Storage', 'Carbon Capture and Utilization',
        'Greenhouse Gas', 'Carbon Capture Utilization and Storage',
        'Direct Air Capture',
        
        # Technical systems (not organizations)
        'Power-to-Liquid', 'Power-to-X', 'Steam Methane Reforming',
        
        # Units of measurement
        'MW', 'GW', 'TWh', 'kWh', 'MWh', 'TJ', 'PJ', 'EJ',
        'Mt', 'Gt', 'kg', 'ton', 'tonne', 'tonnes',
        'kW', 'W', 'J', 'Wh',
        
        # Document references (EXPANDED)
        'OJ L', 'ELI', 'Annex', 'Article', 'Figure', 'Table',
        'Section', 'Chapter', 'pp', 'Vol', 'No', 'Appendix',
        # Specific references
        'Section 1', 'Section 2', 'Section 3', 'Section 4', 'Section 5',
        'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
        'Table 1', 'Table 2', 'Table 3', 'Table 4', 'Table 5',
        'Table 6', 'Table 7', 'Table 8', 'Table 9', 'Table 10',
        'Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure 5',
        
        # Text artifacts
        'et al', 'al', 'ibid', 'op. cit.', 'et al.', 'op', 'cit',
        'al.', 'et', 'e.g.', 'i.e.', 'cf.',
        
        # Generic concepts (not organizations)
        'Societal Transitions', 'Energy Transitions', 'Energy Transition',
        'Climate Change', 'Global Warming',
        
        # Single letters
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        
        # Web artifacts
        'www', 'http', 'https', 'doi', 'pdf', 'html', 'com', 'org', 'net',
        
        # Common nouns
        'Energy', 'Power', 'Gas', 'Oil', 'Coal', 'Wind', 'Solar',
        'Renewable', 'Fossil', 'Nuclear', 'Hydro'
    }
    
    org_mask = (df['label'] == 'ORG') & (df['text'].isin(false_org))
    removed_org = org_mask.sum()
    df = df[~org_mask]
    print(f"  Removed {removed_org} false ORG entities")
    
    # Step 6: Reclassify entities to correct types
    print("\nStep 6: Reclassifying entities to correct types...")
    
    # Remove from LAW - these are document structure, not actual laws
    document_structure_in_law = {
        'Section 1', 'Section 2', 'Section 3', 'Section 4', 'Section 5',
        'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
        'Table 1', 'Table 2', 'Table 3', 'Table 4', 'Table 5',
        'Table 6', 'Table 7', 'Table 8', 'Table 9', 'Table 10',
        'Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure 5',
        'Annex I', 'Annex II', 'Annex III', 'Annex IV',
        'Article 1', 'Article 2', 'Article 3', 'Article 4', 'Article 5',
        'Section', 'Table', 'Figure', 'Annex', 'Article', 'Chapter'
    }
    
    # Remove document structure from LAW
    law_structure_mask = (df['label'] == 'LAW') & (df['text'].isin(document_structure_in_law))
    removed_law_structure = law_structure_mask.sum()
    df = df[~law_structure_mask]
    print(f"  Removed {removed_law_structure} document structure items from LAW")
    
    # Laws/Regulations (currently ORG or GPE) - REAL laws only
    law_reclassifications = {
        'Carbon Border Adjustment Mechanism',
        'Emissions Trading System',
        'Renewable Energy Directive',
        'Paris Agreement',
        'Kyoto Protocol',
        'European Green Deal',
        'European Climate Law',
        'the European Climate Law',
        'Fit for 55',
        'Clean Energy Package'
    }
    
    reclassified = 0
    for entity_text in law_reclassifications:
        mask = (df['text'] == entity_text) & (df['label'].isin(['ORG', 'GPE']))
        if mask.sum() > 0:
            df.loc[mask, 'label'] = 'LAW'
            reclassified += mask.sum()
    
    # Products/Fuels (currently ORG)
    product_reclassifications = {
        'Sustainable Aviation Fuel'
    }
    
    for entity_text in product_reclassifications:
        mask = (df['text'] == entity_text) & (df['label'] == 'ORG')
        if mask.sum() > 0:
            df.loc[mask, 'label'] = 'PRODUCT'
            reclassified += mask.sum()
    
    print(f"  Reclassified {reclassified} entities to correct types")
    
    # Step 7: Fix person name misclassifications
    print("\nStep 7: Fixing person name misclassifications...")
    known_persons = {
        # Authors and researchers
        'A.E Torkayesh', 'Torkayesh', 'A.E. Torkayesh', 'A. E. Torkayesh',
        'Duwe', 'M. Duwe', 'M Duwe',
        'Wang', 'Li', 'Zhang', 'Liu', 'Chen',  # Common Chinese surnames
        'Kim', 'Park', 'Lee',  # Common Korean surnames
        'Smith', 'Johnson', 'Brown',  # Common English surnames
    }
    
    person_fixes = 0
    for person_name in known_persons:
        mask = (df['text'] == person_name) & (df['label'].isin(['ORG', 'GPE']))
        if mask.sum() > 0:
            df.loc[mask, 'label'] = 'PERSON'
            person_fixes += mask.sum()
    print(f"  Fixed {person_fixes} person name misclassifications")
    
    # Step 8: Remove very short entities
    print("\nStep 8: Removing very short entities...")
    short_mask = df['text'].str.len() <= 2
    removed_short = short_mask.sum()
    df = df[~short_mask]
    print(f"  Removed {removed_short} entities with ≤2 characters")
    
    # Step 9: Remove numeric-only entities (except valid years)
    print("\nStep 9: Removing numeric-only entities...")
    numeric_mask = df['text'].str.match(r'^\d+$', na=False)
    # Keep 4-digit years (1900-2099)
    year_mask = df['text'].str.match(r'^(19|20)\d{2}$', na=False)
    remove_numeric = numeric_mask & ~year_mask
    removed_numeric = remove_numeric.sum()
    df = df[~remove_numeric]
    print(f"  Removed {removed_numeric} numeric entities (kept years)")
    
    # Step 10: Remove special character entities
    print("\nStep 10: Removing special character entities...")
    special_mask = df['text'].str.match(r'^[^\w\s]+$', na=False)
    removed_special = special_mask.sum()
    df = df[~special_mask]
    print(f"  Removed {removed_special} special character entities")
    
    # Step 11: Remove entities that are just punctuation or common words
    print("\nStep 11: Removing common stopwords as entities...")
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
    }
    
    stopword_mask = df['text'].str.lower().isin(stopwords)
    removed_stopwords = stopword_mask.sum()
    df = df[~stopword_mask]
    print(f"  Removed {removed_stopwords} stopword entities")
    
    # Step 12: Remove duplicate entity mentions (same text, same doc, same label)
    print("\nStep 12: Removing duplicate entries...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text', 'doc_id', 'label'], keep='first')
    removed_dupes = before_dedup - len(df)
    print(f"  Removed {removed_dupes} duplicate entries")
    
    # Step 13: Remove entities with empty text after cleaning
    print("\nStep 13: Final cleanup...")
    df = df[df['text'].str.len() > 0]
    df = df[df['text'].notna()]
    
    # Final statistics
    final_count = len(df)
    removed_total = original_count - final_count
    
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    print(f"Original entities:  {original_count:,}")
    print(f"After cleaning:     {final_count:,}")
    print(f"Removed:            {removed_total:,} ({removed_total/original_count*100:.1f}%)")
    
    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Generate detailed statistics
    print("\nGenerating statistics...")
    
    stats = {
        'total_entities': int(final_count),
        'entities_removed': int(removed_total),
        'removal_percentage': float(removed_total/original_count*100),
        'entity_types': {k: int(v) for k, v in df['label'].value_counts().to_dict().items()},
        'unique_entities_by_type': {k: int(v) for k, v in df.groupby('label')['text'].nunique().to_dict().items()},
        'top_entities_by_type': {}
    }
    
    for label in df['label'].unique():
        top_entities = df[df['label'] == label]['text'].value_counts().head(25)
        stats['top_entities_by_type'][label] = [(k, int(v)) for k, v in top_entities.items()]
    
    # Save statistics
    stats_path = Path(output_csv).parent / "entity_statistics_ULTIMATE_CLEANED.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    
    # Print top entities for verification
    print("\n" + "="*80)
    print("TOP ENTITIES AFTER ULTIMATE CLEANING")
    print("="*80)
    
    print("\n🏢 Top 20 Organizations:")
    for i, (entity, count) in enumerate(stats['top_entities_by_type'].get('ORG', [])[:20], 1):
        print(f"  {i:2d}. {entity}: {count}")
    
    print("\n🌍 Top 20 Locations:")
    for i, (entity, count) in enumerate(stats['top_entities_by_type'].get('GPE', [])[:20], 1):
        print(f"  {i:2d}. {entity}: {count}")
    
    print("\n⚖️  Top 10 Laws/Regulations:")
    for i, (entity, count) in enumerate(stats['top_entities_by_type'].get('LAW', [])[:10], 1):
        print(f"  {i:2d}. {entity}: {count}")
    
    # Verification checks
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    
    # Check for UK consolidation
    uk_entities = df[df['text'].str.contains('UK|United Kingdom|Britain', case=False, na=False)]['text'].unique()
    print(f"\nUK-related entities: {list(uk_entities)}")
    
    # Check for Germany consolidation
    germany_entities = df[(df['label'] == 'GPE') & df['text'].str.contains('Germany|Deutschland', case=False, na=False)]['text'].unique()
    print(f"Germany-related entities: {list(germany_entities)}")
    
    # Check for Commission consolidation
    commission_entities = df[df['text'].str.contains('Commission', case=False, na=False)]['text'].value_counts().head(5)
    print(f"\nCommission variants remaining:")
    for entity, count in commission_entities.items():
        print(f"  {entity}: {count}")
    
    # Check for Union consolidation
    union_entities = df[df['text'].str.contains('Union', case=False, na=False) & (df['label'] == 'ORG')]['text'].value_counts().head(5)
    print(f"\nUnion variants remaining:")
    for entity, count in union_entities.items():
        print(f"  {entity}: {count}")
    
    return df, stats

if __name__ == "__main__":
    # Perform ultimate cleaning on ORIGINAL data (start fresh)
    input_file = "preprocessed_output/named_entities/all_entities.csv"
    output_file = "preprocessed_output/named_entities/all_entities_ULTIMATE_CLEANED.csv"
    
    print("Starting with original entity data for cleanest results...")
    df, stats = ultimate_clean_entities(input_file, output_file)
    
    print("\n" + "="*80)
    print("✅ ULTIMATE CLEANING COMPLETE!")
    print("="*80)
    print("\nFiles created:")
    print(f"  - {output_file}")
    print(f"  - entity_statistics_ULTIMATE_CLEANED.json")
    print("\nNext steps:")
    print("1. Review verification checks above")
    print("2. Run: python phase3_regenerate.py")
    print("3. Run: python phase3_entity_network.py")