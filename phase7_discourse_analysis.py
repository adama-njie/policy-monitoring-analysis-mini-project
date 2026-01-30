"""
Phase 7: Discourse Analysis
============================
Analyze policy discourse patterns, framing, and rhetorical strategies.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import Counter, defaultdict
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

class PolicyDiscourseAnalyzer:
    """
    Analyze discourse patterns in policy documents:
    - Modal verbs (obligation, possibility, necessity)
    - Hedging and certainty markers
    - Active vs passive voice
    - Temporal framing
    - Stakeholder mentions and agency
    """
    
    def __init__(self, preprocessed_dir: str):
        """
        Initialize discourse analyzer.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
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
            
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                self.documents.append(text)
                self.doc_info.append({
                    'doc_id': doc_id,
                    'filename': meta['filename'],
                    'doc_type': meta['document_type'],
                    'year': meta['year']
                })
            except FileNotFoundError:
                print(f"  ⚠️ Warning: Text file not found for {meta['filename']}")
                continue
        
        print(f"✓ Loaded {len(self.documents)} documents")
        
        # Define discourse markers
        self._define_discourse_markers()
        
        # Create output directory
        self.discourse_dir = self.preprocessed_dir / "discourse_analysis"
        self.discourse_dir.mkdir(exist_ok=True)
        
        print("✓ Discourse analyzer ready")
    
    def _define_discourse_markers(self):
        """Define linguistic markers for discourse analysis."""
        
        # Modal verbs indicating different discourse functions
        self.modals = {
            'obligation': ['must', 'shall', 'should', 'ought'],
            'possibility': ['may', 'might', 'could', 'can'],
            'necessity': ['need', 'require', 'necessary'],
            'permission': ['allow', 'permit', 'enable'],
            'prohibition': ['prohibit', 'forbid', 'ban', 'prevent']
        }
        
        # Hedging markers (uncertainty, politeness)
        self.hedges = [
            'perhaps', 'possibly', 'probably', 'likely', 'unlikely',
            'seem', 'appear', 'tend', 'suggest', 'indicate',
            'somewhat', 'rather', 'quite', 'fairly', 'relatively',
            'approximately', 'roughly', 'about', 'around'
        ]
        
        # Certainty markers (strong statements)
        self.boosters = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'indeed', 'always', 'never', 'absolutely', 'completely',
            'essential', 'critical', 'crucial', 'vital', 'key'
        ]
        
        # Temporal markers
        self.temporal = {
            'past': ['was', 'were', 'had', 'did', 'previous', 'former', 'past'],
            'present': ['is', 'are', 'has', 'do', 'current', 'now', 'present'],
            'future': ['will', 'shall', 'going to', 'future', 'next', 'upcoming',
                      'by 2030', 'by 2050', '2030', '2040', '2050']
        }
        
        # Evaluative language (positive/negative framing)
        self.evaluative = {
            'positive': ['benefit', 'opportunity', 'success', 'progress', 'improve',
                        'effective', 'efficient', 'sustainable', 'innovative', 'clean'],
            'negative': ['risk', 'threat', 'challenge', 'problem', 'barrier',
                        'harmful', 'ineffective', 'unsustainable', 'polluting', 'dirty']
        }
    
    def analyze_modal_verbs(self, text: str) -> Dict:
        """
        Analyze modal verb usage to understand policy directives.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with modal verb counts and proportions
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        modal_counts = {}
        total_modals = 0
        
        for category, modals in self.modals.items():
            count = sum(words.count(modal) for modal in modals) # generator expression
            modal_counts[category] = count
            total_modals += count
        
        # Calculate proportions
        modal_proportions = {
            category: count / total_modals if total_modals > 0 else 0
            for category, count in modal_counts.items()
        }
        
        # Dominant modal type
        if total_modals > 0:
            dominant_modal = max(modal_proportions.items(), key=lambda x: x[1])
        else:
            dominant_modal = ('none', 0)
        
        return {
            'counts': modal_counts,
            'total_modals': total_modals,
            'proportions': modal_proportions,
            'dominant_type': dominant_modal[0],
            'dominant_proportion': dominant_modal[1]
        }
    
    def analyze_hedging_certainty(self, text: str) -> Dict:
        """
        Analyze hedging (uncertainty) vs certainty markers.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with hedging and certainty metrics
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count hedges
        hedge_count = sum(words.count(hedge) for hedge in self.hedges)
        
        # Count boosters (certainty)
        booster_count = sum(words.count(booster) for booster in self.boosters)
        
        # Calculate ratio
        total = hedge_count + booster_count
        if total > 0:
            certainty_ratio = booster_count / total
        else:
            certainty_ratio = 0.5  # Neutral if no markers
        
        # Interpretation
        if certainty_ratio > 0.6:
            stance = 'assertive'
        elif certainty_ratio < 0.4:
            stance = 'hedged'
        else:
            stance = 'balanced'
        
        return {
            'hedge_count': hedge_count,
            'booster_count': booster_count,
            'total_markers': total,
            'certainty_ratio': certainty_ratio,
            'stance': stance
        }
    
    def analyze_voice(self, text: str, sample_size: int = 100) -> Dict:
        """
        Analyze active vs passive voice usage.
        
        Args:
            text: Document text
            sample_size: Number of sentences to analyze (for speed)
            
        Returns:
            Dictionary with voice analysis
        """
        doc = self.nlp(text[:50000])  # Limit for spaCy processing
        
        passive_count = 0
        active_count = 0
        total_sentences = 0
        
        for sent in list(doc.sents)[:sample_size]:
            total_sentences += 1
            
            # Detect passive voice (auxiliary + past participle)
            is_passive = False
            for token in sent:
                if token.dep_ == 'auxpass':  # Passive auxiliary
                    is_passive = True
                    break
            
            if is_passive:
                passive_count += 1
            else:
                active_count += 1
        
        passive_ratio = passive_count / total_sentences if total_sentences > 0 else 0
        
        # Interpretation
        if passive_ratio > 0.4:
            voice_style = 'predominantly_passive'
        elif passive_ratio > 0.2:
            voice_style = 'balanced'
        else:
            voice_style = 'predominantly_active'
        
        return {
            'passive_count': passive_count,
            'active_count': active_count,
            'total_analyzed': total_sentences,
            'passive_ratio': passive_ratio,
            'voice_style': voice_style
        }
    
    def analyze_temporal_framing(self, text: str) -> Dict:
        """
        Analyze temporal orientation (past, present, future focus).
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with temporal analysis
        """
        text_lower = text.lower()
        
        temporal_counts = {}
        total_temporal = 0
        
        for frame, markers in self.temporal.items():
            count = sum(text_lower.count(marker) for marker in markers)
            temporal_counts[frame] = count
            total_temporal += count
        
        # Calculate proportions
        temporal_proportions = {
            frame: count / total_temporal if total_temporal > 0 else 0
            for frame, count in temporal_counts.items()
        }
        
        # Dominant frame
        if total_temporal > 0:
            dominant_frame = max(temporal_proportions.items(), key=lambda x: x[1])
        else:
            dominant_frame = ('none', 0)
        
        return {
            'counts': temporal_counts,
            'total_markers': total_temporal,
            'proportions': temporal_proportions,
            'dominant_frame': dominant_frame[0],
            'dominant_proportion': dominant_frame[1]
        }
    
    def analyze_evaluative_language(self, text: str) -> Dict:
        """
        Analyze positive vs negative framing and sentiment.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with evaluative analysis
        """
        text_lower = text.lower()
        
        positive_count = sum(text_lower.count(word) for word in self.evaluative['positive'])
        negative_count = sum(text_lower.count(word) for word in self.evaluative['negative'])
        
        total_evaluative = positive_count + negative_count
        
        if total_evaluative > 0:
            positivity_ratio = positive_count / total_evaluative
        else:
            positivity_ratio = 0.5
        
        # Framing interpretation
        if positivity_ratio > 0.6:
            framing = 'positive'
        elif positivity_ratio < 0.4:
            framing = 'negative'
        else:
            framing = 'balanced'
        
        # Overall sentiment using TextBlob
        blob = TextBlob(text[:5000])  # Limit for speed
        sentiment = blob.sentiment
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_evaluative': total_evaluative,
            'positivity_ratio': positivity_ratio,
            'framing': framing,
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity
        }
    
    def analyze_stakeholder_agency(self, text: str) -> Dict:
        """
        Analyze which organizations/stakeholders are positioned as active agents.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with agency analysis
        """
        doc = self.nlp(text[:50000])  # Limit for processing
        
        # Track organizations as subjects (agents)
        org_as_subject = Counter()
        org_as_object = Counter()
        
        for sent in doc.sents:
            for token in sent:
                if token.ent_type_ == 'ORG':
                    # Check if organization is subject (agent)
                    if 'subj' in token.dep_:
                        org_as_subject[token.text] += 1
                    # Check if organization is object (recipient)
                    elif 'obj' in token.dep_:
                        org_as_object[token.text] += 1
        
        return {
            'orgs_as_subjects': dict(org_as_subject.most_common(10)),
            'orgs_as_objects': dict(org_as_object.most_common(10)),
            'total_agent_mentions': sum(org_as_subject.values()),
            'total_recipient_mentions': sum(org_as_object.values())
        }
    
    def analyze_all_documents(self) -> pd.DataFrame:
        """
        Run comprehensive discourse analysis on all documents.
        
        Returns:
            DataFrame with discourse analysis results
        """
        print(f"\nAnalyzing discourse patterns in {len(self.doc_info)} documents...")
        
        results = []
        
        for i, (text, info) in enumerate(zip(self.documents, self.doc_info), 1):
            doc_id = info['doc_id']
            filename = info['filename']
            
            print(f"\n[{i}/{len(self.doc_info)}] Analyzing: {filename}")
            
            # Run all analyses
            modal_analysis = self.analyze_modal_verbs(text)
            hedge_analysis = self.analyze_hedging_certainty(text)
            voice_analysis = self.analyze_voice(text)
            temporal_analysis = self.analyze_temporal_framing(text)
            evaluative_analysis = self.analyze_evaluative_language(text)
            agency_analysis = self.analyze_stakeholder_agency(text)
            
            # Compile results
            result = {
                'doc_id': doc_id,
                'filename': filename,
                'document_type': info['doc_type'],
                'year': info['year'],
                
                # Modal verbs
                'dominant_modal': modal_analysis['dominant_type'],
                'obligation_ratio': modal_analysis['proportions']['obligation'],
                'possibility_ratio': modal_analysis['proportions']['possibility'],
                
                # Hedging/Certainty
                'certainty_ratio': hedge_analysis['certainty_ratio'],
                'stance': hedge_analysis['stance'],
                
                # Voice
                'passive_ratio': voice_analysis['passive_ratio'],
                'voice_style': voice_analysis['voice_style'],
                
                # Temporal
                'dominant_temporal': temporal_analysis['dominant_frame'],
                'future_ratio': temporal_analysis['proportions']['future'],
                'present_ratio': temporal_analysis['proportions']['present'],
                
                # Evaluative
                'framing': evaluative_analysis['framing'],
                'positivity_ratio': evaluative_analysis['positivity_ratio'],
                'sentiment_polarity': evaluative_analysis['sentiment_polarity'],
                
                # Full analysis objects (for detailed inspection)
                'modal_analysis': modal_analysis,
                'hedge_analysis': hedge_analysis,
                'voice_analysis': voice_analysis,
                'temporal_analysis': temporal_analysis,
                'evaluative_analysis': evaluative_analysis,
                'agency_analysis': agency_analysis
            }
            
            results.append(result)
            
            print(f"  ✓ Stance: {result['stance']}, "
                  f"Voice: {result['voice_style']}, "
                  f"Framing: {result['framing']}")
        
        return pd.DataFrame(results)
    
    def visualize_discourse_patterns(self, results_df: pd.DataFrame):
        """Generate visualizations of discourse patterns."""
        print("\nGenerating discourse visualizations...")
        
        # 1. Modal verb distribution by document type
        fig1 = px.box(
            results_df,
            x='document_type',
            y='obligation_ratio',
            title='Obligation Language by Document Type',
            labels={'obligation_ratio': 'Proportion of Obligation Modals'}
        )
        fig1.write_html(self.discourse_dir / "modal_by_type.html")
        
        # 2. Certainty vs Hedging
        fig2 = px.scatter(
            results_df,
            x='certainty_ratio',
            y='passive_ratio',
            color='document_type',
            hover_data=['filename'],
            title='Certainty vs Passive Voice',
            labels={'certainty_ratio': 'Certainty', 'passive_ratio': 'Passive Voice'}
        )
        fig2.write_html(self.discourse_dir / "certainty_vs_voice.html")
        
        # 3. Temporal orientation
        fig3 = px.bar(
            results_df,
            x='filename',
            y=['future_ratio', 'present_ratio'],
            title='Temporal Orientation by Document',
            labels={'value': 'Proportion', 'variable': 'Time Frame'}
        )
        fig3.update_xaxes(tickangle=45)
        fig3.write_html(self.discourse_dir / "temporal_orientation.html")
        
        print("✓ Visualizations saved")
    
    def save_results(self, results_df: pd.DataFrame):
        """Save discourse analysis results in dashboard-friendly formats."""
        print("\nSaving results...")
        
        # 1. Save main results as CSV (for dashboard table)
        output_df = results_df.drop(columns=[col for col in results_df.columns 
                                             if col.endswith('_analysis')])
        output_df.to_csv(self.discourse_dir / "discourse_analysis.csv", index=False)
        
        # 2. Save detailed JSON (for dashboard)
        results_dict = results_df.to_dict('records')
        with open(self.discourse_dir / "discourse_analysis_detailed.json", 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # 3. Save summary statistics (for dashboard overview)
        summary_stats = {
            'total_documents': len(results_df),
            'avg_certainty_ratio': float(results_df['certainty_ratio'].mean()),
            'avg_passive_ratio': float(results_df['passive_ratio'].mean()),
            'avg_future_ratio': float(results_df['future_ratio'].mean()),
            'avg_positivity_ratio': float(results_df['positivity_ratio'].mean()),
            'stance_distribution': results_df['stance'].value_counts().to_dict(),
            'framing_distribution': results_df['framing'].value_counts().to_dict(),
            'voice_distribution': results_df['voice_style'].value_counts().to_dict(),
            'dominant_modal_distribution': results_df['dominant_modal'].value_counts().to_dict(),
            'temporal_distribution': results_df['dominant_temporal'].value_counts().to_dict()
        }
        
        with open(self.discourse_dir / "discourse_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # 4. Save by document type (for dashboard comparison)
        type_comparison = results_df.groupby('document_type').agg({
            'certainty_ratio': 'mean',
            'passive_ratio': 'mean',
            'future_ratio': 'mean',
            'positivity_ratio': 'mean',
            'obligation_ratio': 'mean'
        }).reset_index()
        
        type_comparison.to_csv(self.discourse_dir / "discourse_by_type.csv", index=False)
        type_comparison.to_json(self.discourse_dir / "discourse_by_type.json", orient='records', indent=2)
        
        # 5. Save summary report (human-readable)
        with open(self.discourse_dir / "discourse_summary.txt", 'w') as f:
            f.write("DISCOURSE ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Documents analyzed: {len(results_df)}\n\n")
            
            f.write("AVERAGE DISCOURSE CHARACTERISTICS:\n")
            f.write(f"  Certainty ratio: {results_df['certainty_ratio'].mean():.2f}\n")
            f.write(f"  Passive voice ratio: {results_df['passive_ratio'].mean():.2f}\n")
            f.write(f"  Future orientation: {results_df['future_ratio'].mean():.2f}\n")
            f.write(f"  Positivity ratio: {results_df['positivity_ratio'].mean():.2f}\n\n")
            
            f.write("STANCE DISTRIBUTION:\n")
            for stance, count in results_df['stance'].value_counts().items():
                f.write(f"  {stance}: {count}\n")
            
            f.write("\nFRAMING DISTRIBUTION:\n")
            for framing, count in results_df['framing'].value_counts().items():
                f.write(f"  {framing}: {count}\n")
            
            f.write("\nVOICE STYLE DISTRIBUTION:\n")
            for voice, count in results_df['voice_style'].value_counts().items():
                f.write(f"  {voice}: {count}\n")
        
        print(f"✓ Results saved to {self.discourse_dir}/")
        print(f"  Dashboard files:")
        print(f"    - discourse_analysis.csv (main data)")
        print(f"    - discourse_statistics.json (overview stats)")
        print(f"    - discourse_by_type.csv/json (comparison)")
        print(f"    - discourse_analysis_detailed.json (full data)")
        print(f"  Reference:")
        print(f"    - discourse_summary.txt (human-readable)")


if __name__ == "__main__":
    """Main execution function."""
    print("="*80)
    print("PHASE 7: DISCOURSE ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = PolicyDiscourseAnalyzer("preprocessed_output")
    
    # Run analysis
    results_df = analyzer.analyze_all_documents()
    
    # Generate visualizations
    analyzer.visualize_discourse_patterns(results_df)
    
    # Save results
    analyzer.save_results(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("KEY DISCOURSE PATTERNS")
    print("="*80)
    
    print(f"\nMost assertive document:")
    most_assertive = results_df.loc[results_df['certainty_ratio'].idxmax()]
    print(f"  {most_assertive['filename']} (certainty: {most_assertive['certainty_ratio']:.2f})")
    
    print(f"\nMost hedged document:")
    most_hedged = results_df.loc[results_df['certainty_ratio'].idxmin()]
    print(f"  {most_hedged['filename']} (certainty: {most_hedged['certainty_ratio']:.2f})")
    
    print(f"\nMost future-oriented:")
    most_future = results_df.loc[results_df['future_ratio'].idxmax()]
    print(f"  {most_future['filename']} (future: {most_future['future_ratio']:.2f})")
    
    print("\n" + "="*80)
    print("✅ Phase 7 Complete!")
    print("="*80)
    print("\nOutputs saved to: preprocessed_output/discourse_analysis/")
    print("  - discourse_analysis.csv")
    print("  - discourse_analysis_detailed.json")
    print("  - discourse_summary.txt")
    print("  - Visualizations (HTML files)")