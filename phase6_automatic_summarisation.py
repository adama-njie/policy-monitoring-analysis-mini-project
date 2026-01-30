"""
Phase 6: Automatic Summarization
==================================
Generate extractive and abstractive summaries of policy documents.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions

class PolicySummarizer:
    """
    Multi-strategy policy document summarization.
    Combines extractive and abstractive approaches.
    """
    
    def __init__(self, preprocessed_dir: str):
        """
        Initialize summarizer.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        
        print("Loading models...")
        
        # Load sentence transformer for extractive summarization
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load abstractive summarization model (BART or T5)
        print("  Loading abstractive summarization model (this may take a moment)...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # Good for long documents
            device=-1  # CPU (use 0 for GPU)
        )
        
        print("✓ Models loaded")
        
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
                    'year': meta['year'],
                    'word_count': meta['word_count']
                })
            except FileNotFoundError:
                print(f"  ⚠️ Warning: Text file not found for {meta['filename']}")
                continue
        
        print(f"✓ Loaded {len(self.documents)} documents")
        
        # Create output directory
        self.summary_dir = self.preprocessed_dir / "summaries"
        self.summary_dir.mkdir(exist_ok=True)
        
    def extractive_summarization(self, text: str, num_sentences: int = 5) -> Dict:
        """
        Extractive summarization using sentence embeddings and similarity.
        Selects most representative sentences from document.
        
        Args:
            text: Document text
            num_sentences: Number of sentences to extract
            
        Returns:
            Dictionary with summary and metadata
        """
        # Split into sentences (with fallback)
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"  ⚠️ NLTK tokenization failed, using simple fallback: {e}")
            # Simple fallback: split on periods, question marks, exclamation marks
            import re
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'method': 'extractive',
                'num_sentences': len(sentences),
                'compression_ratio': 1.0,
                'selected_indices': list(range(len(sentences)))
            }
        
        # Generate sentence embeddings
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)
        
        # Calculate sentence scores (centrality in similarity network)
        sentence_scores = similarity_matrix.sum(axis=1)
        
        # Get top N sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        
        # Sort by original order (maintain document flow)
        top_indices_sorted = sorted(top_indices)
        
        # Extract sentences
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        summary = ' '.join(summary_sentences)
        
        # Calculate compression ratio
        compression_ratio = len(summary.split()) / len(text.split())
        
        return {
            'summary': summary,
            'method': 'extractive',
            'num_sentences': num_sentences,
            'compression_ratio': compression_ratio,
            'selected_indices': top_indices_sorted,  # Already a list
            'sentence_scores': sentence_scores.tolist()
        }
    
    def abstractive_summarization(self, text: str, max_length: int = 150, 
                                  min_length: int = 50) -> Dict:
        """
        Abstractive summarization using BART transformer.
        Generates new text that captures key points.
        
        Args:
            text: Document text
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Dictionary with summary and metadata
        """
        # BART works better with shorter inputs - take only first portion
        words = text.split()
        original_word_count = len(words)
        
        # Use only first 500 words to avoid model errors
        if len(words) > 500:
            text = ' '.join(words[:500])
            truncated = True
        else:
            truncated = False
        
        try:
            # Generate summary with more conservative parameters
            summary_output = self.summarizer(
                text,
                max_length=max_length,
                min_length=min(min_length, max_length - 10),  # Ensure min < max
                do_sample=False,
                truncation=True
            )
            
            summary = summary_output[0]['summary_text']
            
            # Calculate compression ratio against original full text
            compression_ratio = len(summary.split()) / original_word_count
            
            return {
                'summary': summary,
                'method': 'abstractive',
                'max_length': max_length,
                'min_length': min_length,
                'compression_ratio': compression_ratio,
                'truncated': truncated
            }
        
        except Exception as e:
            print(f"  ⚠️ Abstractive summarization failed: {e}")
            # Fallback: use first few sentences as summary
            sentences = text.split('. ')[:3]
            fallback_summary = '. '.join(sentences) + '.'
            
            return {
                'summary': fallback_summary,
                'method': 'abstractive',
                'compression_ratio': len(fallback_summary.split()) / original_word_count,
                'error': str(e),
                'truncated': truncated
            }
    
    def hybrid_summarization(self, text: str, extractive_sentences: int = 10,
                           abstractive_max_length: int = 150) -> Dict:
        """
        Hybrid approach: Extract key sentences, then abstractively summarize them.
        
        Args:
            text: Document text
            extractive_sentences: Number of sentences to extract first
            abstractive_max_length: Max length of final abstractive summary
            
        Returns:
            Dictionary with summary and metadata
        """
        # Step 1: Extractive summarization to reduce length
        extractive_result = self.extractive_summarization(text, extractive_sentences)
        extracted_text = extractive_result['summary']
        
        # Step 2: Abstractive summarization on extracted text
        abstractive_result = self.abstractive_summarization(
            extracted_text,
            max_length=abstractive_max_length,
            min_length=50
        )
        
        return {
            'summary': abstractive_result['summary'],
            'method': 'hybrid',
            'extractive_sentences': extractive_sentences,
            'abstractive_max_length': abstractive_max_length,
            'compression_ratio': len(abstractive_result['summary'].split()) / len(text.split()),
            'intermediate_summary': extracted_text
        }
    
    def summarize_all_documents(self, method: str = 'extractive', 
                               num_sentences: int = 5,
                               max_length: int = 150):
        """
        Summarize all documents in corpus.
        
        Args:
            method: 'extractive', 'abstractive', or 'hybrid'
            num_sentences: For extractive/hybrid methods
            max_length: For abstractive/hybrid methods
            
        Returns:
            List of summary dictionaries
        """
        print(f"\nSummarizing {len(self.doc_info)} documents using {method} method...")
        
        summaries = []
        
        for i, (text, info) in enumerate(zip(self.documents, self.doc_info), 1):
            doc_id = info['doc_id']
            filename = info['filename']
            
            print(f"[{i}/{len(self.doc_info)}] Summarizing: {filename}")
            
            # Generate summary based on method
            if method == 'extractive':
                result = self.extractive_summarization(text, num_sentences)
            elif method == 'abstractive':
                result = self.abstractive_summarization(text, max_length)
            elif method == 'hybrid':
                result = self.hybrid_summarization(text, num_sentences, max_length)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add metadata
            result['doc_id'] = doc_id
            result['filename'] = filename
            result['document_type'] = info['doc_type']
            result['year'] = info['year']
            result['original_length'] = info['word_count']
            result['summary_length'] = len(result['summary'].split())
            
            summaries.append(result)
            
            print(f"  ✓ Summary length: {result['summary_length']} words "
                  f"(compression: {result['compression_ratio']:.1%})")
        
        return summaries
    
    def generate_topic_summaries(self, topic_id: int, num_sentences: int = 5):
        """
        Generate a summary for all documents in a specific topic.
        
        Args:
            topic_id: Topic ID from Phase 4
            num_sentences: Sentences per document
            
        Returns:
            Combined topic summary
        """
        # Load topic assignments
        topic_file = self.preprocessed_dir / "topic_modeling" / "document_topics.csv"
        topics_df = pd.read_csv(topic_file)
        
        # Get documents in this topic
        topic_docs = topics_df[topics_df['topic'] == topic_id]
        
        print(f"\nGenerating summary for Topic {topic_id}")
        print(f"Documents in topic: {len(topic_docs)}")
        
        topic_summaries = []
        
        for _, row in topic_docs.iterrows():
            doc_id = row['doc_id']
            filename = row['filename']
            
            # Find document text in our loaded documents
            doc_idx = next((i for i, info in enumerate(self.doc_info) 
                          if info['doc_id'] == doc_id), None)
            
            if doc_idx is None:
                print(f"  ⚠️ Warning: Document {filename} not found in loaded documents")
                continue
            
            text = self.documents[doc_idx]
            
            # Summarize
            result = self.extractive_summarization(text, num_sentences)
            result['filename'] = filename
            topic_summaries.append(result)
        
        # Combine all summaries
        combined_summary = '\n\n'.join([
            f"**{s['filename']}**\n{s['summary']}" 
            for s in topic_summaries
        ])
        
        return {
            'topic_id': topic_id,
            'num_documents': len(topic_docs),
            'document_summaries': topic_summaries,
            'combined_summary': combined_summary
        }
    
    def save_summaries(self, summaries: List[Dict], method: str):
        """Save summaries to files in dashboard-friendly formats."""
        print(f"\nSaving summaries...")
        
        # Convert all numpy types to Python native types for JSON serialization
        import numpy as np
        
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        # Clean all summaries
        summaries = convert_to_native(summaries)
        
        # Save as JSON (for dashboard)
        json_path = self.summary_dir / f"summaries_{method}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2)
        
        # Save as CSV (for dashboard table view)
        csv_data = []
        for s in summaries:
            csv_data.append({
                'doc_id': s['doc_id'],
                'filename': s['filename'],
                'document_type': s['document_type'],
                'year': s['year'],
                'original_length': s['original_length'],
                'summary_length': s['summary_length'],
                'compression_ratio': s['compression_ratio'],
                'summary': s['summary']
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = self.summary_dir / f"summaries_{method}.csv"
        csv_df.to_csv(csv_path, index=False)
        
        # Save summary statistics (for dashboard overview)
        stats = {
            'method': method,
            'total_documents': len(summaries),
            'avg_original_length': int(np.mean([s['original_length'] for s in summaries])),
            'avg_summary_length': int(np.mean([s['summary_length'] for s in summaries])),
            'avg_compression_ratio': float(np.mean([s['compression_ratio'] for s in summaries])),
            'total_compression': sum([s['summary_length'] for s in summaries]) / sum([s['original_length'] for s in summaries])
        }
        
        stats_path = self.summary_dir / f"summary_statistics_{method}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        # Save human-readable text file (optional, for reference)
        txt_path = self.summary_dir / f"summaries_{method}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("POLICY DOCUMENT SUMMARIES\n")
            f.write("="*80 + "\n\n")
            f.write(f"Method: {method.upper()}\n")
            f.write(f"Total documents: {len(summaries)}\n")
            f.write(f"Average compression: {np.mean([s['compression_ratio'] for s in summaries]):.1%}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            for s in summaries:
                f.write(f"\n{'='*80}\n")
                f.write(f"Document: {s['filename']}\n")
                f.write(f"Type: {s['document_type']} | Year: {s['year']}\n")
                f.write(f"Original: {s['original_length']} words → Summary: {s['summary_length']} words\n")
                f.write(f"Compression: {s['compression_ratio']:.1%}\n")
                f.write(f"{'='*80}\n\n")
                f.write(s['summary'])
                f.write("\n\n")
        
        print(f"✓ Summaries saved to {self.summary_dir}/")
        print(f"  - {json_path.name} (for dashboard)")
        print(f"  - {csv_path.name} (for dashboard)")
        print(f"  - {stats_path.name} (for dashboard)")
        print(f"  - {txt_path.name} (for reference)")


if __name__ == "__main__":
    """Main execution function."""
    print("="*80)
    print("PHASE 6: AUTOMATIC SUMMARIZATION")
    print("="*80)
    
    # Initialize summarizer
    summarizer = PolicySummarizer("preprocessed_output")
    
    # Method 1: Extractive Summarization (Fast, maintains original wording)
    print("\n" + "="*80)
    print("METHOD 1: EXTRACTIVE SUMMARIZATION")
    print("="*80)
    extractive_summaries = summarizer.summarize_all_documents(
        method='extractive',
        num_sentences=5  # Extract 5 most important sentences
    )
    summarizer.save_summaries(extractive_summaries, 'extractive')
    
    # Method 2: Abstractive Summarization (Slower, generates new text)
    print("\n" + "="*80)
    print("METHOD 2: ABSTRACTIVE SUMMARIZATION")
    print("="*80)
    abstractive_summaries = summarizer.summarize_all_documents(
        method='abstractive',
        max_length=150  # Maximum 150 words
    )
    summarizer.save_summaries(abstractive_summaries, 'abstractive')
    
    # Method 3: Hybrid (Best of both - recommended)
    print("\n" + "="*80)
    print("METHOD 3: HYBRID SUMMARIZATION")
    print("="*80)
    hybrid_summaries = summarizer.summarize_all_documents(
        method='hybrid',
        num_sentences=10,  # First extract 10 sentences
        max_length=150      # Then abstractively summarize to 150 words
    )
    summarizer.save_summaries(hybrid_summaries, 'hybrid')
    
    print("\n" + "="*80)
    print("✅ Phase 6 Complete!")
    print("="*80)
    print("\nOutputs saved to: preprocessed_output/summaries/")
    print("  - summaries_extractive.json/csv/txt")
    print("  - summaries_abstractive.json/csv/txt")
    print("  - summaries_hybrid.json/csv/txt")
    print("\nNext: Phase 7 - Discourse Analysis")