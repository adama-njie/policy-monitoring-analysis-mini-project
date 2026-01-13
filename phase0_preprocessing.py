"""
Phase 0: Improved Policy Document Preprocessing Pipeline
=========================================================
Enhanced version with better PDF handling and segmentation
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import PyPDF2
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

class PolicyDocumentPreprocessor:
    """
    Comprehensive preprocessing pipeline for policy documents.
    Enhanced with better PDF extraction and paragraph segmentation.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing raw policy documents
            output_dir: Directory to save processed outputs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "processed_texts").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "segments").mkdir(exist_ok=True)
        
        self.documents = []
        self.metadata_catalog = []
        self.failed_documents = []
        
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file with multiple fallback methods.
        """
        text = ""
        
        # Method 1: Try PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt('')  # Try empty password
                    except:
                        print(f"  ⚠️  PDF is encrypted and cannot be decrypted")
                        return ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"  ⚠️  Error on page {page_num + 1}: {str(e)}")
                        
        except Exception as e:
            print(f"  ⚠️  PyPDF2 extraction failed: {str(e)}")
        
        # Check if we got meaningful text
        if len(text.strip()) < 100:
            print(f"  ⚠️  Minimal text extracted ({len(text.strip())} chars)")
            print(f"  ℹ️  This might be a scanned PDF requiring OCR")
            return text.strip()
        
        return text
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            doc = Document(file_path)
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            print(f"  ⚠️  Error extracting DOCX: {str(e)}")
        return text
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"  ⚠️  Error reading TXT: {str(e)}")
                return ""
    
    def extract_text(self, file_path: Path) -> str:
        """Route to appropriate extraction method based on file type."""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            print(f"  ⚠️  Unsupported file format: {extension}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        
        # Remove page numbers (common patterns)
        text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s+\|.*?Page', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Lines with only numbers
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'^.*?confidential.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?draft.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up spacing again after removals
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def extract_metadata(self, file_path: Path, text: str) -> Dict:
        """
        Extract metadata from document.
        """
        filename = file_path.stem
        
        # Extract year (from filename or text)
        year_match = re.search(r'(19|20)\d{2}', filename)
        if not year_match and text:
            year_match = re.search(r'\b(19|20)\d{2}\b', text[:1000])
        year = year_match.group(0) if year_match else "unknown"
        
        # Detect document type based on keywords
        text_lower = text.lower() if text else ""
        doc_type = "policy"
        
        type_keywords = {
            'regulation': ['regulation', 'regulatory framework'],
            'directive': ['directive', 'eu directive'],
            'strategy': ['strategy', 'strategic plan'],
            'legislation': ['act', 'legislation', 'statute'],
            'guideline': ['guideline', 'guidance'],
            'report': ['report', 'assessment'],
            'white_paper': ['white paper'],
            'briefing': ['briefing', 'policy brief']
        }
        
        for dtype, keywords in type_keywords.items():
            if any(keyword in text_lower[:2000] for keyword in keywords):
                doc_type = dtype
                break
        
        # Detect issuing authority
        authority = "unknown"
        if text:
            authority_patterns = [
                r'(?:issued by|published by)[:\s]+([A-Z][a-zA-Z\s&]+?)(?:\n|\.)',
                r'([A-Z][a-zA-Z\s&]+)\s+©',
                r'European Commission',
                r'IPCC',
                r'WWF'
            ]
            
            for pattern in authority_patterns:
                match = re.search(pattern, text[:2000])
                if match:
                    authority = match.group(0).strip() if pattern in authority_patterns[2:] else match.group(1).strip()
                    break
        
        # Word and character counts
        if text:
            try:
                words = word_tokenize(text)
                word_count = len(words)
            except:
                words = text.split()
                word_count = len(words)
            
            char_count = len(text)
            
            # Sentence count
            try:
                sentences = sent_tokenize(text)
                sentence_count = len(sentences)
            except:
                sentences = re.split(r'[.!?]+', text)
                sentence_count = len([s for s in sentences if s.strip()])
        else:
            word_count = 0
            char_count = 0
            sentence_count = 0
        
        return {
            'doc_id': f"doc_{len(self.metadata_catalog) + 1:03d}",
            'filename': filename,
            'file_extension': file_path.suffix,
            'year': year,
            'document_type': doc_type,
            'issuing_authority': authority,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': round(word_count / sentence_count, 2) if sentence_count > 0 else 0,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'extraction_quality': 'good' if word_count > 100 else 'poor'
        }
    
    def segment_document(self, text: str) -> Dict:
        """
        Segment document into structural units with improved paragraph detection.
        """
        if not text:
            return {
                'sections': [],
                'section_count': 0,
                'paragraphs': [],
                'paragraph_count': 0,
                'sentences': [],
                'sentence_count': 0,
                'avg_paragraph_length': 0
            }
        
        # Improved paragraph splitting
        # Split on double newlines (standard paragraph breaks)
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]  # Minimum 30 chars
        
        # If we only got 1 paragraph, try alternative splitting
        if len(paragraphs) <= 1:
            # Try splitting on sentence endings followed by capital letters
            paragraphs = re.split(r'\.(\s+[A-Z])', text)
            # Rejoin the capital letters
            rejoined = []
            for i in range(0, len(paragraphs)-1, 2):
                para = paragraphs[i] + '.'
                if i+1 < len(paragraphs):
                    para += paragraphs[i+1]
                if len(para.strip()) > 30:
                    rejoined.append(para.strip())
            paragraphs = rejoined if rejoined else [text]
        
        # Split into sections (look for numbered headings or clear section markers)
        section_pattern = r'\n\s*(?:\d+\.|\d+\)|\b(?:Chapter|Section|Article)\s+\d+)\s+[A-Z][^\n]{10,100}\n'
        sections = re.split(section_pattern, text)
        sections = [s.strip() for s in sections if len(s.strip()) > 100]
        
        # If no sections found, treat whole document as one section
        if len(sections) == 0:
            sections = [text]
        
        # Split into sentences
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'sections': sections,
            'section_count': len(sections),
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs),
            'sentences': sentences,
            'sentence_count': len(sentences),
            'avg_paragraph_length': round(sum(len(p.split()) for p in paragraphs) / len(paragraphs)) if paragraphs else 0
        }
    
    def process_all_documents(self):
        """Process all documents in the input directory."""
        print(f"Starting preprocessing of documents in: {self.input_dir}")
        print("=" * 80)
        
        # Get all supported files
        supported_extensions = ['.pdf', '.docx', '.txt']
        files = [f for f in self.input_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_extensions]
        
        if not files:
            print(f"No supported files found in {self.input_dir}")
            return
        
        print(f"Found {len(files)} documents to process\n")
        
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] Processing: {file_path.name}")
            
            # Extract text
            raw_text = self.extract_text(file_path)
            
            if not raw_text or len(raw_text.strip()) < 50:
                print(f"  ⚠️  Skipped (insufficient text extracted: {len(raw_text.strip())} chars)")
                self.failed_documents.append({
                    'filename': file_path.name,
                    'reason': 'insufficient_text',
                    'chars_extracted': len(raw_text.strip())
                })
                continue
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Extract metadata
            metadata = self.extract_metadata(file_path, cleaned_text)
            
            # Segment document
            segments = self.segment_document(cleaned_text)
            
            # Save processed text
            text_output = self.output_dir / "processed_texts" / f"{metadata['doc_id']}.txt"
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save segments
            segment_output = self.output_dir / "segments" / f"{metadata['doc_id']}_segments.json"
            with open(segment_output, 'w', encoding='utf-8') as f:
                json.dump({
                    'doc_id': metadata['doc_id'],
                    'filename': metadata['filename'],
                    'section_count': segments['section_count'],
                    'paragraph_count': segments['paragraph_count'],
                    'sentence_count': segments['sentence_count'],
                    'sample_paragraphs': segments['paragraphs'][:5],  # First 5
                    'sample_sentences': segments['sentences'][:10]     # First 10
                }, f, indent=2)
            
            # Store metadata
            self.metadata_catalog.append(metadata)
            
            # Store document info
            self.documents.append({
                'doc_id': metadata['doc_id'],
                'text': cleaned_text,
                'segments': segments
            })
            
            print(f"  ✓ Processed: {metadata['word_count']:,} words, {segments['paragraph_count']} paragraphs, {segments['sentence_count']} sentences")
        
        print("\n" + "=" * 80)
        
        if self.failed_documents:
            print(f"⚠️  {len(self.failed_documents)} document(s) failed to process:")
            for failed in self.failed_documents:
                print(f"  - {failed['filename']}: {failed['reason']}")
            print()
        
        print("Preprocessing complete!")
        
    def save_metadata_catalog(self):
        """Save the metadata catalog."""
        # Save as JSON
        json_path = self.output_dir / "metadata" / "metadata_catalog.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_catalog, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / "metadata" / "metadata_catalog.csv"
        df = pd.DataFrame(self.metadata_catalog)
        df.to_csv(csv_path, index=False)
        
        # Save failed documents list
        if self.failed_documents:
            failed_path = self.output_dir / "metadata" / "failed_documents.json"
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_documents, f, indent=2)
        
        print(f"Metadata catalog saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
        if self.failed_documents:
            print(f"  - {failed_path}")
    
    def generate_statistics_report(self):
        """Generate comprehensive statistics report."""
        if not self.metadata_catalog:
            print("No documents processed yet!")
            return
        
        df = pd.DataFrame(self.metadata_catalog)
        
        # Filter out poor quality extractions for statistics
        df_good = df[df['extraction_quality'] == 'good']
        
        report = {
            'summary': {
                'total_documents_attempted': len(df) + len(self.failed_documents),
                'successfully_processed': len(df),
                'failed_processing': len(self.failed_documents),
                'good_quality_extractions': len(df_good),
                'poor_quality_extractions': len(df) - len(df_good),
                'total_words': int(df['word_count'].sum()),
                'total_characters': int(df['char_count'].sum()),
                'total_sentences': int(df['sentence_count'].sum()),
                'avg_words_per_doc': round(df_good['word_count'].mean(), 2) if len(df_good) > 0 else 0,
                'avg_sentences_per_doc': round(df_good['sentence_count'].mean(), 2) if len(df_good) > 0 else 0
            },
            'document_types': df['document_type'].value_counts().to_dict(),
            'years': df['year'].value_counts().to_dict(),
            'authorities': df['issuing_authority'].value_counts().to_dict(),
            'word_count_stats': {
                'min': int(df_good['word_count'].min()) if len(df_good) > 0 else 0,
                'max': int(df_good['word_count'].max()) if len(df_good) > 0 else 0,
                'median': int(df_good['word_count'].median()) if len(df_good) > 0 else 0,
                'std_dev': round(df_good['word_count'].std(), 2) if len(df_good) > 0 else 0
            },
            'extraction_quality': df['extraction_quality'].value_counts().to_dict()
        }
        
        # Save report
        report_path = self.output_dir / "preprocessing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("PREPROCESSING STATISTICS REPORT")
        print("=" * 80)
        print(f"\n📊 SUMMARY")
        print(f"  Documents Attempted: {report['summary']['total_documents_attempted']}")
        print(f"  Successfully Processed: {report['summary']['successfully_processed']}")
        print(f"  Failed Processing: {report['summary']['failed_processing']}")
        print(f"  Good Quality: {report['summary']['good_quality_extractions']}")
        print(f"  Poor Quality: {report['summary']['poor_quality_extractions']}")
        print(f"  Total Words: {report['summary']['total_words']:,}")
        print(f"  Total Sentences: {report['summary']['total_sentences']:,}")
        print(f"  Avg Words/Doc: {report['summary']['avg_words_per_doc']:,.0f}")
        
        print(f"\n📁 DOCUMENT TYPES")
        for doc_type, count in report['document_types'].items():
            print(f"  {doc_type.replace('_', ' ').title()}: {count}")
        
        print(f"\n📅 YEARS")
        for year, count in sorted(report['years'].items()):
            print(f"  {year}: {count}")
        
        print(f"\n📈 WORD COUNT STATISTICS (Good Quality Docs)")
        print(f"  Min: {report['word_count_stats']['min']:,}")
        print(f"  Max: {report['word_count_stats']['max']:,}")
        print(f"  Median: {report['word_count_stats']['median']:,}")
        print(f"  Std Dev: {report['word_count_stats']['std_dev']:,.0f}")
        
        if self.failed_documents:
            print(f"\n⚠️  FAILED DOCUMENTS")
            for failed in self.failed_documents:
                print(f"  - {failed['filename']}: {failed['reason']} ({failed.get('chars_extracted', 0)} chars)")
        
        print(f"\nReport saved to: {report_path}")
        print("=" * 80)
        
        return report


# Example usage
if __name__ == "__main__":
    # Set up paths
    INPUT_DIR = "raw_documents"
    OUTPUT_DIR = "preprocessed_output"
    
    # Initialize preprocessor
    preprocessor = PolicyDocumentPreprocessor(INPUT_DIR, OUTPUT_DIR)
    
    # Process all documents
    preprocessor.process_all_documents()
    
    # Save metadata catalog
    preprocessor.save_metadata_catalog()
    
    # Generate statistics report
    preprocessor.generate_statistics_report()
    
    print("\n✅ Phase 0 Complete! Ready for Phase 1: Semantic Search")
    print("\nNext steps:")
    print("1. Review failed documents (if any) - they may need OCR or decryption")
    print("2. Check segment quality in preprocessed_output/segments/")
    print("3. Verify metadata accuracy in preprocessed_output/metadata/")