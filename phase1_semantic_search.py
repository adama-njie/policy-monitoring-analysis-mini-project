"""
Phase 1: Semantic Search Implementation
========================================
Build an intelligent semantic search engine for policy documents using embeddings.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

class PolicySemanticSearch:
    """
    Semantic search engine for policy documents using sentence transformers.
    """
    
    def __init__(self, preprocessed_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic search engine.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
            model_name: Name of the sentence transformer model to use
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.model_name = model_name
        
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        
        # Load processed documents
        self._load_documents()
        
    def _load_documents(self):
        """Load preprocessed documents and metadata."""
        print("\nLoading preprocessed documents...")
        
        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata" / "metadata_catalog.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load document texts
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
        
        print(f"✓ Loaded {len(self.documents)} documents")
    
    def create_embeddings(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Create embeddings for all documents.
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
        """
        print(f"\nCreating embeddings with chunking...")
        print(f"  Chunk size: {chunk_size} words")
        print(f"  Chunk overlap: {chunk_overlap} words")
        
        all_chunks = []
        chunk_metadata = []
        
        # Split documents into chunks
        for doc in tqdm(self.documents, desc="Chunking documents"):
            words = doc['text'].split()
            
            # Create overlapping chunks
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 50:  # Skip very small chunks
                    continue
                
                chunk_text = ' '.join(chunk_words)
                all_chunks.append(chunk_text)
                
                chunk_metadata.append({
                    'doc_id': doc['doc_id'],
                    'filename': doc['metadata']['filename'],
                    'document_type': doc['metadata']['document_type'],
                    'year': doc['metadata']['year'],
                    'chunk_index': len(chunk_metadata),
                    'chunk_start_word': i,
                    'chunk_text_preview': chunk_text[:200] + '...'
                })
        
        print(f"✓ Created {len(all_chunks)} chunks from {len(self.documents)} documents")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        self.embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        self.chunk_metadata = chunk_metadata
        self.chunks = all_chunks
        
        print(f"✓ Generated embeddings: shape {self.embeddings.shape}")
        
        # Save embeddings
        embeddings_dir = self.preprocessed_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        
        np.save(embeddings_dir / "embeddings.npy", self.embeddings)
        with open(embeddings_dir / "chunk_metadata.json", 'w') as f:
            json.dump(chunk_metadata, f, indent=2)
        
        print(f"✓ Embeddings saved to {embeddings_dir}")
    
    def load_embeddings(self):
        """Load pre-computed embeddings."""
        embeddings_dir = self.preprocessed_dir / "embeddings"
        
        print("Loading pre-computed embeddings...")
        self.embeddings = np.load(embeddings_dir / "embeddings.npy")
        
        with open(embeddings_dir / "chunk_metadata.json", 'r') as f:
            self.chunk_metadata = json.load(f)
        
        print(f"✓ Loaded embeddings: shape {self.embeddings.shape}")
    
    def build_index(self, index_type: str = 'flat'):
        """
        Build FAISS index for fast similarity search.
        
        Args:
            index_type: Type of index ('flat' for exact search, 'ivf' for approximate)
        """
        print(f"\nBuilding FAISS index (type: {index_type})...")
        
        dimension = self.embeddings.shape[1]
        
        if index_type == 'flat':
            # Exact search - slower but accurate
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            # Approximate search - faster for large datasets
            nlist = min(100, len(self.embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.embeddings)
        
        self.index.add(self.embeddings)
        
        print(f"✓ Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5, return_docs: bool = False) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            return_docs: If True, return full documents instead of chunks
            
        Returns:
            List of search results with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk_meta = self.chunk_metadata[idx]
            
            result = {
                'rank': i + 1,
                'similarity_score': float(1 / (1 + distance)),  # Convert distance to similarity
                'distance': float(distance),
                'doc_id': chunk_meta['doc_id'],
                'filename': chunk_meta['filename'],
                'document_type': chunk_meta['document_type'],
                'year': chunk_meta['year'],
                'chunk_index': chunk_meta['chunk_index'],
                'text_preview': chunk_meta['chunk_text_preview']
            }
            
            if return_docs:
                # Get full document text
                doc = next(d for d in self.documents if d['doc_id'] == chunk_meta['doc_id'])
                result['full_text'] = doc['text']
            
            results.append(result)
        
        return results
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Dict]:
        """
        Search within a specific document type.
        
        Args:
            query: Search query
            doc_type: Document type to filter by
            top_k: Number of results to return
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Get all results
        distances, indices = self.index.search(query_embedding, len(self.embeddings))
        
        # Filter by document type
        filtered_results = []
        for distance, idx in zip(distances[0], indices[0]):
            chunk_meta = self.chunk_metadata[idx]
            if chunk_meta['document_type'] == doc_type:
                filtered_results.append({
                    'rank': len(filtered_results) + 1,
                    'similarity_score': float(1 / (1 + distance)),
                    'distance': float(distance),
                    'doc_id': chunk_meta['doc_id'],
                    'filename': chunk_meta['filename'],
                    'document_type': chunk_meta['document_type'],
                    'year': chunk_meta['year'],
                    'chunk_index': chunk_meta['chunk_index'],
                    'text_preview': chunk_meta['chunk_text_preview']
                })
                
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def find_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id: ID of the reference document
            top_k: Number of similar documents to return
        """
        # Get document embedding (average of all its chunks)
        doc_chunks = [i for i, meta in enumerate(self.chunk_metadata) 
                     if meta['doc_id'] == doc_id]
        
        if not doc_chunks:
            return []
        
        doc_embedding = np.mean(self.embeddings[doc_chunks], axis=0, keepdims=True)
        
        # Search
        distances, indices = self.index.search(doc_embedding, len(self.embeddings))
        
        # Group by document and get unique documents
        doc_scores = {}
        for distance, idx in zip(distances[0], indices[0]):
            chunk_doc_id = self.chunk_metadata[idx]['doc_id']
            
            # Skip the same document
            if chunk_doc_id == doc_id:
                continue
            
            if chunk_doc_id not in doc_scores:
                doc_scores[chunk_doc_id] = []
            
            doc_scores[chunk_doc_id].append(float(distance))
        
        # Average scores for each document
        doc_avg_scores = {
            doc: np.mean(scores) 
            for doc, scores in doc_scores.items()
        }
        
        # Sort and get top k
        sorted_docs = sorted(doc_avg_scores.items(), key=lambda x: x[1])[:top_k]
        
        results = []
        for rank, (similar_doc_id, distance) in enumerate(sorted_docs, 1):
            doc_meta = next(d['metadata'] for d in self.documents 
                          if d['doc_id'] == similar_doc_id)
            
            results.append({
                'rank': rank,
                'similarity_score': float(1 / (1 + distance)),
                'distance': float(distance),
                'doc_id': similar_doc_id,
                'filename': doc_meta['filename'],
                'document_type': doc_meta['document_type'],
                'year': doc_meta['year'],
                'word_count': doc_meta['word_count']
            })
        
        return results
    
    def print_search_results(self, results: List[Dict], query: str = None):
        """Pretty print search results."""
        if query:
            print(f"\n{'='*80}")
            print(f"Search Query: {query}")
            print(f"{'='*80}")
        
        for result in results:
            print(f"\n[{result['rank']}] {result['filename']}")
            print(f"    Type: {result['document_type']} | Year: {result['year']}")
            print(f"    Similarity: {result['similarity_score']:.4f}")
            print(f"    Preview: {result['text_preview'][:150]}...")
            print(f"    Doc ID: {result['doc_id']}")
    
    def save_search_results(self, results: List[Dict], query: str, output_path: str):
        """Save search results to JSON file."""
        output = {
            'query': query,
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_results': len(results),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")


def main():
    """Main execution function."""
    # Initialize search engine
    search_engine = PolicySemanticSearch(
        preprocessed_dir="preprocessed_output",
        model_name='all-MiniLM-L6-v2'  # Fast and effective model
    )
    
    # Create embeddings (only needed once)
    print("\n" + "="*80)
    print("Step 1: Creating Embeddings")
    print("="*80)
    search_engine.create_embeddings(chunk_size=500, chunk_overlap=50)
    
    # Build search index
    print("\n" + "="*80)
    print("Step 2: Building Search Index")
    print("="*80)
    search_engine.build_index(index_type='flat')
    
    # Example searches
    print("\n" + "="*80)
    print("Step 3: Example Searches")
    print("="*80)
    
    # Search 1: Renewable energy subsidies
    print("\n--- Search 1: Renewable Energy Subsidies ---")
    results1 = search_engine.search("renewable energy subsidies and incentives", top_k=5)
    search_engine.print_search_results(results1, "renewable energy subsidies and incentives")
    
    # Search 2: Carbon pricing
    print("\n--- Search 2: Carbon Pricing Mechanisms ---")
    results2 = search_engine.search("carbon pricing mechanisms and emissions trading", top_k=5)
    search_engine.print_search_results(results2, "carbon pricing mechanisms and emissions trading")
    
    # Search 3: Hydrogen infrastructure
    print("\n--- Search 3: Hydrogen Infrastructure ---")
    results3 = search_engine.search("hydrogen production and distribution infrastructure", top_k=5)
    search_engine.print_search_results(results3, "hydrogen production and distribution infrastructure")
    
    # Search by document type
    print("\n--- Search 4: Strategies about Energy Transition ---")
    results4 = search_engine.search_by_document_type(
        "energy transition pathways", 
        doc_type="strategy",
        top_k=3
    )
    search_engine.print_search_results(results4, "energy transition pathways (strategies only)")
    
    # Find similar documents
    print("\n--- Search 5: Documents Similar to EU Hydrogen Strategy ---")
    # Find the EU Hydrogen Strategy doc_id
    eu_hydrogen_doc = next(d for d in search_engine.documents 
                          if 'hydrogen strategy' in d['metadata']['filename'].lower())
    
    similar_docs = search_engine.find_similar_documents(eu_hydrogen_doc['doc_id'], top_k=5)
    print(f"\nDocuments similar to: {eu_hydrogen_doc['metadata']['filename']}")
    print("="*80)
    
    for doc in similar_docs:
        print(f"\n[{doc['rank']}] {doc['filename']}")
        print(f"    Type: {doc['document_type']} | Year: {doc['year']}")
        print(f"    Similarity: {doc['similarity_score']:.4f}")
        print(f"    Words: {doc['word_count']:,}")
    
    # Save results
    results_dir = Path("preprocessed_output") / "search_results"
    results_dir.mkdir(exist_ok=True)
    
    search_engine.save_search_results(
        results1, 
        "renewable energy subsidies and incentives",
        results_dir / "search_renewable_energy.json"
    )
    
    print("\n" + "="*80)
    print("✅ Phase 1 Complete!")
    print("="*80)
    print("\nKey Features Implemented:")
    print("  ✓ Document embeddings with chunking")
    print("  ✓ FAISS-based similarity search")
    print("  ✓ Semantic query search")
    print("  ✓ Document type filtering")
    print("  ✓ Similar document finding")
    print("\nNext: Phase 2 - Document Comparison")


if __name__ == "__main__":
    main()