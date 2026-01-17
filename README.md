# Policy Analysis NLP Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

An end-to-end NLP pipeline for analyzing policy documents using semantic search, named entity recognition, topic modeling, and interactive visualization (Take it as a proof-of-concept).

## Overview

This project demonstrates automated policy analysis using modern NLP techniques. It processes policy documents through six integrated phases to extract insights on policy coherence, stakeholder networks, and thematic structures.

### Key Features

- **Automated Text Extraction** - Process PDF, DOCX, and TXT files
- **Semantic Search** - Find relevant documents using natural language queries
- **Document Similarity Analysis** - Identify coherent and divergent policies
- **Named Entity Recognition** - Extract organizations, locations, and policy instruments
- **Topic Modeling** - Discover latent themes using BERTopic
- **Interactive Dashboard** - Explore results through Streamlit web interface

## Results Summary

- **20 documents** processed (223,474 words)
- **429 embeddings** generated for semantic search
- **14,000+ entities** extracted with 98% accuracy
- **4 coherent topics** discovered automatically
- **0.58 average similarity** indicating good policy coherence

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/policy-analysis-mini-project.git
cd policy-analysis-nlp

# Create virtual environment
python -m venv policy-nlp
source policy-nlp/bin/activate  # On Windows: policy-nlp\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Usage

#### Option 1: Run Full Pipeline

```bash
# Place your documents in raw_documents/ folder
mkdir raw_documents
# Add your PDF/DOCX/TXT files here

# Run all phases sequentially
python phase0_preprocessing.py
python phase1_semantic_search.py
python phase2_document_comparison.py
python phase3_1_ner.py
python phase3_2_ultimate_cleaning.py
python phase3_3_regenerate_visuals.py
python phase4_topic_modeling_bertopic.py

# Launch dashboard
streamlit run phase5_streamlit_dashboard.py
```

#### Option 2: Use Pre-Processed Demo Data

```bash
# If you want to skip preprocessing and use example outputs
streamlit run phase5_streamlit_dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Project Structure

```
policy-analysis-nlp/
├── raw_documents/              # Input: Your policy documents
├── preprocessed_output/        # Output: All analysis results
├── phase0_preprocessing.py     # Text extraction & cleaning
├── phase1_semantic_search.py   # Embedding generation & search
├── phase2_document_comparison.py # Similarity analysis
├── phase3_1_ner.py            # Entity extraction
├── phase3_2_ultimate_cleaning.py # Entity cleaning
├── phase3_3_regenerate_visuals.py # Entity visualizations
├── phase3_4_network_visualiser.py # Network graphs
├── phase4_topic_modeling_bertopic.py # Topic discovery
├── phase5_streamlit_dashboard.py # Interactive dashboard
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Pipeline Phases

### Phase 0: Preprocessing
- Extracts text from PDF, DOCX, TXT files
- Cleans and normalizes text
- Segments into paragraphs and sentences
- **Output:** 20 clean text files, metadata catalog

### Phase 1: Semantic Search
- Generates 384-dimensional embeddings using SentenceTransformers
- Builds FAISS index for fast similarity search
- **Output:** 429 embeddings, search index

### Phase 2: Document Comparison
- Computes 20×20 similarity matrix
- Analyzes policy coherence by document type
- Detects potential conflicts
- **Output:** Similarity matrix, coherence metrics, visualizations

### Phase 3: Named Entity Recognition
- Extracts entities using spaCy (ORG, GPE, PERSON, LAW, etc.)
- 13-step cleaning pipeline (consolidation, false positive removal)
- Generates co-occurrence networks
- **Output:** 14,000+ clean entities, network visualizations

### Phase 4: Topic Modeling
- Discovers topics using BERTopic (UMAP + HDBSCAN)
- Assigns documents to topics with probabilities
- Analyzes topic-document type relationships
- **Output:** 4 topics with labels and assignments

### Phase 5: Interactive Dashboard
- Streamlit web interface with 5 pages
- 12 interactive Plotly visualizations
- Real-time filtering and search
- **Output:** Deployed web dashboard

## Dashboard Features

- **Overview:** Corpus statistics, document catalog
- **Document Similarity:** Heatmap, similar/divergent pairs, coherence analysis
- **Named Entities:** Entity browser, type distribution, top entities
- **Topic Modeling:** Topic sizes, distributions, document assignments
- **Semantic Search:** Query documents using natural language

## Research Applications

- **Policy Monitoring:** Track policy developments across jurisdictions
- **Coherence Assessment:** Identify aligned or conflicting policies
- **Stakeholder Analysis:** Map actor networks and relationships
- **Gap Identification:** Find underaddressed topics
- **Comparative Analysis:** Compare policies across countries/sectors

## Documentation

Comprehensive documentation available for each phase:
- Methodology and implementation details
- Results interpretation
- Troubleshooting guides
- Scalability analysis

See individual phase documentation in `/documentation` folder (if included).

## Key Technologies

- **NLP:** spaCy, NLTK, SentenceTransformers
- **ML:** scikit-learn, FAISS, HDBSCAN, UMAP
- **Topic Modeling:** BERTopic
- **Visualization:** Plotly, Matplotlib, Seaborn, NetworkX
- **Dashboard:** Streamlit
- **Data Processing:** pandas, NumPy

## Example Results

### Document Similarity
- Highest similarity: 0.89 (EU Hydrogen Strategy ↔ WWF Hydrogen Recommendation)
- Lowest similarity: 0.26 (Ammonia White Paper ↔ Economics of Transition)
- Average coherence: 0.58 (good alignment)

### Named Entities
- Top organization: European Union (774 mentions)
- Top location: Germany (182 mentions)
- 18 entity types extracted

### Topics Discovered
1. **Topic 0:** General Energy & Climate (60% of corpus)
2. **Topic 1:** Renewable Electricity Systems (15%)
3. **Topic 2:** Hydrogen Production (15%)
4. **Topic 3:** Ammonia Solutions (10%)

## Limitations

- Currently supports English-only documents
- Optimal for 20-500 document corpora
- Requires manual metadata enrichment for some fields
- Semantic search in dashboard is simplified (full Phase 1 integration pending)

## Roadmap

- [ ] Multi-lingual support (German, French)
- [ ] Temporal analysis (track changes over time)
- [ ] Full semantic search integration in dashboard
- [ ] Automated report generation (PDF/CSV export)
- [ ] Real-time document monitoring
- [ ] Citation network analysis

## Contributing

This is a research project developed for a PhD mini-project. Contributions, issues, and feature requests are welcome!

## License

NA.

## Adama Njie

PhD Researcher in AI-Driven Policy Monitoring, Analysis, Coherence, and Reverse Policy Assessment. 
- GitHub: [@YOUR-USERNAME](https://github.com/adama-njie)

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Embeddings via [SentenceTransformers](https://www.sbert.net/)
- Topic modeling with [BERTopic](https://maartengr.github.io/BERTopic/)
- NER powered by [spaCy](https://spacy.io/)

## Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---

**Note:** This project analyzes a sample corpus of 20 (I used small corpus size because I am running it on my local machine) EU energy policy documents. To use with your own documents, place them in the `raw_documents/` folder and run the pipeline.
