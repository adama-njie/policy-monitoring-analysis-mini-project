"""
Phase 5: Interactive Policy Analysis Dashboard - FIXED
=======================================================
Streamlit dashboard integrating all analyses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Policy Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    """Load all analysis results."""
    base_path = Path("preprocessed_output")
    
    data = {}
    
    # Metadata
    with open(base_path / "metadata" / "metadata_catalog.json") as f:
        data['metadata'] = json.load(f)
    
    # Preprocessing stats
    try:
        with open(base_path / "preprocessing_report.json") as f:
            data['preprocessing'] = json.load(f)
    except (FileNotFoundError, KeyError):
        # Create from metadata if report doesn't exist or has wrong structure
        data['preprocessing'] = {
            'summary': {
                'total_documents': len(data['metadata']),
                'total_words': sum(m['word_count'] for m in data['metadata']),
                'total_sentences': sum(m['sentence_count'] for m in data['metadata']),
                'document_types': {},
                'years': {}
            }
        }
        # Count document types
        for m in data['metadata']:
            doc_type = m['document_type']
            year = m['year']
            data['preprocessing']['summary']['document_types'][doc_type] = \
                data['preprocessing']['summary']['document_types'].get(doc_type, 0) + 1
            data['preprocessing']['summary']['years'][year] = \
                data['preprocessing']['summary']['years'].get(year, 0) + 1
    
    # Similarity matrix
    data['similarity'] = pd.read_csv(base_path / "document_comparison" / "similarity_matrix.csv", index_col=0)
    
    # Comparison report
    with open(base_path / "document_comparison" / "comparison_report.json") as f:
        data['comparison'] = json.load(f)
    
    # Entities (use ultimate cleaned if available)
    entity_files = ['all_entities_ULTIMATE_CLEANED.csv', 'all_entities_CLEANED.csv', 'all_entities.csv']
    for entity_file in entity_files:
        entity_path = base_path / "named_entities" / entity_file
        if entity_path.exists():
            data['entities'] = pd.read_csv(entity_path)
            break
    
    # Entity statistics
    stat_files = ['entity_statistics_ULTIMATE_CLEANED.json', 'entity_statistics_CLEANED.json', 'entity_statistics.json']
    for stat_file in stat_files:
        stat_path = base_path / "named_entities" / stat_file
        if stat_path.exists():
            with open(stat_path) as f:
                data['entity_stats'] = json.load(f)
            break
    
    # Topics
    data['topic_info'] = pd.read_csv(base_path / "topic_modeling" / "topic_info.csv")
    data['doc_topics'] = pd.read_csv(base_path / "topic_modeling" / "document_topics.csv")
    
    return data

def show_overview(data):
    """Show overview dashboard."""
    st.markdown('<p class="main-header">📊 Policy Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics with safe access
    col1, col2, col3, col4 = st.columns(4)
    
    # Safely get values
    try:
        total_docs = data['preprocessing']['summary']['total_documents']
    except:
        total_docs = len(data['metadata'])
    
    try:
        total_words = data['preprocessing']['summary']['total_words']
    except:
        total_words = sum(m['word_count'] for m in data['metadata'])
    
    try:
        total_entities = data['entity_stats']['total_entities']
    except:
        total_entities = len(data['entities'])
    
    try:
        total_topics = len(data['topic_info']) - 1
    except:
        total_topics = len(data['topic_info'])
    
    with col1:
        st.metric(
            "Total Documents",
            total_docs,
            help="Number of policy documents analyzed"
        )
    
    with col2:
        st.metric(
            "Total Words",
            f"{total_words:,}",
            help="Total word count across all documents"
        )
    
    with col3:
        st.metric(
            "Entities Extracted",
            f"{total_entities:,}",
            help="Named entities identified"
        )
    
    with col4:
        st.metric(
            "Topics Discovered",
            total_topics,
            help="Latent topics identified by BERTopic"
        )
    
    st.markdown("---")
    
    # Document overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Document Types")
        try:
            doc_types_dict = data['preprocessing']['summary']['document_types']
        except:
            # Calculate from metadata
            doc_types_dict = {}
            for m in data['metadata']:
                dt = m['document_type']
                doc_types_dict[dt] = doc_types_dict.get(dt, 0) + 1
        
        doc_types = pd.DataFrame(
            list(doc_types_dict.items()),
            columns=['Type', 'Count']
        )
        fig = px.pie(doc_types, values='Count', names='Type', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Documents by Year")
        try:
            years_dict = data['preprocessing']['summary']['years']
        except:
            # Calculate from metadata
            years_dict = {}
            for m in data['metadata']:
                y = m['year']
                years_dict[y] = years_dict.get(y, 0) + 1
        
        years = pd.DataFrame(
            list(years_dict.items()),
            columns=['Year', 'Count']
        )
        years = years.sort_values('Year')
        fig = px.bar(years, x='Year', y='Count',
                     color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Document list
    st.subheader("📄 Document Catalog")
    
    metadata_df = pd.DataFrame(data['metadata'])
    display_df = metadata_df[['filename', 'document_type', 'year', 'word_count', 'sentence_count']]
    display_df.columns = ['Filename', 'Type', 'Year', 'Words', 'Sentences']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def show_similarity_analysis(data):
    """Show document similarity analysis."""
    st.header("🔗 Document Similarity Analysis")
    
    # Similarity heatmap
    st.subheader("Similarity Matrix")
    
    # FIX: Remove 'annot' parameter - Plotly doesn't support it
    # Use text_auto=True instead
    fig = px.imshow(
        data['similarity'],
        labels=dict(x="Document", y="Document", color="Similarity"),
        x=data['similarity'].columns,
        y=data['similarity'].index,
        color_continuous_scale='RdYlGn',
        aspect="auto",
        text_auto='.2f'  # This shows values with 2 decimal places
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top similar pairs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Most Similar Pairs")
        similar_pairs = data['comparison']['top_similar_pairs'][:10]
        
        for i, pair in enumerate(similar_pairs, 1):
            with st.expander(f"{i}. Similarity: {pair['similarity']:.3f}"):
                st.write(f"**Document 1:** {pair['doc1']}")
                st.write(f"**Document 2:** {pair['doc2']}")
                st.write(f"**Types:** {pair['doc1_type']} & {pair['doc2_type']}")
    
    with col2:
        st.subheader("⚠️ Most Divergent Pairs")
        divergent_pairs = data['comparison']['top_divergent_pairs'][:10]
        
        for i, pair in enumerate(divergent_pairs, 1):
            with st.expander(f"{i}. Similarity: {pair['similarity']:.3f}"):
                st.write(f"**Document 1:** {pair['doc1']}")
                st.write(f"**Document 2:** {pair['doc2']}")
                st.write(f"**Types:** {pair['doc1_type']} & {pair['doc2_type']}")
    
    st.markdown("---")
    
    # Coherence analysis
    st.subheader("📊 Coherence by Document Type")
    
    # Debug: Check if coherence_analysis exists
    try:
        if 'coherence_analysis' not in data['comparison']:
            st.warning("Coherence analysis not found. Run Phase 2 document comparison first.")
        elif 'within_type' not in data['comparison']['coherence_analysis']:
            st.warning("Within-type coherence data not found.")
        else:
            coherence_data = []
            for doc_type, stats in data['comparison']['coherence_analysis']['within_type'].items():
                coherence_data.append({
                    'Document Type': doc_type.capitalize(),
                    'Average Similarity': stats['avg_similarity'],
                    'Document Count': stats['count']
                })
            
            if len(coherence_data) > 0:
                coherence_df = pd.DataFrame(coherence_data)
                
                fig = px.bar(
                    coherence_df,
                    x='Document Type',
                    y='Average Similarity',
                    color='Document Count',
                    text='Average Similarity',
                    color_continuous_scale='Viridis'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No document types with multiple documents for coherence analysis.")
    except Exception as e:
        st.error(f"Error displaying coherence analysis: {e}")
        st.info("This may indicate Phase 2 results need to be regenerated.")

def show_entity_analysis(data):
    """Show named entity analysis."""
    st.header("🏢 Named Entity Analysis")
    
    # Entity statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Entities", f"{data['entity_stats']['total_entities']:,}")
    
    with col2:
        st.metric("Entity Types", len(data['entity_stats']['entity_types']))
    
    with col3:
        total_unique = sum(data['entity_stats']['unique_entities_by_type'].values())
        st.metric("Unique Entities", f"{total_unique:,}")
    
    st.markdown("---")
    
    # Entity type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Entity Type Distribution")
        entity_types = pd.DataFrame(
            list(data['entity_stats']['entity_types'].items()),
            columns=['Type', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig = px.bar(
            entity_types,
            x='Type',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top Entities by Type")
        
        entity_type = st.selectbox(
            "Select Entity Type",
            options=list(data['entity_stats']['top_entities_by_type'].keys())
        )
        
        top_entities = data['entity_stats']['top_entities_by_type'][entity_type][:15]
        
        entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
        
        fig = px.bar(
            entities_df,
            y='Entity',
            x='Count',
            orientation='h',
            color='Count',
            color_continuous_scale='Viridis',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Entity browser
    st.subheader("🔍 Entity Browser")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_type = st.multiselect(
            "Filter by Type",
            options=data['entities']['label'].unique(),
            default=[]
        )
        
        search_term = st.text_input("Search Entity", "")
    
    with col2:
        filtered_entities = data['entities'].copy()
        
        if filter_type:
            filtered_entities = filtered_entities[filtered_entities['label'].isin(filter_type)]
        
        if search_term:
            filtered_entities = filtered_entities[
                filtered_entities['text'].str.contains(search_term, case=False, na=False)
            ]
        
        # Group by entity
        entity_counts = filtered_entities.groupby(['text', 'label']).size().reset_index(name='count')
        entity_counts = entity_counts.sort_values('count', ascending=False).head(50)
        
        st.dataframe(
            entity_counts.rename(columns={'text': 'Entity', 'label': 'Type', 'count': 'Count'}),
            use_container_width=True,
            hide_index=True
        )

def show_topic_analysis(data):
    """Show topic modeling analysis."""
    st.header("🎯 Topic Modeling Analysis")
    
    # Topic overview
    st.subheader("📊 Topic Overview")
    
    topic_info = data['topic_info'][data['topic_info']['Topic'] != -1]  # Exclude outliers
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Topics Discovered", len(topic_info))
    
    with col2:
        st.metric("Largest Topic", topic_info['Count'].max())
    
    with col3:
        outliers = data['topic_info'][data['topic_info']['Topic'] == -1]
        outlier_count = outliers['Count'].values[0] if len(outliers) > 0 else 0
        st.metric("Outliers", outlier_count)
    
    st.markdown("---")
    
    # Topic distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Topic Sizes")
        
        fig = px.bar(
            topic_info,
            x='Topic',
            y='Count',
            color='Count',
            text='Count',
            color_continuous_scale='Viridis',
            labels={'Count': 'Documents', 'Topic': 'Topic ID'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Topic Distribution")
        
        fig = px.pie(
            topic_info,
            values='Count',
            names='topic_label',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Topic details
    st.subheader("🔍 Topic Details")
    
    for _, row in topic_info.iterrows():
        with st.expander(f"Topic {row['Topic']}: {row['Name'][:60]}..."):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Documents", row['Count'])
                
            with col2:
                st.write("**Top Words:**")
                # Parse representation string
                words = row['Name'].split('_')[1:6]  # Get first 5 words
                st.write(", ".join(words))
            
            # Show documents in this topic
            topic_docs = data['doc_topics'][data['doc_topics']['topic'] == row['Topic']]
            topic_docs = topic_docs.sort_values('probability', ascending=False)
            
            st.write("**Documents:**")
            for _, doc in topic_docs.iterrows():
                st.write(f"- {doc['filename']} ({doc['doc_type']}, {doc['year']}) - Prob: {doc['probability']:.3f}")
    
    st.markdown("---")
    
    # Topic-Document Type relationship
    st.subheader("📊 Topics by Document Type")
    
    topic_type_pivot = pd.crosstab(
        data['doc_topics']['topic'],
        data['doc_topics']['doc_type']
    )
    
    fig = px.imshow(
        topic_type_pivot,
        labels=dict(x="Document Type", y="Topic", color="Count"),
        color_continuous_scale='Blues',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

def show_search(data):
    """Show semantic search interface."""
    st.header("🔍 Semantic Search")
    
    st.info("ℹ️ Search functionality requires loading the search model. This feature connects to Phase 1 results.")
    
    query = st.text_input("Enter your search query:", placeholder="e.g., hydrogen production and distribution")
    
    if query:
        st.warning("⚠️ Full search implementation requires the sentence transformer model and FAISS index from Phase 1. This is a simplified version.")
        
        # Simple keyword-based search as fallback
        metadata_df = pd.DataFrame(data['metadata'])
        
        # Load documents
        results = []
        for meta in data['metadata']:
            doc_path = Path("preprocessed_output/processed_texts") / f"{meta['doc_id']}.txt"
            if doc_path.exists():
                # FIX: Specify encoding='utf-8' and add error handling
                try:
                    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().lower()
                    
                    # Simple keyword matching
                    query_words = query.lower().split()
                    matches = sum(1 for word in query_words if word in text)
                    
                    if matches > 0:
                        results.append({
                            'filename': meta['filename'],
                            'doc_type': meta['document_type'],
                            'year': meta['year'],
                            'relevance': matches / len(query_words)
                        })
                except Exception as e:
                    st.warning(f"Could not read {meta['filename']}: {e}")
                    continue
        
        if results:
            results_df = pd.DataFrame(results).sort_values('relevance', ascending=False).head(10)
            
            st.subheader("Search Results")
            
            for _, row in results_df.iterrows():
                with st.expander(f"{row['filename']} - Relevance: {row['relevance']:.2%}"):
                    st.write(f"**Type:** {row['doc_type']}")
                    st.write(f"**Year:** {row['year']}")
        else:
            st.warning("No results found.")

def main():
    """Main dashboard application."""
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_all_data()
    
    # Sidebar
    st.sidebar.title("📊 Policy Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "📈 Overview",
            "🔗 Document Similarity",
            "🏢 Named Entities",
            "🎯 Topic Modeling",
            "🔍 Semantic Search"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Safe statistics display with error handling
    try:
        total_docs = data['preprocessing']['summary']['total_documents']
    except (KeyError, TypeError):
        total_docs = len(data['metadata'])
    
    try:
        total_words = data['preprocessing']['summary']['total_words']
    except (KeyError, TypeError):
        total_words = sum(m['word_count'] for m in data['metadata'])
    
    try:
        total_entities = data['entity_stats']['total_entities']
    except (KeyError, TypeError):
        total_entities = len(data['entities'])
    
    try:
        total_topics = len(data['topic_info']) - 1
    except:
        total_topics = len(data['topic_info'])
    
    st.sidebar.info(
        f"""
        **Corpus Statistics:**
        - Documents: {total_docs}
        - Words: {total_words:,}
        - Entities: {total_entities:,}
        - Topics: {total_topics}
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export Options:**")
    
    if st.sidebar.button("📥 Download Summary Report"):
        st.sidebar.success("Report generation available in full version")
    
    # Main content
    if page == "📈 Overview":
        show_overview(data)
    elif page == "🔗 Document Similarity":
        show_similarity_analysis(data)
    elif page == "🏢 Named Entities":
        show_entity_analysis(data)
    elif page == "🎯 Topic Modeling":
        show_topic_analysis(data)
    elif page == "🔍 Semantic Search":
        show_search(data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Policy Analysis Dashboard | PhD Mini-Project"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()