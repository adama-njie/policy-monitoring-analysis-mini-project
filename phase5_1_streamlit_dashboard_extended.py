"""
Complete Policy Analysis Dashboard - Phases 0-7
================================================
Integrating all analysis phases in one dashboard.
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    """Load all analysis results including phases 6-7."""
    base_path = Path("preprocessed_output")
    
    data = {}
    
    # Metadata
    with open(base_path / "metadata" / "metadata_catalog.json") as f:
        data['metadata'] = json.load(f)
    
    # Preprocessing stats
    try:
        with open(base_path / "preprocessing_report.json") as f:
            data['preprocessing'] = json.load(f)
    except:
        data['preprocessing'] = {'summary': {'total_documents': len(data['metadata'])}}
    
    # Similarity
    data['similarity'] = pd.read_csv(base_path / "document_comparison" / "similarity_matrix.csv", index_col=0)
    
    with open(base_path / "document_comparison" / "comparison_report.json") as f:
        data['comparison'] = json.load(f)
    
    # Entities
    for entity_file in ['all_entities_ULTIMATE_CLEANED.csv', 'all_entities_CLEANED.csv', 'all_entities.csv']:
        if (base_path / "named_entities" / entity_file).exists():
            data['entities'] = pd.read_csv(base_path / "named_entities" / entity_file)
            break
    
    for stat_file in ['entity_statistics_ULTIMATE_CLEANED.json', 'entity_statistics_CLEANED.json']:
        if (base_path / "named_entities" / stat_file).exists():
            with open(base_path / "named_entities" / stat_file) as f:
                data['entity_stats'] = json.load(f)
            break
    
    # Topics
    data['topic_info'] = pd.read_csv(base_path / "topic_modeling" / "topic_info.csv")
    data['doc_topics'] = pd.read_csv(base_path / "topic_modeling" / "document_topics.csv")
    
    # NEW: Load summarization data (Phase 6)
    summary_path = base_path / "summaries"
    if summary_path.exists():
        for method in ['hybrid', 'extractive', 'abstractive']:
            summary_file = summary_path / f"summaries_{method}.csv"
            stats_file = summary_path / f"summary_statistics_{method}.json"
            
            if summary_file.exists():
                data['summaries'] = pd.read_csv(summary_file)
                data['summary_method'] = method
                
                if stats_file.exists():
                    with open(stats_file) as f:
                        data['summary_stats'] = json.load(f)
                break
    
    # NEW: Load discourse analysis data (Phase 7)
    discourse_path = base_path / "discourse_analysis"
    if discourse_path.exists():
        discourse_file = discourse_path / "discourse_analysis.csv"
        stats_file = discourse_path / "discourse_statistics.json"
        by_type_file = discourse_path / "discourse_by_type.json"
        
        if discourse_file.exists():
            data['discourse'] = pd.read_csv(discourse_file)
        
        if stats_file.exists():
            with open(stats_file) as f:
                data['discourse_stats'] = json.load(f)
        
        if by_type_file.exists():
            with open(by_type_file) as f:
                data['discourse_by_type'] = json.load(f)
    
    return data

def show_overview(data):
    """Show overview dashboard."""
    st.markdown('<p class="main-header">📊 Policy Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        st.metric("Entities Extracted", f"{total_entities:,}")
    
    with col4:
        st.metric("Topics Discovered", total_topics)
    
    st.markdown("---")
    
    # Document overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Document Types")
        try:
            doc_types_dict = data['preprocessing']['summary']['document_types']
        except:
            doc_types_dict = {}
            for m in data['metadata']:
                dt = m['document_type']
                doc_types_dict[dt] = doc_types_dict.get(dt, 0) + 1
        
        doc_types = pd.DataFrame(list(doc_types_dict.items()), columns=['Type', 'Count'])
        fig = px.pie(doc_types, values='Count', names='Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Documents by Year")
        try:
            years_dict = data['preprocessing']['summary']['years']
        except:
            years_dict = {}
            for m in data['metadata']:
                y = m['year']
                years_dict[y] = years_dict.get(y, 0) + 1
        
        years = pd.DataFrame(list(years_dict.items()), columns=['Year', 'Count']).sort_values('Year')
        fig = px.bar(years, x='Year', y='Count', color='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Document list
    st.subheader("📄 Document Catalog")
    metadata_df = pd.DataFrame(data['metadata'])
    display_df = metadata_df[['filename', 'document_type', 'year', 'word_count', 'sentence_count']]
    display_df.columns = ['Filename', 'Type', 'Year', 'Words', 'Sentences']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_similarity_analysis(data):
    """Show document similarity analysis."""
    st.header("🔗 Document Similarity Analysis")
    
    st.subheader("Similarity Matrix")
    fig = px.imshow(
        data['similarity'],
        labels=dict(x="Document", y="Document", color="Similarity"),
        x=data['similarity'].columns,
        y=data['similarity'].index,
        color_continuous_scale='RdYlGn',
        text_auto='.2f'
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Most Similar Pairs")
        for i, pair in enumerate(data['comparison']['top_similar_pairs'][:10], 1):
            with st.expander(f"{i}. Similarity: {pair['similarity']:.3f}"):
                st.write(f"**Document 1:** {pair['doc1']}")
                st.write(f"**Document 2:** {pair['doc2']}")
    
    with col2:
        st.subheader("⚠️ Most Divergent Pairs")
        for i, pair in enumerate(data['comparison']['top_divergent_pairs'][:10], 1):
            with st.expander(f"{i}. Similarity: {pair['similarity']:.3f}"):
                st.write(f"**Document 1:** {pair['doc1']}")
                st.write(f"**Document 2:** {pair['doc2']}")

def show_entity_analysis(data):
    """Show named entity analysis."""
    st.header("🏢 Named Entity Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Entities", f"{data['entity_stats']['total_entities']:,}")
    
    with col2:
        st.metric("Entity Types", len(data['entity_stats']['entity_types']))
    
    with col3:
        total_unique = sum(data['entity_stats']['unique_entities_by_type'].values())
        st.metric("Unique Entities", f"{total_unique:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Entity Type Distribution")
        entity_types = pd.DataFrame(
            list(data['entity_stats']['entity_types'].items()),
            columns=['Type', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig = px.bar(entity_types, x='Type', y='Count', color='Count', text='Count')
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
        
        fig = px.bar(entities_df, y='Entity', x='Count', orientation='h', color='Count', text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def show_topic_analysis(data):
    """Show topic modeling analysis."""
    st.header("🎯 Topic Modeling Analysis")
    
    topic_info = data['topic_info'][data['topic_info']['Topic'] != -1]
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Topic Sizes")
        fig = px.bar(topic_info, x='Topic', y='Count', color='Count', text='Count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Topic Distribution")
        fig = px.pie(topic_info, values='Count', names='topic_label')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Topic Details")
    for _, row in topic_info.iterrows():
        with st.expander(f"Topic {row['Topic']}: {row['Name'][:60]}..."):
            st.metric("Documents", row['Count'])
            topic_docs = data['doc_topics'][data['doc_topics']['topic'] == row['Topic']]
            topic_docs = topic_docs.sort_values('probability', ascending=False)
            
            st.write("**Documents:**")
            for _, doc in topic_docs.iterrows():
                st.write(f"- {doc['filename']} ({doc['doc_type']}, {doc['year']}) - Prob: {doc['probability']:.3f}")

def show_summarization(data):
    """Show document summarization results."""
    st.header("📝 Document Summaries")
    
    if 'summaries' not in data:
        st.warning("⚠️ Summaries not found. Run Phase 6 first.")
        st.code("python phase6_summarization.py")
        return
    
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = data.get('summary_stats', {})
    
    with col1:
        st.metric("Method", data.get('summary_method', 'Unknown').title())
    
    with col2:
        st.metric("Avg Original Length", f"{stats.get('avg_original_length', 0):,} words")
    
    with col3:
        st.metric("Avg Summary Length", f"{stats.get('avg_summary_length', 0):,} words")
    
    with col4:
        compression = stats.get('avg_compression_ratio', 0)
        st.metric("Avg Compression", f"{compression:.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Compression by Document Type")
        type_compression = data['summaries'].groupby('document_type')['compression_ratio'].mean().reset_index()
        fig = px.bar(type_compression, x='document_type', y='compression_ratio', 
                     color='compression_ratio', text='compression_ratio')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Summary Length Distribution")
        fig = px.histogram(data['summaries'], x='summary_length', nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📖 Browse Summaries")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_type = st.multiselect(
            "Filter by Type",
            options=data['summaries']['document_type'].unique(),
            default=[]
        )
    
    if filter_type:
        filtered_summaries = data['summaries'][data['summaries']['document_type'].isin(filter_type)]
    else:
        filtered_summaries = data['summaries']
    
    for _, row in filtered_summaries.iterrows():
        with st.expander(f"📄 {row['filename']} ({row['document_type']}, {row['year']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original", f"{row['original_length']:,} words")
            with col2:
                st.metric("Summary", f"{row['summary_length']:,} words")
            with col3:
                st.metric("Compression", f"{row['compression_ratio']:.1%}")
            
            st.markdown("**Summary:**")
            st.write(row['summary'])

def show_discourse_analysis(data):
    """Show discourse analysis results."""
    st.header("🗣️ Discourse Analysis")
    
    if 'discourse' not in data:
        st.warning("⚠️ Discourse analysis not found. Run Phase 7 first.")
        st.code("python phase7_discourse.py")
        return
    
    st.subheader("📊 Discourse Patterns Overview")
    
    stats = data.get('discourse_stats', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        certainty = stats.get('avg_certainty_ratio', 0)
        st.metric("Avg Certainty", f"{certainty:.2f}", help="0 = hedged, 1 = assertive")
    
    with col2:
        passive = stats.get('avg_passive_ratio', 0)
        st.metric("Avg Passive Voice", f"{passive:.1%}")
    
    with col3:
        future = stats.get('avg_future_ratio', 0)
        st.metric("Avg Future Focus", f"{future:.1%}")
    
    with col4:
        positivity = stats.get('avg_positivity_ratio', 0)
        st.metric("Avg Positivity", f"{positivity:.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Stance Distribution")
        stance_dist = stats.get('stance_distribution', {})
        if stance_dist:
            fig = px.pie(values=list(stance_dist.values()), names=list(stance_dist.keys()))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Framing Distribution")
        framing_dist = stats.get('framing_distribution', {})
        if framing_dist:
            fig = px.pie(values=list(framing_dist.values()), names=list(framing_dist.keys()))
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Discourse Patterns by Document Type")
    
    if 'discourse_by_type' in data:
        by_type_df = pd.DataFrame(data['discourse_by_type'])
        
        metric = st.selectbox(
            "Select Metric",
            ['certainty_ratio', 'passive_ratio', 'future_ratio', 'positivity_ratio', 'obligation_ratio']
        )
        
        fig = px.bar(by_type_df, x='document_type', y=metric, color=metric, text=metric)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📋 Document Discourse Profiles")
    
    for _, row in data['discourse'].head(10).iterrows():
        with st.expander(f"📄 {row['filename']} ({row['document_type']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Linguistic Style:**")
                st.write(f"- Stance: {row['stance']}")
                st.write(f"- Voice: {row['voice_style']}")
                st.write(f"- Framing: {row['framing']}")
            
            with col2:
                st.write("**Ratios:**")
                st.write(f"- Certainty: {row['certainty_ratio']:.2f}")
                st.write(f"- Passive: {row['passive_ratio']:.1%}")
                st.write(f"- Future: {row['future_ratio']:.1%}")
            
            with col3:
                st.write("**Modal Focus:**")
                st.write(f"- Dominant: {row['dominant_modal']}")
                st.write(f"- Temporal: {row['dominant_temporal']}")
                st.write(f"- Positivity: {row['positivity_ratio']:.1%}")

def show_search(data):
    """Show semantic search interface."""
    st.header("🔍 Semantic Search")
    st.info("ℹ️ Simple keyword-based search")
    
    query = st.text_input("Enter search query:", placeholder="e.g., hydrogen production")
    
    if query:
        results = []
        for meta in data['metadata']:
            doc_path = Path("preprocessed_output/processed_texts") / f"{meta['doc_id']}.txt"
            if doc_path.exists():
                try:
                    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().lower()
                    
                    query_words = query.lower().split()
                    matches = sum(1 for word in query_words if word in text)
                    
                    if matches > 0:
                        results.append({
                            'filename': meta['filename'],
                            'doc_type': meta['document_type'],
                            'year': meta['year'],
                            'relevance': matches / len(query_words)
                        })
                except:
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
    
    with st.spinner("Loading data..."):
        data = load_all_data()
    
    st.sidebar.title("📊 Policy Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "📈 Overview",
            "🔗 Document Similarity",
            "🏢 Named Entities",
            "🎯 Topic Modeling",
            "📝 Summaries",
            "🗣️ Discourse Analysis",
            "🔍 Semantic Search"
        ]
    )
    
    st.sidebar.markdown("---")
    
    try:
        total_docs = len(data['metadata'])
        total_words = sum(m['word_count'] for m in data['metadata'])
        total_entities = data.get('entity_stats', {}).get('total_entities', 0)
        total_topics = len(data['topic_info']) - 1
    except:
        total_docs = total_words = total_entities = total_topics = 0
    
    st.sidebar.info(
        f"""
        **Corpus Statistics:**
        - Documents: {total_docs}
        - Words: {total_words:,}
        - Entities: {total_entities:,}
        - Topics: {total_topics}
        """
    )
    
    # Route to appropriate page
    if page == "📈 Overview":
        show_overview(data)
    elif page == "🔗 Document Similarity":
        show_similarity_analysis(data)
    elif page == "🏢 Named Entities":
        show_entity_analysis(data)
    elif page == "🎯 Topic Modeling":
        show_topic_analysis(data)
    elif page == "📝 Summaries":
        show_summarization(data)
    elif page == "🗣️ Discourse Analysis":
        show_discourse_analysis(data)
    elif page == "🔍 Semantic Search":
        show_search(data)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Complete Policy Analysis Dashboard | Phases 0-7"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()