# cord19_app.py
# Streamlit App for CORD-19 Explorer (Part 4)
# Interactive dashboard with data views and visualizations.
# Run AFTER analysis script: streamlit run cord19_app.py
# Loads cleaned_metadata.csv and PNG files.

import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt  # For any in-app plots if needed

# Page setup
st.set_page_config(page_title="CORD-19 Explorer", page_icon="🦠", layout="wide")

# Main app
def main():
    st.title("🦠 CORD-19 Data Explorer")
    st.write("Explore COVID-19 research papers from the CORD-19 dataset. Use sidebar for navigation.")

    # Load cleaned data
    @st.cache_data
    def load_data():
        return pd.read_csv('cleaned_metadata.csv')

    df = load_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", ["Overview", "Data Sample", "Interactive Analysis", "Visualizations"])

    if page == "Overview":
        st.header("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Papers", len(df))
            st.metric("Years Covered", f"{df['publish_year'].min()} - {df['publish_year'].max()}")
        with col2:
            st.metric("Avg Abstract Words", f"{df['abstract_word_count'].mean():.0f}")
            st.metric("Top Year", df['publish_year'].mode().iloc[0])
        
        st.write("This app analyzes ~10K sampled papers (full dataset: 1M+). Key columns: title, journal, publish_year, abstract.")

    elif page == "Data Sample":
        st.header("🔍 Sample Data")
        # Show filtered sample
        sample_size = st.slider("Number of rows to show:", 5, 20, 10)
        st.dataframe(df[['title', 'authors', 'journal', 'publish_year', 'abstract_word_count']].head(sample_size), use_container_width=True)

    elif page == "Interactive Analysis":
        st.header("⚙️ Interactive Widgets")
        
        # Slider for year range (as in assignment example)
        min_year, max_year = st.slider("Select Year Range", 2019, 2022, (2019, 2022))
        filtered_df = df[(df['publish_year'] >= min_year) & (df['publish_year'] <= max_year)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Papers in Selected Years")
            st.write(len(filtered_df))
        with col2:
            # Dropdown for top journals
            top_j = filtered_df['journal'].value_counts().head(5)
            selected_journal = st.selectbox("Select a Journal:", top_j.index.tolist())
            st.subheader(f"Papers in {selected_journal}:")
            st.write(top_j[selected_journal])
        
        # Show filtered titles
        st.subheader("Sample Titles in Range")
        for title in filtered_df['title'].head(5).tolist():
            st.write(f"• {title}")

    elif page == "Visualizations":
        st.header("📈 Visualizations")
        
        # Display saved PNGs
        tab1, tab2, tab3, tab4 = st.tabs(["Publications Over Time", "Top Journals", "Word Cloud", "Journal Distribution"])
        
        with tab1:
            image = Image.open('pubs_over_time.png')
            st.image(image, caption="Number of Publications Over Time", use_column_width=True)
        
        with tab2:
            image = Image.open('top_journals.png')
            st.image(image, caption="Top 10 Journals", use_column_width=True)
        
        with tab3:
            image = Image.open('title_wordcloud.png')
            st.image(image, caption="Word Cloud of Paper Titles", use_column_width=True)
        
        with tab4:
            image = Image.open('source_distribution.png')
            st.image(image, caption="Distribution of Papers by Journal", use_column_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with Streamlit | Data: CORD-19")

if __name__ == "__main__":
    main()
