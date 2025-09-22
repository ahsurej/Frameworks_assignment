
# cord19_analysis.py
# CORD-19 Dataset Analysis: Parts 1-3 (Data Loading, Cleaning, Analysis & Visualization)
# This script loads metadata.csv, explores/cleans the data, performs analysis, and generates visualizations.
# Run this FIRST: python cord19_analysis.py
# Outputs: Console stats, cleaned_metadata.csv, and 4 PNG plots (pubs_over_time.png, top_journals.png, title_wordcloud.png, source_distribution.png)
# Author: [Your Name] - Assignment Submission
# Dataset: CORD-19 metadata.csv (place in same folder)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import numpy as np

# Set plot style for better visuals
plt.style.use('default')  # Use 'seaborn' if available, else default
sns.set_palette("husl")

# =====================================
# PART 1: DATA LOADING AND BASIC EXPLORATION
# =====================================
def load_and_explore_data():
    print("=== PART 1: Data Loading and Basic Exploration ===")
    
    # Load CSV (sample 10K rows for speed; remove .sample() for full ~1M rows)
    df = pd.read_csv('metadata.csv', low_memory=False).sample(n=10000, random_state=42)
    
    print(f"Dataset loaded! Shape: {df.shape}")
    
    # First few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Dimensions
    print(f"\nDimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Missing values (key columns)
    important_cols = ['title', 'abstract', 'publish_year', 'journal', 'authors']
    missing = df[important_cols].isnull().sum()
    print("\nMissing values in key columns:")
    print(missing)
    
    # Numerical stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumerical stats:")
        print(df[numeric_cols].describe())
    
    return df

# =====================================
# PART 2: DATA CLEANING AND PREPARATION
# =====================================
def clean_and_prepare_data(df):
    print("\n=== PART 2: Data Cleaning and Preparation ===")
    
    # High missing columns (>50%)
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    print("Columns with >50% missing:")
    print(high_missing)
    
    # Handle missing: Drop no-title rows
    initial_len = len(df)
    df_clean = df.dropna(subset=['title'])
    print(f"Dropped {initial_len - len(df_clean)} rows with missing titles.")
    
    # Clean publish_year
    if 'publish_year' in df_clean.columns:
        df_clean['publish_year'] = pd.to_numeric(df_clean['publish_year'], errors='coerce')
        median_year = df_clean['publish_year'].median()
        df_clean['publish_year'].fillna(median_year, inplace=True)
        df_clean['publish_year'] = df_clean['publish_year'].astype(int)
        print(f"Filled publish_year with median: {median_year}")
    
    # Fill abstract for word count
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(x.split()))
    print(f"Mean abstract word count: {df_clean['abstract_word_count'].mean():.2f}")
    
    # Save cleaned
    df_clean.to_csv('cleaned_metadata.csv', index=False)
    print(f"Cleaned data saved (shape: {df_clean.shape})")
    
    return df_clean

# =====================================
# PART 3: DATA ANALYSIS AND VISUALIZATION
# =====================================
def analyze_and_visualize(df_clean):
    print("\n=== PART 3: Data Analysis and Visualization ===")
    
    # Analysis: Papers by year (2019-2022)
    df_clean['publish_year'] = df_clean['publish_year'].astype(int)
    year_counts = df_clean[df_clean['publish_year'].between(2019, 2022)]['publish_year'].value_counts().sort_index()
    print("\nPapers by year (2019-2022):")
    print(year_counts)
    
    # Top journals
    df_clean['journal'] = df_clean['journal'].fillna('Unknown')
    top_journals = df_clean['journal'].value_counts().head(10)
    print("\nTop 10 journals:")
    print(top_journals)
    
    # Frequent words in titles
    def clean_text(text):
        if pd.isna(text):
            return []
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return text.split()
    
    all_titles = [word for title in df_clean['title'] for word in clean_text(title)]
    word_freq = Counter(all_titles)
    common_words = word_freq.most_common(20)
    print("\nTop words in titles (excluding common stop words):")
    stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'is', 'on', 'with'}
    for word, count in common_words:
        if word not in stop_words and len(word) > 3:
            print(f"{word}: {count}")
    
    # Viz 1: Publications over time
    plt.figure(figsize=(10, 6))
    year_counts.plot(kind='line', marker='o')
    plt.title('Publications Over Time (2019-2022)')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.grid(True)
    plt.savefig('pubs_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Viz 2: Top journals bar chart
    plt.figure(figsize=(12, 6))
    top_journals.plot(kind='bar')
    plt.title('Top 10 Journals')
    plt.xlabel('Journal')
    plt.ylabel('Papers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Viz 3: Word cloud
    title_text = ' '.join(df_clean['title'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(title_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Titles')
    plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Viz 4: Journal count distribution (log scale)
    journal_counts = df_clean['journal'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(journal_counts + 1), bins=20, edgecolor='black')
    plt.title('Distribution of Papers by Journal (Log Scale)')
    plt.xlabel('Log10(Papers per Journal + 1)')
    plt.ylabel('Frequency')
    plt.savefig('source_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run all parts
if __name__ == "__main__":
    df = load_and_explore_data()
    df_clean = clean_and_prepare_data(df)
    analyze_and_visualize(df_clean)
    print("\n=== Parts 1-3 Complete! Check console, cleaned_metadata.csv, and PNG files. ===")
