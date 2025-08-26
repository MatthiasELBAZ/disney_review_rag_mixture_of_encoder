# Stampli Home Assignment 🏰

A data analysis project for Disneyland reviews using LLM-powered feature extraction and RAG-based querying.

## Project Overview

This project analyzes Disneyland review data through:
1. **Exploratory Data Analysis (EDA)** - Understanding rating patterns, visitor demographics, and temporal trends
2. **LLM Feature Extraction** - Using GPT-4o-mini to extract structured insights from review text
3. **RAG System** - Combining Superlinked embeddings with LangChain for natural language querying

## Setup

### 1. Install Dependencies with UV

```bash
# Install uv package manager if not already installed
pip install uv

# Sync dependencies from pyproject.toml
uv sync
```

### 2. Environment Variables

Create a `.env` file from the example:
```bash
cp env_example.txt .env
```

Add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Data

The project uses two main datasets in `disney-reviews/`:

- **`DisneylandReviews.csv`** (31MB) - Original dataset with 42k+ reviews from 2010-2019
- **`disney_reviews_fr_analysis.csv`** (255KB) - Pre-processed French reviews with LLM-extracted features

### Dataset Features

**Original columns:**
- Review_ID, Rating, Reviewer_Location, Review_Text, Branch, Year_Month

**LLM-extracted features (41 total columns):**
- Sentiment analysis (overall + aspect-based for attractions, food, staff, etc.)
- Visitor demographics (type, origin, children presence)
- Trip context (frequency, duration, special occasions)
- Purchase mentions (tickets, fast passes, souvenirs)
- Engagement signals (attractions mentioned, recommendation intent)

## Running the Notebook

Open and run `stampli_hs.ipynb` sequentially. 

### ⚠️ Important - Do NOT Run Cell 38

**Cell 38 contains the LLM feature extraction code that:**
- Processes reviews through OpenAI's GPT-4o-mini
- Takes hours to complete for large datasets
- Consumes significant API tokens
- **Results are already saved in `disney_reviews_fr_analysis.csv`**

### Notebook Structure

1. **Cells 1-37**: EDA and setup (safe to run)
2. **Cell 38**: ❌ **DO NOT RUN** - LLM extraction (pre-computed)
3. **Cells 39-56**: Analysis with pre-processed data (safe to run)

### Key Sections

- **EDA (Cells 1-35)**: Rating distributions, temporal patterns, location analysis
- **Feature Extraction (Cells 36-39)**: LLM pipeline description (don't run cell 38)
- **Advanced Analysis (Cells 40-51)**: Working with extracted features
- **RAG System (Cells 52-56)**: Natural language querying with Superlinked + LangChain

## Technologies Used

- **pandas/matplotlib/seaborn** - Data analysis and visualization
- **LangChain + OpenAI** - LLM-powered feature extraction
- **Superlinked** - Mixed-type embeddings and semantic search
- **UV** - Modern Python dependency management

## Key Insights

- Rating distribution skews positive (most visitors happy)
- European visitors tend to rate more critically than North Americans
- Disneyland Paris receives lower ratings than California/Hong Kong
- Long reviews correlate with negative sentiment
- LLM extraction enables rich analysis of unstructured text data 