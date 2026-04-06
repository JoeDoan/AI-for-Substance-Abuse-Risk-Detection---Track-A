# Individual Contribution Report — Manan Koradiya

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** Data Engineering Lead & Behavioral Analysis Engineer  
**Submission Date:** April 6, 2026

---

## Summary
I was responsible for identifying the right datasets, building the data ingestion and sub-sampling strategy, and implementing the Behavioral Pattern Analysis module (Core Task 2). This was one of the most critical early decisions in the project — a poor dataset choice would have undermined our entire pipeline. I also built the EDA tooling to validate data quality and the K-Means clustering module to address the challenge's requirement for behavioral theme discovery.

## Key Contributions

### 1. Dataset Research & Curation
- Identified and proposed the three-dataset ensemble strategy to capture a diverse range of social signals:
  1. `kamruzzaman-asif/reddit-mental-health-classification` (HuggingFace) — 1.1M posts, multi-class Reddit data
  2. `reihanenamdari/mental-health-corpus` (Kaggle) — 28K posts, binary distress labels
  3. Pre-approved KUC Kaggle hackathon dataset as a reference baseline
- Prepared the approval request email to Dr. Yugyung Lee covering all three datasets.

### 2. Stratified Sub-sampling (`src/build_vector_db.py`)
- Engineered the targeted sub-sampling logic to produce a focused 43,551-row subset from 1.1M rows:
  - All `addiction` (~7,750) and `alcoholism` (~5,530) posts
  - 5,000 posts from each of `depression`, `suicidewatch`, `anxiety`, `ptsd`, `mentalhealth`
  - 5,000 posts from safe control classes for baseline vector space separation
- Ensured the subset was balanced enough to prevent class imbalance from distorting our embedding space.

### 3. Exploratory Data Analysis (`src/eda.py`)
- Wrote EDA scripts to analyze class distributions, label frequencies, and text length statistics across all datasets.
- Confirmed that raw addiction and alcoholism classes were semantically rich enough to serve as the core of our vector database.
- Exported results to `data/eda_results.txt` for documentation.

### 4. Behavioral Pattern Analysis — K-Means Clustering (`src/topic_clustering.py`)
- Implemented the **Core Task 2 (Temporal and Behavioral Analysis)** module using embedding-based clustering, aligned with Track A's AI Modeling approach.
- Fetched all 43,551 document embeddings from ChromaDB in batches, filtered for 38,552 High Risk posts, and sub-sampled 5,000 for clustering efficiency.
- Applied **K-Means (k=8)** on the 384-dimensional all-MiniLM-L6-v2 embedding vectors using scikit-learn.
- Extracted **top TF-IDF keywords** per cluster to auto-label emerging behavioral themes:
  - Cluster 0: PTSD & Trauma (574 posts, 11.5%)
  - Cluster 2: Active Drug Addiction & Weed (654 posts, 13.1%)
  - Cluster 4: Alcohol Use & Sobriety Struggles (635 posts, 12.7%)
  - Cluster 5: Anxiety & Panic (668 posts, 13.4%)
  - Cluster 7: Suicidal Ideation & Crisis (457 posts, 9.1%)
- Exported clustering results to `data/cluster_results.csv` and generated PCA 2D scatter visualization.

## Deliverables
- `src/build_vector_db.py`
- `src/eda.py`
- `src/topic_clustering.py`
- `data/stratified_subset.csv`
- `data/eda_results.txt`
- `data/cluster_results.csv`
- Dataset, Data Strategy, and Behavioral Clustering sections of the final report
