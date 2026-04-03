# Individual Contribution Report — Manan Koradiya

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** Data Engineering Lead

---

## Summary
I was responsible for identifying the right datasets and building the data ingestion and sub-sampling strategy. This was one of the most critical early decisions in the project — a poor dataset choice would have undermined our entire pipeline. I also built the EDA tooling to validate that the data was suitable for our signal detection goals.

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
  - 5,000 posts from safe control classes for baseline separation
- Ensured the subset was balanced enough to prevent class imbalance from distorting our embedding space.

### 3. Exploratory Data Analysis (`src/eda.py`)
- Wrote EDA scripts to analyze class distributions, label frequencies, and text length statistics across all datasets.
- Confirmed that raw addiction and alcoholism classes were semantically rich enough to serve as the core of our vector database.

## Deliverables
- `src/build_vector_db.py`
- `src/eda.py`
- `data/stratified_subset.csv`
- `data/eda_results.txt`
- Dataset and Data Strategy section of `REPORT.md`
