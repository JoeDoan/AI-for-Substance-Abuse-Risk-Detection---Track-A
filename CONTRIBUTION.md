# Team Contribution Report

**Project:** SubstanceWatch AI — NSF NRT Challenge 1: AI for Substance Abuse Risk Detection  
**Event:** UMKC Research-A-Thon 2026  
**Team Members:** Joe Doan · Manan Koradiya · Ruixuan Hou · Aditya Naredla  
**Submission Date:** April 6, 2026

---

## Contribution Summary

| Team Member | Primary Role | Key Deliverables |
|---|---|---|
| Joe Doan | Project Lead & AI Pipeline Architect | System design, ML classifier, pipeline evaluation, backend integration |
| Manan Koradiya | Data Engineering Lead | Dataset curation, stratified sampling, EDA, behavioral clustering |
| Ruixuan Hou | RAG & Vector DB Engineer | ChromaDB setup, embedding pipeline, retrieval logic, cluster visualization |
| Aditya Naredla | LLM Prompt Engineer & Frontend | Gemini prompt engineering, Streamlit UI (Single + Batch), Word report |

---

## Individual Contributions

### Joe Doan — Project Lead & AI Pipeline Architect
- Designed the overall **Dual-Threshold Pipeline Architecture**, defining the separation of responsibilities between the ML Tripwire, ChromaDB RAG layer, and Gemini LLM Evaluator
- Implemented `src/classifier.py`: trained the TF-IDF + Logistic Regression binary classification model and tuned the decision threshold (0.4) to maximize Recall (97.29%)
- Built `src/evaluate_pipeline.py`: designed and executed the **multi-approach performance comparison**, running ML-Only, RAG-Only, and Full Pipeline (Dual-Threshold) independently on a 60-sample balanced test set with Gemini API evaluation
- Led project planning, established team workflows, coordinated submission deliverables and GitHub repository structure
- Authored the technical architecture section and Evaluation Section 6.2 of the final report

### Manan Koradiya — Data Engineering Lead
- Identified and sourced the three-dataset ensemble strategy: `reddit-mental-health-classification` (HuggingFace), `mental-health-corpus` (Kaggle), and the pre-approved Kaggle KUC dataset
- Implemented the **Stratified Sub-sampling** logic in `src/build_vector_db.py`, engineering the targeted 43,551-row subset from 1.1M source rows
- Performed Exploratory Data Analysis (`src/eda.py`), mapping class distributions and identifying key substance abuse clusters within the datasets
- Built `src/topic_clustering.py`: implemented **K-Means clustering (k=8)** on ChromaDB embedding vectors to uncover 8 distinct behavioral risk profiles (PTSD, Alcohol, Suicide, Anxiety, etc.) as required by Core Task 2
- Authored the Dataset, Data Strategy, and Behavioral Clustering sections of the final report

### Ruixuan Hou — RAG & Vector Database Engineer
- Set up and configured the persistent **ChromaDB vector database** and integrated the `all-MiniLM-L6-v2` SentenceTransformer embedding pipeline
- Implemented the **strict L2 distance threshold (< 1.2)** for RAG retrieval to enforce high-confidence evidence retrieval and prevent LLM hallucination
- Designed the "Insufficient historical data" fallback logic that blocks low-confidence RAG context from being passed to the LLM
- Generated all **data visualizations** for the report: dataset composition chart, ML evaluation chart, approach comparison bar chart, and PCA 2D cluster scatter plot (`data/fig_*.png`, `data/cluster_visualization.png`)
- Validated RAG retrieval quality through manual inspection of retrieved evidence posts against test inputs
- Authored the Embedding, RAG, and Evaluation sections of the final report

### Aditya Naredla — LLM Prompt Engineer & Frontend Developer
- Engineered the **Gemini 2.5 system prompt**, designing the temporal intent differentiation logic that distinguishes Active Risk from Retrospective/Recovery narratives
- Implemented the **Tab 1 (Single Post Analysis)** UI in `src/app.py` including the Statistical Tripwire section and Final AI Evaluation section with two-column layout
- Built **Tab 2 (Batch CSV Analysis)** in `src/app.py`: file uploader, ML bulk scan, pie chart distribution, histogram, Top-5 flagged posts table, and CSV download functionality
- Designed `generate_report.py` to auto-generate the full submission **Word report** (`.docx`) with 6 embedded charts, 6 data tables, and all evaluation results
- Generated evaluation metrics and authored the Evaluation Results and Innovation Highlights sections of the final report

---

## Equal Contribution Statement

All four team members contributed equally to the project. Each member owned a distinct vertical of the pipeline (Data, ML/Evaluation, Vector DB/Visualization, LLM+UI/Report), ensuring that every component received focused, expert attention. All members participated in design reviews, adversarial testing validation, and final report writing.
