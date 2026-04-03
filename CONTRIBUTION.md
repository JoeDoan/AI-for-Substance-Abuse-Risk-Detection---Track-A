# Team Contribution Report

**Project:** SubstanceWatch AI — NSF NRT Challenge 1: AI for Substance Abuse Risk Detection  
**Event:** UMKC Research-A-Thon 2026  
**Team Members:** Joe Doan · Manan Koradiya · Ruixuan Hou · Aditya Naredla

---

## Contribution Summary

| Team Member | Primary Role | Key Deliverables |
|---|---|---|
| Joe Doan | Project Lead & AI Pipeline Architect | System design, ML classifier, backend integration |
| Manan Koradiya | Data Engineering Lead | Dataset curation, stratified sampling, EDA |
| Ruixuan Hou | RAG & Vector DB Engineer | ChromaDB setup, embedding pipeline, retrieval logic |
| Aditya Naredla | LLM Prompt Engineer & Frontend | Gemini prompt engineering, Streamlit UI, evaluation |

---

## Individual Contributions

### Joe Doan — Project Lead & AI Pipeline Architect
- Designed the overall **Dual-Threshold Pipeline Architecture**, defining the separation of responsibilities between the ML Tripwire, ChromaDB RAG layer, and Gemini LLM Evaluator
- Implemented `src/classifier.py`: trained the TF-IDF + Logistic Regression binary classification model and tuned the decision threshold (0.4) to maximize Recall (97.29%)
- Led project planning, established team workflows, coordinated submission deliverables and GitHub repository structure
- Authored the technical architecture section of the final report

### Manan Koradiya — Data Engineering Lead
- Identified and sourced the three-dataset ensemble strategy: `reddit-mental-health-classification` (HuggingFace), `mental-health-corpus` (Kaggle), and the pre-approved Kaggle KUC dataset
- Implemented the **Stratified Sub-sampling** logic in `src/build_vector_db.py`, engineering the targeted 43,551-row subset from 1.1M source rows
- Performed Exploratory Data Analysis (`src/eda.py`), mapping class distributions and identifying key substance abuse clusters within the datasets
- Authored the Dataset and Data Strategy section of the final report

### Ruixuan Hou — RAG & Vector DB Engineer
- Set up and configured the persistent **ChromaDB vector database** and integrated the `all-MiniLM-L6-v2` SentenceTransformer embedding pipeline
- Implemented the **strict L2 distance threshold (< 1.2)** for RAG retrieval to enforce high-confidence evidence retrieval and prevent LLM hallucination
- Designed the "Insufficient historical data" fallback logic that blocks low-confidence RAG context from being passed to the LLM
- Validated RAG retrieval quality through manual inspection of retrieved evidence posts against test inputs
- Authored the Embedding and RAG sections of the final report

### Aditya Naredla — LLM Prompt Engineer & Frontend Developer
- Engineered the **Gemini 2.5 system prompt**, designing the temporal intent differentiation logic that distinguishes Active Risk from Retrospective/Recovery narratives
- Implemented the Dual-Assessment UI in `src/app.py` using Streamlit, including the Statistical Tripwire section and the Final AI Evaluation section
- Designed the UX framing strategy (ML as "Tripwire", LLM as "Final Evaluator") to ensure the dashboard is intuitive and non-contradictory for public health analysts
- Generated evaluation metrics and authored the Evaluation Results section of the final report

---

## Equal Contribution Statement

All four team members contributed equally to the project. Each member owned a distinct vertical of the pipeline (Data, ML, Vector DB, LLM+UI), ensuring that every component received focused, expert attention. All members participated in design reviews and final report writing.
