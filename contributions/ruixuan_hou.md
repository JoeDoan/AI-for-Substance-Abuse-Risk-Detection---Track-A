# Individual Contribution Report — Ruixuan Hou

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** RAG & Vector Database Engineer & Data Visualization Lead  
**Submission Date:** April 6, 2026

---

## Summary
I was responsible for designing and building the Retrieval-Augmented Generation (RAG) layer — the component that provides historical evidence grounding for the LLM — and for generating all data visualizations used in the final report. My key challenge on the RAG side was balancing retrieval coverage (finding relevant cases) with retrieval quality (only passing high-confidence context to Gemini). On the visualization side, my goal was to produce clear, professional charts that accurately represent our system's performance for judges.

## Key Contributions

### 1. Embedding Pipeline
- Integrated `sentence-transformers/all-MiniLM-L6-v2` through ChromaDB's embedding function API to generate dense 384-dimensional vector representations of all 43,551 posts.
- Implemented batched embedding (2,000 rows per batch) with `tqdm` progress tracking to safely handle the large embedding job without memory overflow.

### 2. ChromaDB Vector Database Setup
- Configured a `PersistentClient` ChromaDB instance stored at `data/chroma_db` so the vector DB survives application restarts without needing to re-embed.
- Structured metadata storage (`label` per document) to enable context-aware evidence citations in LLM responses and cluster analysis.

### 3. Strict Similarity Threshold for RAG Precision
- Implemented the **L2 Distance < 1.2 threshold** on retrieved results — the most important engineering decision in the RAG layer.
- Designed the **"Insufficient historical data" fallback** that blocks low-confidence context from reaching the LLM, preventing hallucination on out-of-distribution inputs.
- Validated the threshold through manual testing across multiple input types (active risk posts, recovery stories, promotional content).

### 4. Data Visualizations for Report (`data/fig_*.png`)
- Generated all four publication-quality charts embedded in the Word report:
  - **`fig_dataset_composition.png`**: horizontal bar chart of all 10 subreddit classes showing the 43,551-post composition (color-coded High Risk vs. Safe)
  - **`fig_ml_evaluation.png`**: double panel — precision/recall/F1 by class + overall 97% performance bar chart
  - **`fig_approach_comparison.png`**: grouped bar chart comparing ML-Only, RAG-Only, and Full Pipeline across Precision, Recall, and F1
  - **`cluster_visualization.png`**: PCA 2D scatter plot of K-Means cluster assignments across 5,000 High Risk posts
- All charts designed with dark theme aesthetic consistent with the project's public health dashboard style.

## Deliverables
- `data/chroma_db/` (persistent vector database, gitignored due to size)
- RAG retrieval logic in `src/app.py`
- `data/fig_dataset_composition.png`
- `data/fig_ml_evaluation.png`
- `data/fig_approach_comparison.png`
- `data/cluster_visualization.png`
- Embedding, RAG, and Visualization sections of the final report
