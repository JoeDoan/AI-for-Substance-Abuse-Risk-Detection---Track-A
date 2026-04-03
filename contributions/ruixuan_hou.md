# Individual Contribution Report — Ruixuan Hou

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** RAG & Vector Database Engineer

---

## Summary
I was responsible for designing and building the Retrieval-Augmented Generation (RAG) layer — the component that provides historical evidence grounding for the LLM. My key challenge was balancing retrieval coverage (finding relevant cases) with retrieval quality (only passing high-confidence context to Gemini). A poor RAG implementation would cause the LLM to either hallucinate or make overconfident inferences from noisy context.

## Key Contributions

### 1. Embedding Pipeline
- Integrated `sentence-transformers/all-MiniLM-L6-v2` through ChromaDB's embedding function API to generate dense vector representations of all 43,551 posts.
- Implemented batched embedding (2,000 rows per batch) with a `tqdm` progress bar to safely handle the large embed job without memory overflow.

### 2. ChromaDB Vector Database Setup
- Configured a `PersistentClient` ChromaDB instance stored at `data/chroma_db` so the vector DB survives application restarts without needing to re-embed.
- Structured metadata storage (`label` per document) to enable context-aware evidence citations in the LLM response.

### 3. Strict Similarity Threshold for RAG Precision
- Implemented the **L2 Distance < 1.2 threshold** on retrieved results — the most important engineering decision in the RAG layer.
- Designed the **"Insufficient historical data" fallback** that blocks low-confidence context from reaching the LLM, preventing hallucination on out-of-distribution inputs.
- Validated the threshold through manual testing across multiple input types (active risk posts, recovery stories, promotional content).

## Deliverables
- `data/chroma_db/` (persistent vector database)
- RAG retrieval logic in `src/app.py`
- Embedding and RAG sections of `REPORT.md`
