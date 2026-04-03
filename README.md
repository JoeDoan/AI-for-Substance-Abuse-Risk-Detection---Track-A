# NSF NRT Research-A-Thon 2026
# Challenge 1: AI for Substance Abuse Risk Detection from Social Signals
# Track A: AI Modeling and Reasoning

**Team Members:** Joe Doan · Manan Koradiya · Ruixuan Hou · Aditya Naredla  
**Submission Date:** April 6, 2026  
**Event Date:** April 10, 2026 | UMKC Student Union Room 401

---

## 1. Introduction & Problem Statement

Substance abuse represents a growing societal crisis in the United States. According to the CDC, drug overdoses kill over 100,000 Americans per year, with early warning signals frequently surfacing in digital environments such as online forums and social communities before they escalate to physical crises. Traditional public health monitoring relies on retrospective data, clinical reporting, and expensive surveys — all of which are inherently delayed.

This project addresses that gap by building an AI-driven pipeline, **SubstanceWatch AI**, that identifies substance abuse risk signals and emotional distress from public, anonymized social media data. Our system transforms unstructured text into interpretable, actionable insights that can support early awareness and population-level intervention.

---

## 2. Dataset & Data Strategy

We curated an ensemble of three public datasets to provide both depth and diversity of signals:

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Reddit Mental Health Classification | HuggingFace (`kamruzzaman-asif`) | 1.1M posts | Core social signals: addiction, alcoholism, depression, PTSD |
| Mental Health Corpus | Kaggle (`reihanenamdari`) | ~28K posts | Baseline emotional distress (binary labeled) |
| *Pre-approved reference* | Kaggle KUC Hackathon 2018 | ~200K | Medical drug reviews for baseline comparison |

**Stratified Sub-sampling Strategy:** Rather than embedding all 1.1M Reddit posts (which would be computationally prohibitive), we engineered a targeted 43,551-row subset using the following logic:

- **All** posts from `addiction` (~7,750) and `alcoholism` (~5,530) subreddits
- **5,000** posts each from `depression`, `suicidewatch`, `anxiety`, `ptsd`, `mentalhealth`
- **5,000** posts from control/safe classes (`jokes`, `parenting`, `fitness`)

This subset ensures the vector space has dense, well-separated clusters for direct addiction signals, underlying emotional distress, and safe baseline content — enabling high-quality semantic retrieval.

---

## 3. System Architecture: Dual-Threshold Pipeline

Our core innovation is a **Dual-Threshold Strategy** that leverages two AI layers with distinct roles:

```
[Raw Text Input]
       │
       ▼
┌─────────────────────────────────────────────┐
│  LAYER 1: ML Statistical Tripwire           │
│  Model: TF-IDF + Logistic Regression        │
│  Role: Cast a WIDE net (maximize Recall)    │
│  Threshold: 0.4 (high sensitivity)          │
│  Output: "🟡 INITIAL SIGNAL DETECTED"       │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  LAYER 2: RAG Retrieval (ChromaDB)          │
│  Model: all-MiniLM-L6-v2 Embeddings         │
│  Role: Retrieve high-confidence evidence    │
│  Threshold: L2 Distance < 1.2 (strict)      │
│  Output: Top-3 similar historical cases     │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  LAYER 3: LLM Final Evaluator (Gemini 2.5)  │
│  Role: Temporal + Intent reasoning          │
│  Differentiates Active Risk vs. Retrospective│
│  Output: "🟢/🔴 FINAL AI ASSESSMENT"        │
│           + Risk Summary + Supporting Evidence│
└─────────────────────────────────────────────┘
```

This separation of concerns is a deliberate architectural decision. In public health, **missing a real distress signal (False Negative) is far more dangerous than a False Positive.** The ML layer is therefore tuned for maximum Recall (97.29%), while the LLM layer applies high-Precision contextual reasoning to eliminate noise and prevent false alarms.

---

## 4. Quick Start & Installation

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoeDoan/AI-for-Substance-Abuse-Risk-Detection---Track-A.git
   cd "AI-for-Substance-Abuse-Risk-Detection---Track-A"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_genai_api_key_here
   ```

4. **Build the Vector Database and ML Models:**
   *Because the vector database and datasets are too large for GitHub, you need to generate them locally before running the app. Since we only sample a 40k row subset, it will embed locally very quickly!*
   ```bash
   python src/eda.py
   python src/build_vector_db.py
   python src/classifier.py
   ```

5. **Run the Dashboard:**
   ```bash
   streamlit run src/app.py
   ```
   The Streamlit app will be available at `http://localhost:8501`.

---

## 5. Technical Implementation

### 5.1 Risk Signal Detection (ML Layer)

We trained a `TF-IDF + Logistic Regression` pipeline on the 43,551-row stratified subset. The binary classification target maps all addiction, alcoholism, depression, suicidewatch, PTSD, and anxiety posts to `High Risk (1)`, and safe classes to `Safe (0)`.

**Decision threshold lowered to 0.4** to maximize Recall — ensuring no genuine distress signals are missed at the tripwire stage.

### 5.2 Embedding-Based RAG (ChromaDB Layer)

All 43,551 rows were embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a local, persistent `ChromaDB` vector database. When a user submits a query, we retrieve the Top-3 most semantically similar historical posts.

**Strict L2 Distance threshold (< 1.2)** is enforced on retrieved results. If no post clears this bar, the system outputs: *"Insufficient historical data to assess risk"* — preventing the LLM from hallucinating unsupported evidence.

### 5.3 LLM Reasoning & Explainability (Gemini 2.5 Layer)

Using the RAG-retrieved context as grounding, Gemini 2.5 is prompted to:
1. **Classify temporal intent:** Differentiate `Active Risk` (current distress, cravings) from `Retrospective/Recovery Narratives` (discussing past addiction, sharing stories, promoting podcasts).
2. **Output a Final AI Assessment Tag** that overrides the statistical baseline when warranted.
3. **Extract Supporting Evidence** by directly quoting the specific phrases that triggered or cleared the risk assessment.

This makes every output fully transparent, auditable, and ethically aligned with public health standards.

---

## 6. Evaluation Results

The classification model was evaluated on an 80/20 train-test split:

| Metric | High Risk Class | Safe Class | Weighted Avg |
|---|---|---|---|
| **Precision** | 0.9905 | 0.82 | **0.97** |
| **Recall** | **0.9729** | 0.93 | 0.97 |
| **F1 Score** | 0.9816 | 0.87 | **0.97** |
| **Accuracy** | — | — | **97%** |

**Key Design Win:** By targeting Recall > 97% at the ML layer, we ensure fewer than 3% of genuine distress signals are missed. The LLM layer then filters out false positives through semantic reasoning, achieving a complete Dual-Threshold system that prioritizes both safety and reliability.

**RAG Quality:** Manual validation confirmed that retrieved evidence posts are contextually and topically relevant to the input query. The strict distance cutoff (< 1.2) successfully prevented RAG context from low-confidence matches in all tested cases.

---

## 7. Innovation Highlights

1. **Dual-Threshold Design:** Separates the "catching" problem (Recall-optimized ML) from the "filtering" problem (Precision-optimized LLM reasoning), rather than forcing one model to do both.

2. **Active vs. Retrospective Classification:** The Gemini 2.5 prompt explicitly differentiates between active crisis signals and recovery/promotional content — a nuance most binary classifiers cannot capture.

3. **Grounded LLM Outputs:** All LLM summaries are grounded in retrieved evidence from the ChromaDB. If no high-confidence match exists, the LLM is blocked from generating unsupported claims.

4. **Ethical AI by Design:** No personally identifiable information (PII) is used at any stage. The system operates at population-level and explicitly anonymizes all data.

---

## 8. Conclusion & Future Work

SubstanceWatch AI demonstrates that a well-architected combination of classical ML, vector embeddings, and LLM reasoning can create a reliable, explainable, and ethically sound substance abuse risk detection system.

**Future Directions:**
- Integrate the CDC/NIDA public health datasets for cross-source temporal trend analysis
- Add a BERTopic topic modeling layer to surface emerging risk themes community-wide
- Deploy a temporal spike dashboard to detect population-level behavioral shifts by week/month

---

## 9. References

1. CDC National Drug Overdose Data: https://data.cdc.gov
2. Reddit Mental Health Classification Dataset: https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification  
3. Mental Health Corpus: https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus  
4. Reimers & Gurevych (2019). Sentence-BERT. https://arxiv.org/abs/1908.10084
5. Lewis et al. (2020). Retrieval-Augmented Generation. https://arxiv.org/abs/2005.11401
6. Google Gemini 2.5: https://deepmind.google/technologies/gemini/

---

**GitHub Repository:** https://github.com/JoeDoan/AI-for-Substance-Abuse-Risk-Detection---Track-A.git  
**Contact:** Joe Doan — jdoan@umkc.edu
