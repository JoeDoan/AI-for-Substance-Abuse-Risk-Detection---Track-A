# Individual Contribution Report — Joe Doan

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** Project Lead & AI Pipeline Architect

---

## Summary
I served as the team lead and was responsible for the overall system design and the core ML classification layer. My primary goal was to ensure that each component of our pipeline had a clearly defined purpose, and that our Dual-Threshold architecture was coherent and well-reasoned from a public health standpoint.

## Key Contributions

### 1. System Architecture Design
- Designed the **Dual-Threshold Pipeline**: separating the Recall-optimized ML Tripwire from the Precision-optimized LLM Evaluator to correctly address both False Negative risk (in public health) and False Positive noise.
- Made the critical architectural decision to use ChromaDB as a strict evidence gate — if no high-confidence match is found, the LLM is blocked from speculating.

### 2. ML Classification Pipeline (`src/classifier.py`)
- Trained a `TF-IDF + Logistic Regression` pipeline on the 43,551-row stratified subset.
- Defined the binary classification target: mapping `addiction`, `alcoholism`, `depression`, `suicidewatch`, `ptsd`, `anxiety`, `mentalhealth` → High Risk; safe topics → Safe.
- Tuned the decision threshold to `0.4` to maximize Recall, resulting in:
  - **Recall:** 97.29%
  - **Precision:** 99.05%
  - **F1 Score:** 98.16%
- Saved the trained model to `data/risk_classifier.pkl` for reuse in the Streamlit app.

### 3. Project Coordination
- Coordinated team roles, dataset approval email to Dr. Lee, and submission checklist.
- Led design review sessions and authored the system architecture section of `REPORT.md`.

## Deliverables
- `src/classifier.py`
- `data/risk_classifier.pkl`
- `data/evaluation_metrics.txt`
- Architecture design and report sections
