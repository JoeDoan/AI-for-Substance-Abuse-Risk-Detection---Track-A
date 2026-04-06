# Individual Contribution Report — Joe Doan

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** Project Lead & AI Pipeline Architect  
**Submission Date:** April 6, 2026

---

## Summary
I served as the team lead and was responsible for the overall system design, the core ML classification layer, and the multi-approach performance evaluation framework. My primary goal was to ensure that each component of our pipeline had a clearly defined purpose, that the Dual-Threshold architecture was coherent from a public health standpoint, and that our design was validated against quantifiable benchmarks.

## Key Contributions

### 1. System Architecture Design
- Designed the **Dual-Threshold Pipeline**: separating the Recall-optimized ML Tripwire from the Precision-optimized LLM Evaluator to correctly address both False Negative risk (in public health) and False Positive noise.
- Made the critical architectural decision to use ChromaDB as a strict evidence gate — if no high-confidence match is found (L2 distance ≥ 1.2), the LLM is blocked from speculating.
- Defined the precise handoff logic between all three pipeline layers (ML → RAG → LLM) and ensured the design was reproducible for the Research-A-Thon live demo.

### 2. ML Classification Pipeline (`src/classifier.py`)
- Trained a `TF-IDF + Logistic Regression` pipeline on the 43,551-row stratified subset.
- Defined the binary classification target: mapping `addiction`, `alcoholism`, `depression`, `suicidewatch`, `ptsd`, `anxiety`, `mentalhealth` → High Risk (1); safe control classes → Safe (0).
- Tuned the decision threshold to `0.4` to maximize Recall, resulting in:
  - **Recall:** 97.29% (fewer than 3% of genuine distress signals missed)
  - **Precision:** 99.05%
  - **F1 Score:** 98.16%
- Saved trained model to `data/risk_classifier.pkl` for reuse in both the Streamlit app and the evaluation pipeline.

### 3. Multi-Approach Performance Evaluation (`src/evaluate_pipeline.py`)
- Designed and implemented the comparative evaluation framework validating our Dual-Threshold design quantitatively.
- Created a balanced 60-sample test set (30 High Risk + 30 Safe) from `stratified_subset.csv`.
- Ran three independent approaches on the same test set:
  - **Approach A — ML Only:** TF-IDF + LR with threshold 0.4
  - **Approach B — RAG Only:** ChromaDB retrieval with L2 < 1.2
  - **Approach C — Full Dual-Threshold Pipeline:** ML → RAG → Gemini 2.5 LLM
- Results demonstrated that the Full Pipeline achieves **Precision = 1.000** (zero false positives) while the ML layer maintains maximum Recall as the safety net.
- Exported full comparison report to `data/approach_comparison.txt`.

### 4. Project Coordination
- Coordinated team roles, dataset approval with Dr. Yugyung Lee, and final submission checklist.
- Led design review sessions and authored the System Architecture and Evaluation Section 6.2 of the final report and README.

## Deliverables
- `src/classifier.py`
- `src/evaluate_pipeline.py`
- `data/risk_classifier.pkl`
- `data/evaluation_metrics.txt`
- `data/approach_comparison.txt`
- Architecture design, README Section 6.2, and report sections
