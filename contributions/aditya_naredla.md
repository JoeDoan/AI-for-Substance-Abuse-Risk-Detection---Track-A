# Individual Contribution Report — Aditya Naredla

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** LLM Prompt Engineer & Frontend Developer

---

## Summary
I was responsible for the two most user-facing components of the project: the Gemini 2.5 reasoning layer and the Streamlit dashboard. My key challenge on the LLM side was making sure Gemini correctly differentiates between genuinely dangerous active risk and benign retrospective discussions. On the UI side, my goal was to make the Dual-Threshold strategy visually intuitive to judges in 3 seconds or less.

## Key Contributions

### 1. Gemini 2.5 Prompt Engineering
- Designed the full multi-task reasoning prompt that instructs Gemini to:
  1. Classify temporal intent: **Active Risk** (current distress, cravings, relapse) vs. **Retrospective/Recovery Narratives** (past stories, podcast promotion, stable recovery).
  2. Output a `🔴 FINAL AI ASSESSMENT: HIGH RISK` or `🟢 FINAL AI ASSESSMENT: LOW RISK` tag as the first line.
  3. Extract Supporting Evidence by directly quoting exact phrases from the input.
  4. Cross-reference the ChromaDB-retrieved cases for behavioral pattern matching.
- Validated the temporal differentiation logic against a real podcast promotion post — the LLM correctly identified it as retrospective and output `LOW RISK` despite keywords like "opioids" and "addiction."

### 2. Streamlit UI Development (`src/app.py`)
- Built the full interactive Streamlit dashboard including the text area input, "Analyze Risk" button, and results panel.
- Implemented the **Dual-Assessment UI framing** to eliminate contradictory signals:
  - Section 1: "🟡 INITIAL SIGNAL DETECTED" (ML Tripwire output)
  - Section 2: "🔴/🟢 FINAL AI ASSESSMENT" (LLM Final Evaluator output)
- Added warning banners for low RAG confidence cases.

### 3. Evaluation & Reporting
- Generated and documented final model metrics in `data/evaluation_metrics.txt`.
- Authored the Evaluation Results and Innovation Highlights sections of `REPORT.md`.

## Deliverables
- `src/app.py` (Streamlit UI)
- Gemini prompt engineering logic
- `data/evaluation_metrics.txt`
- Evaluation and Innovation sections of `REPORT.md`
