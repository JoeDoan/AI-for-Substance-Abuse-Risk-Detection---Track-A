# Individual Contribution Report — Aditya Naredla

**Project:** SubstanceWatch AI — NSF NRT Challenge 1  
**Role:** LLM Prompt Engineer & Frontend Developer & Report Lead  
**Submission Date:** April 6, 2026

---

## Summary
I was responsible for the most user-facing components of the project: the Gemini 2.5 reasoning layer, the Streamlit dashboard (both Single Post and Batch Analysis tabs), and the automated Word report generation. My key challenge on the LLM side was making Gemini correctly differentiate between genuinely dangerous active risk and benign retrospective discussions. On the UI side, my goal was to make the Dual-Threshold strategy visually intuitive. For the report, I automated the entire generation pipeline so the team could regenerate a polished Word document at any time.

## Key Contributions

### 1. Gemini 2.5 Prompt Engineering
- Designed the full multi-task reasoning prompt that instructs Gemini to:
  1. Classify temporal intent: **Active Risk** (current distress, cravings, relapse) vs. **Retrospective/Recovery Narratives** (past stories, podcast promotion, stable recovery).
  2. Output a `🔴 FINAL AI ASSESSMENT: HIGH RISK` or `🟢 FINAL AI ASSESSMENT: LOW RISK` tag as the first line of the response.
  3. Extract **Supporting Evidence** by directly quoting exact phrases from the input that triggered the assessment.
  4. Cross-reference ChromaDB-retrieved historical cases for behavioral pattern matching.
- Validated prompt quality using 3 adversarial test scenarios (`test_adversarial.py`), all passing with expected outcomes.

### 2. Tab 1 — Single Post Analysis UI (`src/app.py`)
- Built the full interactive Streamlit dashboard with two-column layout for the analysis results.
- Implemented the **Dual-Assessment UI framing** to eliminate contradictory signals:
  - Column 1: "🟡 INITIAL SIGNAL DETECTED" (ML Tripwire output + raw probability)
  - Column 2: "🔴/🟢 FINAL AI ASSESSMENT" (LLM Final Evaluator output)
- Added warning banners for low RAG confidence cases.

### 3. Tab 2 — Batch CSV Analysis UI (`src/app.py`)
- Designed and implemented the full **Batch CSV Analysis** tab for population-level signal scanning:
  - File uploader accepting CSV with `text` column (max 100 rows enforced)
  - Bulk ML Tripwire classification across all rows simultaneously
  - Real-time metrics: total posts analyzed, High Risk count/percentage, Safe count
  - Pie chart (risk distribution) and histogram (probability score distribution) with dark theme
  - Top-5 highest-risk posts table for manual review prioritization
  - Full results dataframe with CSV download button

### 4. Word Report Generation (`generate_report.py`)
- Built the automated **Word report (.docx) generation pipeline** using `python-docx`.
- Designed a full 8-section report template: Introduction, Dataset, Architecture, Evaluation (ML + Multi-Approach), Behavioral Clustering, Adversarial Testing, Innovation, References.
- Embedded all 6 visual figures (data charts + dashboard screenshots) with descriptive captions.
- Included 6 formatted tables (dataset table, architecture table, evaluation tables, cluster table, adversarial table).
- Automated margin/font configuration to minimize page count while maintaining readability.

## Deliverables
- `src/app.py` (Streamlit UI — Tab 1 + Tab 2)
- `generate_report.py`
- `SubstanceWatch_AI_Report.docx`
- Gemini prompt engineering logic
- `data/evaluation_metrics.txt`
- Evaluation, Innovation, and Adversarial sections of the final report
