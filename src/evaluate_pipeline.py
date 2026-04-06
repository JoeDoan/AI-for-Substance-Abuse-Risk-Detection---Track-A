"""
evaluate_pipeline.py — Multi-Approach Performance Comparison
Core Task 1 / Track A: NSF NRT Challenge 1

Compares 3 detection approaches on a balanced mini test set:
  A. ML Only     (TF-IDF + Logistic Regression, threshold 0.4)
  B. RAG Only    (ChromaDB semantic similarity, distance < 1.2)
  C. Full Pipeline (ML → RAG → LLM — Dual-Threshold System)

Usage:
    python src/evaluate_pipeline.py

Outputs:
    data/approach_comparison.txt  — printable comparison table
    Prints results to console
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES_PER_CLASS = 30   # 30 High Risk + 30 Safe = 60 total
ML_THRESHOLD  = 0.4
RAG_THRESHOLD = 1.2        # L2 distance cutoff
RANDOM_STATE  = 99
LLM_DELAY_SEC = 1.5        # polite delay between Gemini API calls
OUTPUT_FILE   = "data/approach_comparison.txt"

HIGH_RISK_LABELS = {"addiction", "alcoholism", "suicidewatch", "depression", "ptsd", "mentalhealth", "anxiety"}
SAFE_LABELS      = {"jokes", "parenting", "fitness", "personalfinance"}


# ── Helper: LLM judge ─────────────────────────────────────────────────────────
LLM_PROMPT = """You are an expert AI Public Health Analyst.
Analyze this social media post for substance abuse or emotional distress risk.

Post: "{text}"

RAG Context: {rag_context}

Reply with ONLY one of these exact strings (no extra text):
HIGH_RISK
LOW_RISK
"""

def llm_classify(text: str, rag_context: str, llm_model) -> int:
    """Returns 1 = High Risk, 0 = Safe."""
    prompt = LLM_PROMPT.format(text=text[:800], rag_context=rag_context[:400])
    try:
        resp = llm_model.generate_content(prompt)
        answer = resp.text.strip().upper()
        return 1 if "HIGH_RISK" in answer else 0
    except Exception as e:
        print(f"     [LLM Error] {e}")
        return -1  # unknown


def main():
    print("=" * 65)
    print("  SubstanceWatch AI — Approach Comparison Evaluation")
    print("  (Core Task 1: Compare rule-based / RAG / Full Pipeline)")
    print("=" * 65)

    # ── 1. Build test set ────────────────────────────────────────────────────
    print("\n📥 Loading stratified_subset.csv...")
    df = pd.read_csv("data/stratified_subset.csv")

    rng = np.random.default_rng(RANDOM_STATE)

    high_risk_df = df[df["label"].isin(HIGH_RISK_LABELS)]
    safe_df      = df[df["label"].isin(SAFE_LABELS)]

    hr_sample  = high_risk_df.sample(n=N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
    safe_sample = safe_df.sample(n=N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE)

    test_df = pd.concat([hr_sample, safe_sample]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df["ground_truth"] = test_df["label"].apply(lambda x: 1 if x in HIGH_RISK_LABELS else 0)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    print(f"   Test set: {len(test_df)} posts  ({N_SAMPLES_PER_CLASS} High Risk + {N_SAMPLES_PER_CLASS} Safe)")

    y_true = test_df["ground_truth"].tolist()
    texts  = test_df["text"].tolist()

    # ── 2. Load models ───────────────────────────────────────────────────────
    print("\n🔧 Loading ML classifier & ChromaDB...")
    clf = joblib.load("data/risk_classifier.pkl")

    client = chromadb.PersistentClient(path="./data/chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="reddit_mental_health", embedding_function=ef)

    llm_model = genai.GenerativeModel("gemini-2.5-flash")

    # ── APPROACH A: ML Only ──────────────────────────────────────────────────
    print("\n🅐 Running Approach A — ML Only (TF-IDF + Logistic Regression)...")
    probs_a = clf.predict_proba(texts)[:, 1]
    y_pred_a = (probs_a >= ML_THRESHOLD).astype(int)

    # ── APPROACH B: RAG Only ─────────────────────────────────────────────────
    print("🅑 Running Approach B — RAG Only (ChromaDB semantic similarity)...")
    y_pred_b = []
    for i, text in enumerate(texts):
        results = collection.query(query_texts=[text], n_results=3)
        flag = 0  # default Safe
        if results.get("distances") and results["distances"]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                if dist < RAG_THRESHOLD and meta.get("label", "") in HIGH_RISK_LABELS:
                    flag = 1
                    break
        y_pred_b.append(flag)
        if (i + 1) % 10 == 0:
            print(f"   RAG: {i+1}/{len(texts)}")

    # ── APPROACH C: Full Dual-Threshold Pipeline ──────────────────────────────
    print("🅒 Running Approach C — Full Pipeline (ML → RAG → LLM)...")
    print("   [This will make ~60 Gemini API calls ... ~2-3 minutes]")
    y_pred_c = []
    llm_errors = 0

    for i, text in enumerate(texts):
        # Step 1: ML gate
        prob = clf.predict_proba([text])[0][1]
        if prob <= ML_THRESHOLD:
            # ML says Safe → skip LLM, return Safe
            y_pred_c.append(0)
            print(f"   [{i+1:02d}/{len(texts)}] ML cleared → Safe (prob={prob:.3f})")
            continue

        # Step 2: RAG retrieval
        rag_results = collection.query(query_texts=[text], n_results=3)
        valid_cases = []
        if rag_results.get("distances") and rag_results["distances"]:
            for doc, meta, dist in zip(rag_results["documents"][0], rag_results["metadatas"][0], rag_results["distances"][0]):
                if dist < RAG_THRESHOLD:
                    valid_cases.append(f"- [{meta['label']}] (d={dist:.2f}): {doc[:150]}")
        rag_context = "\n".join(valid_cases) if valid_cases else "No high-confidence similar cases found."

        # Step 3: LLM decision
        time.sleep(LLM_DELAY_SEC)
        decision = llm_classify(text, rag_context, llm_model)
        if decision == -1:
            llm_errors += 1
            decision = 1  # conservative fallback: flag as High Risk on error

        y_pred_c.append(decision)
        tag = "🔴 HIGH RISK" if decision == 1 else "🟢 Safe"
        print(f"   [{i+1:02d}/{len(texts)}] {tag}  (ML prob={prob:.3f})")

    # ── 3. Compute Metrics ───────────────────────────────────────────────────
    def metrics(y_true, y_pred, name):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        return {"Approach": name, "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc}

    results = [
        metrics(y_true, y_pred_a.tolist(), "A — ML Only (TF-IDF + LR)"),
        metrics(y_true, y_pred_b,          "B — RAG Only (ChromaDB)"),
        metrics(y_true, y_pred_c,          "C — Dual-Threshold (Full Pipeline)"),
    ]
    df_results = pd.DataFrame(results)

    # ── 4. Print Report ──────────────────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print("  APPROACH COMPARISON RESULTS")
    print(f"  Test Set: {len(test_df)} posts ({N_SAMPLES_PER_CLASS} High Risk + {N_SAMPLES_PER_CLASS} Safe)")
    if llm_errors > 0:
        print(f"  ⚠️  LLM API errors: {llm_errors} (fell back to High Risk)")
    print("=" * 65)

    header = f"{'Approach':<38} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Acc':>6}"
    sep    = "-" * 65
    print(header)
    print(sep)
    for _, row in df_results.iterrows():
        line = (f"{row['Approach']:<38} "
                f"{row['Precision']:>6.4f} "
                f"{row['Recall']:>7.4f} "
                f"{row['F1']:>6.4f} "
                f"{row['Accuracy']:>6.4f}")
        print(line)
    print(sep)

    # Interpretation
    best_f1 = df_results.loc[df_results["F1"].idxmax(), "Approach"]
    best_rec = df_results.loc[df_results["Recall"].idxmax(), "Approach"]
    print(f"\n  Best F1:     {best_f1}")
    print(f"  Best Recall: {best_rec}")
    print("\n  Key insight: The Full Pipeline (C) uses LLM reasoning to reduce")
    print("  false positives while the ML layer (A) maximizes Recall as a safety net.")

    # ── 5. Save Output ───────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    report_lines = [
        "SubstanceWatch AI — Approach Comparison Report",
        f"Test Set: {len(test_df)} posts ({N_SAMPLES_PER_CLASS} High Risk + {N_SAMPLES_PER_CLASS} Safe)\n",
        header,
        sep,
    ]
    for _, row in df_results.iterrows():
        report_lines.append(
            f"{row['Approach']:<38} "
            f"{row['Precision']:>6.4f} "
            f"{row['Recall']:>7.4f} "
            f"{row['F1']:>6.4f} "
            f"{row['Accuracy']:>6.4f}"
        )
    report_lines.append(sep)
    report_lines.append(f"\nBest F1:     {best_f1}")
    report_lines.append(f"Best Recall: {best_rec}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✅ Report saved to {OUTPUT_FILE}")

    # Return results dict for README update hint
    return df_results


if __name__ == "__main__":
    main()
