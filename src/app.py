import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from dotenv import load_dotenv

# Load env variables (API Key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

@st.cache_resource
def load_resources():
    clf = None
    if os.path.exists("data/risk_classifier.pkl"):
        clf = joblib.load("data/risk_classifier.pkl")

    client = chromadb.PersistentClient(path="./data/chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection(name="reddit_mental_health", embedding_function=ef)
    except:
        collection = None

    return clf, collection

st.set_page_config(page_title="SubstanceWatch AI — Risk Signal Analyst", layout="wide")
st.title("🔍 SubstanceWatch AI")
st.markdown("### Substance Abuse & Distress Risk Detection | NSF NRT Challenge 1 · Track A")
st.divider()

clf, collection = load_resources()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 Single Post Analysis", "📊 Batch CSV Analysis"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Post Analysis (original functionality)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Analyze a single social media post or narrative")
    user_input = st.text_area(
        "Enter social media text, forum post, or anonymous narrative for analysis:",
        height=200,
        placeholder='e.g. "Everything is falling apart. Lost my job yesterday and I\'m thinking about hitting the slopes tonight just to numb it all..."'
    )

    if st.button("🧠 Analyze Risk", key="btn_single"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing signals with LLM & Retrieving historical evidence..."):

                # 1. RAG Retrieval
                similar_cases_context = ""
                if collection:
                    results = collection.query(query_texts=[user_input], n_results=3)
                    valid_cases = []
                    if 'distances' in results and results['distances']:
                        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                            if dist < 1.2:
                                valid_cases.append(f"- Case (Labeled as {meta['label']}): '{doc[:300]}...'")

                    if valid_cases:
                        similar_cases_context = "Similar Historical Cases from Database:\n\n" + "\n".join(valid_cases)
                    else:
                        similar_cases_context = "Insufficient historical data to assess risk (No high-confidence matches found)."
                        st.warning("Warning: Insufficient historical data to confidently assess risk.")

                # 2. ML Risk Classification
                ml_risk_tag = "Unknown"
                if clf:
                    prob = clf.predict_proba([user_input])[0][1]
                    if prob > 0.4:
                        ml_risk_tag = "🟡 INITIAL SIGNAL DETECTED (High likelihood of substance/distress language)"
                    elif prob > 0.2:
                        ml_risk_tag = "🟡 WEAK SIGNAL DETECTED"
                    else:
                        ml_risk_tag = "⚪ NO INITIAL SIGNAL"

                # 3. LLM Reasoning
                prompt = f"""
                You are an expert AI Psychological and Public Health Analyst.
                Analyze the following social media input text for ANY signals of emotional distress, relapse, or substance abuse.

                User Input:
                "{user_input}"

                {similar_cases_context}

                Task:
                1. Evaluate the temporal context and user intent carefully. Differentiate between 'Active Risk' (current distress, cravings, immediate relapse signals) and 'Retrospective/Recovery Narratives' (discussing past addiction, sharing recovery stories, or promoting content like blogs/podcasts).
                2. Provide a 'Final AI Assessment Tag' at the very top of your summary (e.g., '🔴 FINAL AI ASSESSMENT: HIGH RISK' or '🟢 FINAL AI ASSESSMENT: LOW RISK'). Below it, write a concise 'Risk Summary' explaining your decision. **CRITICAL:** If the post is primarily reflective, promotional, or indicates stable recovery, gracefully assign a '🟢 FINAL AI ASSESSMENT: LOW RISK' tag and explain that this is a retrospective/recovery narrative, not an active crisis.
                3. Provide 'Supporting Evidence', explicitly quoting specific phrases from the User Input that led to your conclusion.
                4. If available, briefly note if the User shares behavioral patterns with the Similar Historical Cases provided.

                Format your response clearly. Keep it professional and empathetic.
                """

                try:
                    response = model.generate_content(prompt)
                    analysis_text = response.text
                except Exception as e:
                    analysis_text = f"Error calling Gemini LLM: {str(e)}"

                # Display Results
                st.divider()
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("1. ML Tripwire")
                    st.markdown(f"**{ml_risk_tag}**")
                    if clf:
                        st.caption(f"Raw probability: {prob:.4f}")
                with col2:
                    st.subheader("2. Final AI Evaluation (LLM + RAG)")
                    st.write(analysis_text)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Batch-analyze a dataset of social media posts using the ML Tripwire")
    st.info(
        "📌 **How it works:** Upload a CSV with a `text` column (max 100 rows). "
        "The ML Tripwire will scan every post instantly. "
        "For deep LLM analysis of flagged posts, copy them to the **Single Post Analysis** tab.",
        icon="ℹ️"
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file (must contain a `text` column)",
        type=["csv"],
        key="batch_uploader"
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_upload = None

        if df_upload is not None:
            if "text" not in df_upload.columns:
                st.error("❌ CSV must contain a column named `text`. Please check your file.")
            else:
                # Cap at 100 rows
                if len(df_upload) > 100:
                    st.warning(f"⚠️ File has {len(df_upload)} rows. Only the first 100 will be analyzed.")
                    df_upload = df_upload.head(100)

                df_upload["text"] = df_upload["text"].fillna("").astype(str)
                df_upload = df_upload[df_upload["text"].str.strip().str.len() > 5].reset_index(drop=True)

                st.success(f"✅ Loaded **{len(df_upload)} posts**. Running ML Tripwire analysis...")

                if clf is None:
                    st.error("ML model not loaded. Please run `python src/classifier.py` first.")
                else:
                    with st.spinner("Scanning all posts with ML Tripwire..."):
                        probs = clf.predict_proba(df_upload["text"].tolist())[:, 1]
                        df_upload["ml_probability"] = probs
                        df_upload["ml_label"] = df_upload["ml_probability"].apply(
                            lambda p: "🔴 High Risk" if p > 0.4 else ("🟡 Weak Signal" if p > 0.2 else "🟢 Safe")
                        )
                        df_upload["risk_category"] = df_upload["ml_probability"].apply(
                            lambda p: "High Risk" if p > 0.4 else "Safe"
                        )

                    # ── METRICS ──
                    n_high = (df_upload["risk_category"] == "High Risk").sum()
                    n_safe = (df_upload["risk_category"] == "Safe").sum()
                    total = len(df_upload)

                    st.divider()
                    st.subheader("📈 Analysis Summary")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Posts Analyzed", total)
                    m2.metric("🔴 High Risk Flagged", n_high, delta=f"{n_high/total*100:.1f}%")
                    m3.metric("🟢 Safe", n_safe, delta=f"{n_safe/total*100:.1f}%")

                    st.divider()

                    # ── CHARTS ──
                    col_pie, col_bar = st.columns(2)

                    with col_pie:
                        st.markdown("**Risk Distribution**")
                        fig_pie, ax_pie = plt.subplots(figsize=(4, 4), facecolor='#0e1117')
                        ax_pie.set_facecolor('#0e1117')
                        sizes = [n_high, n_safe]
                        labels = ['High Risk', 'Safe']
                        colors = ['#e05252', '#52b788']
                        explode = (0.05, 0)
                        wedges, texts, autotexts = ax_pie.pie(
                            sizes, labels=labels, colors=colors,
                            autopct='%1.1f%%', startangle=140,
                            explode=explode, textprops={'color': 'white', 'fontsize': 11}
                        )
                        for at in autotexts:
                            at.set_color('white')
                            at.set_fontsize(10)
                        ax_pie.set_title("High Risk vs Safe", color='white', fontsize=12, pad=10)
                        st.pyplot(fig_pie, use_container_width=True)
                        plt.close(fig_pie)

                    with col_bar:
                        st.markdown("**Probability Score Distribution**")
                        fig_hist, ax_hist = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
                        ax_hist.set_facecolor('#1a1a2e')
                        ax_hist.hist(
                            df_upload[df_upload["risk_category"] == "High Risk"]["ml_probability"],
                            bins=15, color='#e05252', alpha=0.8, label='High Risk', edgecolor='none'
                        )
                        ax_hist.hist(
                            df_upload[df_upload["risk_category"] == "Safe"]["ml_probability"],
                            bins=15, color='#52b788', alpha=0.8, label='Safe', edgecolor='none'
                        )
                        ax_hist.axvline(x=0.4, color='#ffd60a', linestyle='--', linewidth=1.5, label='Threshold (0.4)')
                        ax_hist.set_xlabel("ML Probability Score", color='white')
                        ax_hist.set_ylabel("Count", color='white')
                        ax_hist.set_title("Score Distribution", color='white')
                        ax_hist.tick_params(colors='white')
                        ax_hist.spines[:].set_color('#333')
                        legend = ax_hist.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
                        st.pyplot(fig_hist, use_container_width=True)
                        plt.close(fig_hist)

                    st.divider()

                    # ── TOP 5 HIGH RISK POSTS ──
                    st.subheader("🚨 Top 5 Highest-Risk Posts")
                    st.caption("Posts most likely to contain substance abuse or distress signals (copy text to Single Analysis tab for LLM deep-dive)")
                    top5 = df_upload.nlargest(5, "ml_probability")[["text", "ml_probability", "ml_label"]].copy()
                    top5.index = range(1, 6)
                    top5.columns = ["Text (truncated)", "ML Probability", "Risk Label"]
                    top5["Text (truncated)"] = top5["Text (truncated)"].apply(lambda x: x[:300] + "..." if len(x) > 300 else x)
                    st.dataframe(top5, use_container_width=True)

                    st.divider()

                    # ── FULL RESULTS TABLE ──
                    st.subheader("📋 Full Results")
                    display_df = df_upload[["text", "ml_probability", "ml_label"]].copy()
                    display_df.columns = ["Text", "ML Probability", "Risk Label"]
                    display_df["Text"] = display_df["Text"].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)
                    display_df["ML Probability"] = display_df["ML Probability"].round(4)
                    st.dataframe(display_df, use_container_width=True, height=350)

                    # ── DOWNLOAD ──
                    download_df = df_upload[["text", "ml_probability", "ml_label", "risk_category"]].copy()
                    download_df.columns = ["text", "ml_probability", "ml_label", "risk_category"]
                    csv_bytes = download_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Full Results as CSV",
                        data=csv_bytes,
                        file_name="substancewatch_batch_results.csv",
                        mime="text/csv"
                    )
