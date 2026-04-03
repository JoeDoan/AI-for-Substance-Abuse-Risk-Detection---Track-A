import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import joblib
import os
from dotenv import load_dotenv

# Load env variables (API Key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using Gemini 1.5/2.5 Pro or Flash
    # Update to gemini-2.5 models if available, otherwise gemini-1.5-pro is generally safe
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

@st.cache_resource
def load_resources():
    # Load fast ML Classifier
    clf = None
    if os.path.exists("data/risk_classifier.pkl"):
        clf = joblib.load("data/risk_classifier.pkl")
        
    # Load ChromaDB for RAG
    client = chromadb.PersistentClient(path="./data/chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection(name="reddit_mental_health", embedding_function=ef)
    except:
        collection = None
        
    return clf, collection

st.set_page_config(page_title="AI Risk Signal Analyst", layout="wide")
st.title("Substance Abuse & Distress Risk Analyzer")
st.markdown("### Public Health Signal Detection Dashboard")

clf, collection = load_resources()

# Interface
user_input = st.text_area("Enter social media text, forum post, or anonymous narrative for analysis:", height=200)

if st.button("Analyze Risk"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing signals with LLM & Retrieving historical evidence..."):
            
            # 1. RAG Retrieval (Find similar historical posts)
            similar_cases_context = ""
            if collection:
                results = collection.query(
                    query_texts=[user_input],
                    n_results=3
                )
                
                valid_cases = []
                if 'distances' in results and results['distances']:
                    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                        # Using L2 distance threshold < 1.2 as a strict cutoff for semantic similarity
                        if dist < 1.2:
                            valid_cases.append(f"- Case (Labeled as {meta['label']}): '{doc[:300]}...'")
                
                if valid_cases:
                    similar_cases_context = "Similar Historical Cases from Database:\n\n" + "\n".join(valid_cases)
                else:
                    similar_cases_context = "Insufficient historical data to assess risk (No high-confidence matches found)."
                    st.warning("Warning: Insufficient historical data to confidently assess risk.")
            
            # 2. Risk Classification (Statistical)
            ml_risk_tag = "Unknown"
            if clf:
                prob = clf.predict_proba([user_input])[0][1]
                # High recall threshold
                if prob > 0.4:
                    ml_risk_tag = "🟡 INITIAL SIGNAL DETECTED (High likelihood of substance/distress language)"
                elif prob > 0.2:
                    ml_risk_tag = "🟡 WEAK SIGNAL DETECTED"
                else:
                    ml_risk_tag = "⚪ NO INITIAL SIGNAL"
            
            # 3. LLM Reasoning (Risk Summary & Evidence Extraction)
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
            st.subheader("1. Statistical Tripwire (ML Keyword Baseline)")
            st.markdown(f"**{ml_risk_tag}**")
            
            st.subheader("2. Final AI Evaluation & Evidence (LLM + RAG)")
            st.write(analysis_text)
