import joblib
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from dotenv import load_dotenv

user_input = """I’ve been thinking a lot lately about where it actually started for me. Not the moment things fell apart, but the moment the seed got planted years before I even knew it existed.

I was 16, obsessively academic, doing research at a national laboratory, running on pure pressure and ambition. Lost 60 pounds too fast, ended up with gallstones, had surgery. They gave me opioids for the pain afterward. Standard stuff.

But for the first time in my life I felt the weight come off my shoulders. Weight I didn’t even know I was carrying. I remember thinking — I had no idea a human being could actually feel this light. I didn’t even finish the prescription. But the seed was planted.

Years later when life got hard enough, I remembered that feeling. And that was it.

I’m curious whether others had a similar experience — not necessarily with opioids, but that one specific moment where something shifted and you didn’t realize until much later that it was the beginning of everything.

I’ve been processing a lot of this stuff out loud lately with a friend of mine who went through his own version of the same journey. If anyone’s interested in the longer conversation we had about origin stories and where addiction actually begins, we talked about it openly on our podcast Dead Reckoning — https://www.youtube.com/@TheDeadReckoningPodcast?sub_confirmation=1"""

# 1. ML Classifier Run
try:
    clf = joblib.load("data/risk_classifier.pkl")
    prob = clf.predict_proba([user_input])[0][1]

    ml_risk_tag = "Unknown"
    if prob > 0.4:
        ml_risk_tag = "🔴 HIGH RISK (Distress/Substance Abuse Detected)"
    elif prob > 0.2:
        ml_risk_tag = "🟠 MODERATE RISK"
    else:
        ml_risk_tag = "🟢 LOW RISK"

    print(f"--- ML Classification (Recall-Optimized) ---")
    print(f"Assigned Tag: {ml_risk_tag}")
    print(f"Raw Probability of High Risk: {prob:.4f}\n")
except Exception as e:
    print(f"Error in ML: {e}")

# 2. ChromaDB RAG
try:
    db_path = "./data/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="reddit_mental_health", embedding_function=ef)

    results = collection.query(query_texts=[user_input], n_results=3)

    valid_cases = []
    if 'distances' in results and results['distances']:
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            if dist < 1.2:
                valid_cases.append(f"- Case (Labeled as {meta['label']} | Distance: {dist:.2f}): '{doc[:300]}...'")

    if valid_cases:
        similar_cases_context = "Similar Historical Cases from Database:\n\n" + "\n".join(valid_cases)
    else:
        similar_cases_context = "Insufficient historical data to assess risk (No high-confidence matches found)."

    print("--- RAG Context Extracted ---")
    print(similar_cases_context + "\n")
except Exception as e:
    print(f"Error in RAG: {e}")
    similar_cases_context = ""

# 3. LLM API Call
load_dotenv(".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

prompt = f"""
You are an expert AI Psychological and Public Health Analyst. 
Analyze the following social media input text for ANY signals of emotional distress, relapse, or substance abuse.

User Input:
"{user_input}"

{similar_cases_context}

Task:
1. Evaluate the temporal context and user intent carefully. Differentiate between 'Active Risk' (current distress, cravings, immediate relapse signals) and 'Retrospective/Recovery Narratives' (discussing past addiction, sharing recovery stories, or promoting content like blogs/podcasts).
2. Provide a concise 'Risk Summary' explaining the risk level. **CRITICAL:** If the post is primarily reflective, promotional, or indicates stable recovery, explicitly state that this is a retrospective/recovery discussion, not an active crisis, and override/lower the assessed risk level.
3. Provide 'Supporting Evidence', explicitly quoting specific phrases from the User Input that led to your conclusion.
4. If available, briefly note if the User shares behavioral patterns with the Similar Historical Cases provided.

Format your response clearly. Keep it professional and empathetic.
"""

try:
    response = model.generate_content(prompt)
    print("--- Gemini LLM Final Summary ---")
    print(response.text)
except Exception as e:
    print(f"API Error: {e}")
