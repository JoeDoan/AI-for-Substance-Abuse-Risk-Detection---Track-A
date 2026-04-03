import joblib
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

clf = joblib.load("data/risk_classifier.pkl")
db_path = "./data/chroma_db"
client = chromadb.PersistentClient(path=db_path)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="reddit_mental_health", embedding_function=ef)

SCENARIOS = {
    "Scenario 1 — The Slippery Slope": (
        "Celebrating 5 years clean today. It feels surreal to even type that. The program works, guys. But I have to be honest, today is also the anniversary of my mom's passing. "
        "I'm sitting in my car outside the old neighborhood. I haven't parked here in half a decade. My brain keeps whispering that just one hit wouldn't reset the clock if nobody knows. "
        "I'm just so tired of being strong today. Just posting this to keep myself accountable, I guess. I haven't done anything yet, but the steering wheel is starting to feel real heavy."
    ),
    "Scenario 2 — The Screenwriter": (
        "Hey everyone, I'm a writer working on a screenplay about the opioid epidemic and I want to make sure I get the internal monologue right. In my current draft, the protagonist writes in his journal: "
        "'The cravings are a physical ache. I just want the noise to stop. I know the clinic is open, but my dealer is closer.' Does this sound authentic to those of you who have been through the darkest parts of withdrawal? "
        "I don't want to romanticize it, I want it to be raw and real. Thanks for your help and respect to everyone fighting the good fight."
    ),
    "Scenario 3 — The Slang & Metaphor": (
        "Everything is falling apart. Lost my job yesterday and my girl just walked out. I can't deal with this reality right now. The shadow is creeping back in and I'm honestly thinking about hitting the slopes tonight just to numb it all out. "
        "I know I promised I wouldn't go skiing ever again after last winter, but the sheer panic I'm feeling right now is choking me. Just need the white noise to take over. Tell me I'm being stupid."
    ),
}

PROMPT_TEMPLATE = """
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

results = {}

for name, text in SCENARIOS.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # ML
    prob = clf.predict_proba([text])[0][1]
    if prob > 0.4:
        ml_tag = "🟡 INITIAL SIGNAL DETECTED"
    elif prob > 0.2:
        ml_tag = "🟡 WEAK SIGNAL DETECTED"
    else:
        ml_tag = "⚪ NO INITIAL SIGNAL"
    print(f"\n[ML TRIPWIRE]\n  Tag: {ml_tag}\n  Raw probability: {prob:.4f}")

    # RAG
    rag_results = collection.query(query_texts=[text], n_results=3)
    valid = []
    if rag_results.get('distances'):
        for doc_text, meta, dist in zip(rag_results['documents'][0], rag_results['metadatas'][0], rag_results['distances'][0]):
            if dist < 1.2:
                valid.append(f"- Case (label={meta['label']}, distance={dist:.2f}): '{doc_text[:200]}...'")
    rag_context = ("Similar Historical Cases:\n" + "\n".join(valid)) if valid else "Insufficient historical data to assess risk."
    print(f"\n[RAG CONTEXT]\n{rag_context[:400]}...")

    # LLM
    prompt = PROMPT_TEMPLATE.format(user_input=text, similar_cases_context=rag_context)
    response = model.generate_content(prompt)
    llm_output = response.text
    print(f"\n[GEMINI LLM OUTPUT]\n{llm_output}")

    results[name] = {"ml_tag": ml_tag, "ml_prob": prob, "llm": llm_output, "rag": rag_context}

print("\n\n✅ All 3 adversarial tests completed.")
