import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# 1. Load and Sub-sample Data
print("Loading HuggingFace dataset...")
ds = load_dataset("kamruzzaman-asif/reddit-mental-health-classification", split="train")
df = ds.to_pandas()

print("Initial shape:", df.shape)

# Create the subset based on stratified rules
addiction_dfs = []

# All addiction & alcoholism
addiction_dfs.append(df[df['label'] == 'addiction'])
addiction_dfs.append(df[df['label'] == 'alcoholism'])

# 5,000 each from emotional distress classes
distress_classes = ['depression', 'suicidewatch', 'anxiety', 'ptsd', 'mentalhealth']
for cls in distress_classes:
    cls_df = df[df['label'] == cls]
    # Sample up to 5000 (if less, take all)
    n = min(len(cls_df), 5000)
    addiction_dfs.append(cls_df.sample(n=n, random_state=42))

# 5,000 total from safe/control classes
safe_classes = ['jokes', 'parenting', 'fitness', 'personalfinance']
safe_df = df[df['label'].isin(safe_classes)]
addiction_dfs.append(safe_df.sample(n=5000, random_state=42))

# Combine, shuffle, and clean
subset_df = pd.concat(addiction_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
subset_df['text'] = subset_df['text'].fillna("").astype(str)

# Clean: Shorten texts that are absurdly long, remove empty
subset_df = subset_df[subset_df['text'].str.strip().str.len() > 10].copy()
subset_df['text'] = subset_df['text'].apply(lambda x: x[:2000]) # cap string length for embedding speed
print("Final subset shape:", subset_df.shape)

# Save subset to disk for classification model
subset_csv_path = "data/stratified_subset.csv"
subset_df.to_csv(subset_csv_path, index=False)
print(f"Saved subset to {subset_csv_path}")

# 2. Build ChromaDB Vector Database
print("\nInitializing ChromaDB Persistent Client...")
db_path = "./data/chroma_db"
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

# Delete collection if it already exists to avoid duplication
try:
    client.delete_collection("reddit_mental_health")
except Exception:
    pass

# We will let ChromaDB use the default all-MiniLM-L6-v2 embedding function under the hood
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="reddit_mental_health", embedding_function=ef)

docs = subset_df['text'].tolist()
# Metadata should just contain the label
metadatas = [{"label": label} for label in subset_df['label'].tolist()]
ids = [str(i) for i in range(len(subset_df))]

print("Embedding and adding to Vector DB (this may take a few minutes)...")
batch_size = 2000
for i in tqdm(range(0, len(docs), batch_size)):
    end_idx = min(i + batch_size, len(docs))
    collection.add(
        documents=docs[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx]
    )

print("Vector DB successfully built!")
