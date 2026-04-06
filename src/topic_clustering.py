"""
topic_clustering.py — Temporal & Behavioral Pattern Analysis
Core Task 2 / Track A: NSF NRT Challenge 1

This script performs K-Means clustering on the ChromaDB embeddings to uncover
behavioral patterns and emerging themes within High Risk social media posts.
Approach: Embedding-based clustering (Track A style, not dashboard visualizations).

Usage:
    python src/topic_clustering.py

Outputs:
    data/cluster_results.csv        — per-post cluster assignment + top keywords
    data/cluster_visualization.png  — PCA 2D scatter plot colored by cluster
"""

import os
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH       = "./data/chroma_db"
COLLECTION    = "reddit_mental_health"
N_CLUSTERS    = 8
RANDOM_STATE  = 42
TOP_KW_PER_CLUSTER = 8
OUTPUT_CSV    = "data/cluster_results.csv"
OUTPUT_PNG    = "data/cluster_visualization.png"
MAX_POSTS     = 5000   # cap for speed — use a sample of High Risk posts

HIGH_RISK_LABELS = {"addiction", "alcoholism", "suicidewatch", "depression", "ptsd", "mentalhealth", "anxiety"}

def load_high_risk_embeddings(db_path: str, collection_name: str):
    """Pull High Risk documents + embeddings from ChromaDB."""
    print("🔗 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.get_collection(name=collection_name, embedding_function=ef)

    print("📥 Fetching all documents (this may take ~30s)...")
    # Fetch in batches to avoid memory issues
    total = col.count()
    print(f"   Total documents in DB: {total:,}")

    all_docs, all_metas, all_ids, all_embeddings = [], [], [], []

    batch_size = 2000
    offset = 0
    while offset < total:
        batch = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        all_ids.extend(batch["ids"])
        all_embeddings.extend(batch["embeddings"])
        offset += batch_size
        print(f"   Fetched {min(offset, total):,} / {total:,}")

    # Filter High Risk only
    hr_docs, hr_embeddings, hr_labels = [], [], []
    for doc, meta, emb in zip(all_docs, all_metas, all_embeddings):
        if meta.get("label", "") in HIGH_RISK_LABELS:
            hr_docs.append(doc)
            hr_labels.append(meta["label"])
            hr_embeddings.append(emb)

    print(f"✅ {len(hr_docs):,} High Risk posts loaded.")

    # Sub-sample if too large
    if len(hr_docs) > MAX_POSTS:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(len(hr_docs), MAX_POSTS, replace=False)
        hr_docs = [hr_docs[i] for i in idx]
        hr_labels = [hr_labels[i] for i in idx]
        hr_embeddings = [hr_embeddings[i] for i in idx]
        print(f"   Sub-sampled to {MAX_POSTS:,} posts for clustering speed.")

    return hr_docs, hr_labels, np.array(hr_embeddings, dtype=np.float32)


def run_kmeans(embeddings: np.ndarray, n_clusters: int):
    """Fit K-Means on the embedding matrix."""
    print(f"\n🤖 Running K-Means (k={n_clusters}) on {embeddings.shape[0]:,} posts × {embeddings.shape[1]} dims...")
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embeddings)
    inertia = km.inertia_
    print(f"   Converged. Inertia: {inertia:.2f}")
    return labels, km


def extract_cluster_keywords(docs: list, cluster_labels: np.ndarray, n_clusters: int, top_n: int = 8):
    """Get top TF-IDF keywords for each cluster."""
    print("\n🔑 Extracting top keywords per cluster via TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3
    )
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}
    for k in range(n_clusters):
        mask = cluster_labels == k
        if mask.sum() == 0:
            cluster_keywords[k] = []
            continue
        cluster_tfidf = X[mask].mean(axis=0).A1
        top_idx = cluster_tfidf.argsort()[-top_n:][::-1]
        cluster_keywords[k] = [feature_names[i] for i in top_idx]

    return cluster_keywords


def name_clusters(cluster_keywords: dict) -> dict:
    """Auto-generate a short theme name from top keywords."""
    cluster_names = {}
    for k, kws in cluster_keywords.items():
        if kws:
            # Use first 3 keywords as the label
            cluster_names[k] = f"C{k}: {' · '.join(kws[:3])}"
        else:
            cluster_names[k] = f"Cluster {k}"
    return cluster_names


def plot_clusters(embeddings: np.ndarray, cluster_labels: np.ndarray, cluster_names: dict, out_path: str):
    """PCA 2D scatter plot colored by cluster."""
    print("\n🎨 Generating PCA 2D cluster visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    n_clusters = len(cluster_names)
    cmap = cm.get_cmap("tab10", n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0e1117")
    ax.set_facecolor("#12121f")

    for k in range(n_clusters):
        mask = cluster_labels == k
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colors[k], s=8, alpha=0.55,
            label=cluster_names.get(k, f"C{k}")
        )

    ax.set_title(
        f"SubstanceWatch AI — High Risk Post Clusters (K-Means, k={n_clusters})\n"
        f"PCA Variance Explained: {var_explained:.1f}%",
        color="white", fontsize=13, pad=12
    )
    ax.set_xlabel("PC1", color="#aaa")
    ax.set_ylabel("PC2", color="#aaa")
    ax.tick_params(colors="#555")
    ax.spines[:].set_color("#333")

    legend = ax.legend(
        loc="upper right",
        fontsize=7.5,
        facecolor="#1a1a2e",
        labelcolor="white",
        markerscale=2.5,
        framealpha=0.9,
        title="Clusters",
        title_fontsize=8
    )
    legend.get_title().set_color("white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Saved: {out_path}")


def main():
    print("=" * 65)
    print("  SubstanceWatch AI — Behavioral Pattern Clustering (Core Task 2)")
    print("=" * 65)

    # 1. Load embeddings
    docs, labels_orig, embeddings = load_high_risk_embeddings(DB_PATH, COLLECTION)

    # 2. K-Means
    cluster_labels, km_model = run_kmeans(embeddings, N_CLUSTERS)

    # 3. Keywords
    cluster_keywords = extract_cluster_keywords(docs, cluster_labels, N_CLUSTERS, TOP_KW_PER_CLUSTER)
    cluster_names    = name_clusters(cluster_keywords)

    # 4. Visualization
    os.makedirs("data", exist_ok=True)
    plot_clusters(embeddings, cluster_labels, cluster_names, OUTPUT_PNG)

    # 5. Save CSV
    df_out = pd.DataFrame({
        "text":           docs,
        "original_label": labels_orig,
        "cluster_id":     cluster_labels,
        "cluster_name":   [cluster_names[k] for k in cluster_labels],
        "top_keywords":   [", ".join(cluster_keywords[k]) for k in cluster_labels],
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved: {OUTPUT_CSV}")

    # 6. Print Cluster Report
    print("\n" + "=" * 65)
    print("  CLUSTER REPORT")
    print("=" * 65)
    for k in range(N_CLUSTERS):
        n_in_cluster = (cluster_labels == k).sum()
        pct = n_in_cluster / len(cluster_labels) * 100
        kws = ", ".join(cluster_keywords[k])
        print(f"\n  Cluster {k} ({n_in_cluster} posts, {pct:.1f}%)")
        print(f"  Theme:    {cluster_names[k]}")
        print(f"  Keywords: {kws}")

        # Most common original labels in this cluster
        mask = np.array(cluster_labels) == k
        orig_in_cluster = [labels_orig[i] for i in range(len(labels_orig)) if mask[i]]
        from collections import Counter
        top_labels = Counter(orig_in_cluster).most_common(3)
        print(f"  Top original labels: {top_labels}")

    print(f"\n✅ Cluster analysis complete!")
    print(f"   → Visualization: {OUTPUT_PNG}")
    print(f"   → Full results:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
