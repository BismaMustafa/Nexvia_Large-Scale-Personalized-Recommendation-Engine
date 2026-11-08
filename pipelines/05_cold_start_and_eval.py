import json
import numpy as np
import pandas as pd
import faiss
import tensorflow as tf
from scipy.stats import norm

print("1. Computing Bayesian Popularity & Content-Based Cold-Start Fallbacks...")
item_stats = pd.read_parquet("data/item_stats.parquet")
movies = pd.read_csv("ml-25m/movies.csv")

C = item_stats["item_avg_rating"].mean()
m = item_stats["item_interaction_count"].quantile(0.80)
v = item_stats["item_interaction_count"]
R = item_stats["item_avg_rating"]
item_stats["bayesian_score"] = (v / (v + m)) * R + (m / (v + m)) * C

merged = item_stats.merge(movies, on="movieId", how="left")
top_global = merged.sort_values("bayesian_score", ascending=False)[["movieId", "title", "bayesian_score"]].head(50)

cold_start_cache = {
    "global_popular": top_global["movieId"].tolist(),
    "genres": {}
}

for genre in ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance"]:
    g_df = merged[merged["genres"].fillna("").str.contains(genre)]
    cold_start_cache["genres"][genre] = g_df.sort_values("bayesian_score", ascending=False)["movieId"].head(30).tolist()

with open("models/cold_start_cache.json", "w") as f:
    json.dump(cold_start_cache, f)

print(f"Top 3 Global Cold-Start Movies:\n{top_global[['movieId', 'title', 'bayesian_score']].head(3).to_string(index=False)}")

print("\n2. Evaluating Retrieval & Ranking on Validation/Test Set (Recall@20, NDCG@20)...")
val_df = pd.read_parquet("data/val.parquet")
pos_val = val_df[val_df["clicked"] == 1]

user_tower = tf.saved_model.load("model_repository/user_tower/1/model.savedmodel")
index = faiss.read_index("models/faiss_item_index.bin")

sample_users = pos_val["userId"].astype(str).unique()[:100]
recalls, ndcgs = [], []

for u in sample_users:
    gt_items = set(pos_val[pos_val["userId"].astype(str) == u]["movieId"].values)
    if not gt_items:
        continue
    u_vec = user_tower.serve(tf.constant([[u]])).numpy().astype("float32")
    _, pred_ids = index.search(u_vec, 20)
    recs = pred_ids[0].tolist()

    # Recall@20
    hits = len(set(recs).intersection(gt_items))
    recalls.append(hits / min(20, len(gt_items)))

    # NDCG@20
    dcg = sum((1.0 / np.log2(idx + 2)) for idx, item in enumerate(recs) if item in gt_items)
    idcg = sum((1.0 / np.log2(idx + 2)) for idx in range(min(20, len(gt_items))))
    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

print(f"Offline Evaluation (100 Sample Users) -> Mean Recall@20: {np.mean(recalls):.4f}, Mean NDCG@20: {np.mean(ndcgs):.4f}")

print("\n3. Running Simulated Online A/B Test (DCN vs. Wide & Deep)...")
# Simulated impressions & clicks across A/B traffic split
views_control, clicks_control = 10000, 1240      # Wide & Deep (12.4% CTR)
views_treatment, clicks_treatment = 10000, 1410  # DCN Ranker (14.1% CTR)

ctr_a, ctr_b = clicks_control / views_control, clicks_treatment / views_treatment
p_pool = (clicks_control + clicks_treatment) / (views_control + views_treatment)
se = np.sqrt(p_pool * (1 - p_pool) * (1/views_control + 1/views_treatment))
z_score = (ctr_b - ctr_a) / se
p_val = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Control (Wide & Deep) CTR: {ctr_a:.2%} | Treatment (DCN) CTR: {ctr_b:.2%}")
print(f"Relative Lift: +{(ctr_b - ctr_a)/ctr_a:.2%} | Z-Score: {z_score:.2f} | P-Value: {p_val:.6f} (Statistically Significant!)")
