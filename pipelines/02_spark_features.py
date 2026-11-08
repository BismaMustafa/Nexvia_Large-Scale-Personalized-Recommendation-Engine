import os
import pandas as pd
import numpy as np

print("1. Loading train.parquet for Feature Engineering...")
df = pd.read_parquet("data/train.parquet")
df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)

print("2. Computing Point-in-Time User Rolling Features (Window = last 10 interactions)...")
# Shift(1) ensures we only use past ratings prior to the current timestamp (no data leakage)
df["user_avg_rating"] = (
    df.groupby("userId")["rating"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    .fillna(3.5)
)

print("3. Computing Item Popularity & Click Statistics...")
item_stats = df.groupby("movieId", as_index=False).agg(
    item_interaction_count=("rating", "count"),
    item_avg_rating=("rating", "mean"),
    item_positive_clicks=("clicked", "sum")
)

print("4. Joining features and saving enriched tables...")
enriched_train = df.merge(item_stats, on="movieId", how="left")
enriched_train.to_parquet("data/enriched_train.parquet", index=False)
item_stats.to_parquet("data/item_stats.parquet", index=False)

print(f"Done! Enriched train shape: {enriched_train.shape}, Unique items: {len(item_stats)}")
