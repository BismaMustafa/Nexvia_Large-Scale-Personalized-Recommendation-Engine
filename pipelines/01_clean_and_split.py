import pandas as pd

print("1. Loading MovieLens 25M dataset (ratings.csv)...")
# Memory bachane aur fast testing ke liye pehle 2 million rows le rahe hain
# (Agar aapka PC 16GB+ RAM wala hai aur poora 25M chalana chahein toh nrows=2_000_000 hata dein)
ratings = pd.read_csv("ml-25m/ratings.csv", nrows=2_000_000)

print("2. Filtering sparse users and items (>= 5 interactions)...")
user_counts = ratings["userId"].value_counts()
item_counts = ratings["movieId"].value_counts()
ratings = ratings[
    ratings["userId"].isin(user_counts[user_counts >= 5].index) &
    ratings["movieId"].isin(item_counts[item_counts >= 5].index)
].copy()

ratings["clicked"] = (ratings["rating"] >= 3.5).astype(int)
ratings = ratings.sort_values("timestamp")

print("3. Splitting data by time (80% Train, 10% Val, 10% Test)...")
t_val = ratings["timestamp"].quantile(0.80)
t_test = ratings["timestamp"].quantile(0.90)

train_df = ratings[ratings["timestamp"] < t_val]
val_df = ratings[(ratings["timestamp"] >= t_val) & (ratings["timestamp"] < t_test)]
test_df = ratings[ratings["timestamp"] >= t_test]

train_df.to_parquet("data/train.parquet", index=False)
val_df.to_parquet("data/val.parquet", index=False)
test_df.to_parquet("data/test.parquet", index=False)
print(f"Done! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} rows saved in data/ folder.")
