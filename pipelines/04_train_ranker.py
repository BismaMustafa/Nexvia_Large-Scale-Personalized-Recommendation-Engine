import tensorflow as tf
import pandas as pd
import numpy as np

print("1. Loading enriched training data for Stage-2 Ranking...")
df = pd.read_parquet("data/enriched_train.parquet").tail(300_000).copy()

df["userId"] = df["userId"].astype(str)
df["movieId"] = df["movieId"].astype(str)
df["log_item_count"] = np.log1p(df["item_interaction_count"].values).astype("float32")
df["user_avg_rating"] = df["user_avg_rating"].astype("float32")
df["item_avg_rating"] = df["item_avg_rating"].astype("float32")
labels = df["clicked"].values.astype("float32")

unique_users = df["userId"].unique()
unique_movies = df["movieId"].unique()

ds = tf.data.Dataset.from_tensor_slices((
    {
        "user_id": df["userId"].values,
        "movie_id": df["movieId"].values,
        "user_avg_rating": df["user_avg_rating"].values,
        "item_avg_rating": df["item_avg_rating"].values,
        "log_item_count": df["log_item_count"].values,
    },
    labels
)).shuffle(50_000).batch(1024)

def build_inputs_and_embeddings():
    u_in = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_id")
    m_in = tf.keras.Input(shape=(1,), dtype=tf.string, name="movie_id")
    u_avg = tf.keras.Input(shape=(1,), dtype=tf.float32, name="user_avg_rating")
    i_avg = tf.keras.Input(shape=(1,), dtype=tf.float32, name="item_avg_rating")
    i_cnt = tf.keras.Input(shape=(1,), dtype=tf.float32, name="log_item_count")

    u_idx = tf.keras.layers.StringLookup(vocabulary=unique_users, mask_token=None)(u_in)
    u_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(unique_users) + 1, 32)(u_idx))

    m_idx = tf.keras.layers.StringLookup(vocabulary=unique_movies, mask_token=None)(m_in)
    m_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(unique_movies) + 1, 32)(m_idx))

    x0 = tf.keras.layers.Concatenate()([u_emb, m_emb, u_avg, i_avg, i_cnt])
    inputs = [u_in, m_in, u_avg, i_avg, i_cnt]
    return inputs, x0

# --- Model 1: Deep & Cross Network (DCN) ---
print("2. Building and Training DCN Ranker (Treatment Model)...")
inputs_dcn, x0_dcn = build_inputs_and_embeddings()

# Explicit Cross Network Layer
cross_proj = tf.keras.layers.Dense(x0_dcn.shape[-1], use_bias=True)(x0_dcn)
x_cross = tf.keras.layers.Multiply()([x0_dcn, cross_proj])
x_cross = tf.keras.layers.Add()([x0_dcn, x_cross])

# Deep MLP Path
x_deep = tf.keras.layers.Dense(128, activation="relu")(x0_dcn)
x_deep = tf.keras.layers.Dropout(0.2)(x_deep)
x_deep = tf.keras.layers.Dense(64, activation="relu")(x_deep)

combined_dcn = tf.keras.layers.Concatenate()([x_cross, x_deep])
out_dcn = tf.keras.layers.Dense(1, activation="sigmoid", name="click_prob")(combined_dcn)

dcn_model = tf.keras.Model(inputs=inputs_dcn, outputs=out_dcn, name="dcn_ranker")
dcn_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")]
)
dcn_model.fit(ds, epochs=2)
dcn_model.export("model_repository/dcn_ranker/1/model.savedmodel")

# --- Model 2: Wide & Deep Ranker (Control Model for A/B Testing) ---
print("3. Building and Training Wide & Deep Ranker (Control Model)...")
inputs_wd, x0_wd = build_inputs_and_embeddings()
wide_out = tf.keras.layers.Dense(16, activation="linear")(x0_wd)
deep_out = tf.keras.layers.Dense(128, activation="relu")(x0_wd)
deep_out = tf.keras.layers.Dense(64, activation="relu")(deep_out)
combined_wd = tf.keras.layers.Concatenate()([wide_out, deep_out])
out_wd = tf.keras.layers.Dense(1, activation="sigmoid", name="click_prob")(combined_wd)

wd_model = tf.keras.Model(inputs=inputs_wd, outputs=out_wd, name="wide_deep_ranker")
wd_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")]
)
wd_model.fit(ds, epochs=2)
wd_model.export("model_repository/wide_deep_ranker/1/model.savedmodel")

print("Done! Both DCN and Wide & Deep Ranking models trained and exported!")
