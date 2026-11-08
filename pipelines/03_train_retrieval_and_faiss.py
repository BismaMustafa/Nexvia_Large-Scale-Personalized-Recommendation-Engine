import tensorflow as tf
import numpy as np
import pandas as pd
import faiss

print("1. Loading positive interactions for Two-Tower Retrieval...")
df = pd.read_parquet("data/enriched_train.parquet")
# Fast training ke liye recent 300,000 positive interactions le rahe hain
pos_df = df[df["clicked"] == 1][["userId", "movieId"]].tail(300_000).astype(str)

unique_user_ids = pos_df["userId"].unique()
unique_movie_ids = df["movieId"].astype(str).unique()
print(f"Users in training: {len(unique_user_ids)}, Total catalog movies: {len(unique_movie_ids)}")

embedding_dim = 64

# User Tower
user_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_id")
u_lookup = tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None)(user_input)
u_emb = tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)(u_lookup)
u_flat = tf.keras.layers.Flatten()(u_emb)
u_dense = tf.keras.layers.Dense(embedding_dim, activation="relu")(u_flat)
u_norm = tf.keras.layers.UnitNormalization(axis=1, name="user_embedding")(u_dense)
user_tower = tf.keras.Model(inputs=user_input, outputs=u_norm, name="user_tower")

# Item Tower
item_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="movie_id")
i_lookup = tf.keras.layers.StringLookup(vocabulary=unique_movie_ids, mask_token=None)(item_input)
i_emb = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dim)(i_lookup)
i_flat = tf.keras.layers.Flatten()(i_emb)
i_dense = tf.keras.layers.Dense(embedding_dim, activation="relu")(i_flat)
i_norm = tf.keras.layers.UnitNormalization(axis=1, name="item_embedding")(i_dense)
item_tower = tf.keras.Model(inputs=item_input, outputs=i_norm, name="item_tower")

# Two-Tower In-Batch Softmax Contrastive Training Model
class TwoTowerModel(tf.keras.Model):
    def __init__(self, user_tower, item_tower, temperature=0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        users, items = data["userId"], data["movieId"]
        with tf.GradientTape() as tape:
            u_vec = self.user_tower(users, training=True)
            i_vec = self.item_tower(items, training=True)
            # Cosine similarity matrix across batch (Batch x Batch)
            logits = tf.matmul(u_vec, i_vec, transpose_b=True) / self.temperature
            labels = tf.range(tf.shape(users)[0])
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            )
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

print("2. Training Two-Tower Model (3 Epochs)...")
ds = tf.data.Dataset.from_tensor_slices({
    "userId": pos_df["userId"].values,
    "movieId": pos_df["movieId"].values
}).shuffle(50_000).batch(1024)

model = TwoTowerModel(user_tower, item_tower)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
model.fit(ds, epochs=3)

print("3. Saving User Tower for Serving...")
user_tower.export("model_repository/user_tower/1/model.savedmodel")

print("4. Computing Item Embeddings & Building FAISS Index...")
movies_ds = tf.data.Dataset.from_tensor_slices(unique_movie_ids).batch(1024)
item_embeddings = np.vstack([item_tower.predict(batch, verbose=0) for batch in movies_ds]).astype("float32")
id_map = np.array([int(x) for x in unique_movie_ids], dtype=np.int64)

quantizer = faiss.IndexFlatIP(embedding_dim)
index = faiss.IndexIDMap(quantizer)
index.add_with_ids(item_embeddings, id_map)
faiss.write_index(index, "models/faiss_item_index.bin")

# Quick test search
test_user_vec = user_tower.predict(tf.constant([unique_user_ids[0]]), verbose=0).astype("float32")
scores, top_ids = index.search(test_user_vec, 5)
print(f"Done! FAISS Index saved with {index.ntotal} items. Sample Top-5 Movie IDs for User {unique_user_ids[0]}: {top_ids[0]}")
