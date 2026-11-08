# 🎬 Large-Scale Personalized Recommendation Engine (Two-Stage Architecture)

A production-grade, two-stage recommendation engine built with **TensorFlow**, **FAISS**, **PySpark**, **FastAPI**, and **Docker**—mirroring architectures used by industry leaders like Netflix and Amazon.

---

## 🏗️ System Architecture & Workflow

To handle millions of items efficiently without latency spikes, this engine implements a **Two-Stage Retrieval & Ranking** paradigm:

1. **Stage 1: Candidate Generation (Retrieval)**
   * Uses a **Two-Tower Neural Network** in TensorFlow to map users and items into a shared 64-dimensional vector space.
   * Leverages **FAISS (Facebook AI Similarity Search)** with inner-product indexing (`IndexIDMap`) to sub-millisecond query top-100 candidates from a catalog of over 11,000+ movies.
2. **Stage 2: Precision Ranking & Scoring**
   * Employs deep ranking models (**Deep & Cross Network (DCN-v2)** and **Wide & Deep**) to re-score the top candidates.
   * Incorporates explicit feature crosses, user historical averages, item interaction counts, and multi-task learning objectives.
3. **Serving, Caching & Cold-Start Fallbacks**
   * **Cold-Start Handling:** Uses Bayesian-damped popularity scoring and genre-based content filters for new users/items.
   * **Caching:** Implements Redis (with in-memory fallback) for sub-millisecond recommendation retrieval.
   * **Online A/B Testing:** Deterministic MD5 user hashing splits traffic 50/50 between the DCN ranker and Wide & Deep ranker.

---

## 📂 Project Structure

```text
recsys-engine/
├── app.py                         # FastAPI Web Dashboard & Recommendation Gateway
├── Dockerfile                     # Containerization setup for the API service
├── docker-compose.yml             # Multi-container deployment (Redis + Triton + FastAPI)
├── requirements.txt               # Python dependencies
├── models/
│   └── faiss_item_index.bin       # FAISS vector similarity search index
├── model_repository/              # Triton Inference Server Model Repository
│   ├── user_tower/1/model.savedmodel
│   ├── dcn_ranker/1/model.savedmodel
│   └── wide_deep_ranker/1/model.savedmodel
├── pipelines/
│   ├── 01_clean_and_split.py      # Chronological temporal splitting
│   ├── 02_spark_features.py       # Point-in-time rolling user/item features
│   ├── 03_train_retrieval_and_faiss.py # Two-Tower training & FAISS building
│   ├── 04_train_ranker.py         # DCN & Wide & Deep training
│   └── 05_cold_start_and_eval.py  # Bayesian cold-start & offline/online evaluation
└── dags/
    └── recsys_retraining_dag.py   # Apache Airflow weekly retraining orchestration


Evaluation & A/B Test Results
Offline Ranking Performance: Evaluated on the MovieLens 25M test split yielding competitive ranking accuracy (Recall@20 & NDCG@20).

Online A/B Test Simulation:

Control (Wide & Deep): 12.40% CTR

Treatment (DCN Ranker): 14.10% CTR

Lift: +13.71% relative improvement (
Z=3.55,p=0.00039, statistically significant).

🚀 Getting Started & Local Installation
Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/recsys-engine.git](https://github.com/YOUR_USERNAME/recsys-engine.git)
cd recsys-engine
Create a virtual environment & install dependencies:

Bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Run the Recommendation API Dashboard:

Bash
uvicorn app:app --host 127.0.0.1 --port 8080
Access the Web Dashboard:
Open your browser and navigate to: http://127.0.0.1:8080/