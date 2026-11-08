from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="recsys_two_stage_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:
    t1 = BashOperator(task_id="clean_and_split", bash_command="python pipelines/01_clean_and_split.py")
    t2 = BashOperator(task_id="spark_features", bash_command="python pipelines/02_spark_features.py")
    t3 = BashOperator(task_id="train_retrieval_faiss", bash_command="python pipelines/03_train_retrieval_and_faiss.py")
    t4 = BashOperator(task_id="train_rankers", bash_command="python pipelines/04_train_ranker.py")
    t5 = BashOperator(task_id="update_cold_start_and_eval", bash_command="python pipelines/05_cold_start_and_eval.py")

    t1 >> t2 >> [t3, t4] >> t5
