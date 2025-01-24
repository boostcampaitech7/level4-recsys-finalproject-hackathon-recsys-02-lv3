from airflow import DAG
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup

from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import os
from glob import glob

from dags.utils import Config, Directory


def load_to_db(table, postgres_conn_id):
    # PostgresHook 연결 설정
    pg_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    
    # CSV 파일 읽기
    csv_path = os.path.join(Directory.TRANSFORM_DIR, f"spotify/{table}_table.csv")
    df = pd.read_csv(csv_path)
    
    # 현재 최대 track_id 가져오기
    max_id_query = f"SELECT COALESCE(MAX({table}_id), 0) FROM {table}"
    max_id = pg_hook.get_first(max_id_query)[0]
    print(f"max_id : {max_id}")
    # track_id 생성 및 데이터프레임에 추가
    df[f'{table}_id'] = list(range(max_id + 1, max_id + 1 + len(df)))
    
    # 데이터 삽입을 위한 리스트 생성
    rows = df.values.tolist()
    target_fields = df.columns.tolist()
    
    # 데이터 삽입
    pg_hook.insert_rows(
        table=table,
        rows=rows,
        target_fields=target_fields
    )


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG('load_to_db_dag',
        default_args=default_args,
        schedule='0 0 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    with TaskGroup("load_to_db_task") as load_to_db_task:
        src_path = os.path.join(Directory.TRANSFORM_DIR, "/spotify/*.csv")
        tables = ['track', 'artist', 'genre', 'track_artist', 'track_genre']
        for table in tables:
            get_info_task = PythonOperator(
                task_id=f"{table}_load_to_db",
                python_callable=load_to_db,
                op_kwargs={
                    "table": table,
                    "postgres_conn_id": Config.RDB_POSTGRES_CONNECTION
                }
            )

    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> load_to_db_task >> end_task