from datetime import datetime, timedelta
import numpy as np
import torch
from omegaconf import OmegaConf
import os
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import XCom
from airflow.utils.session import create_session

from dags.sql import user_embedding, track_embedding
from dags.utils import Config, Directory
from LightGCN.code.model import LightGCN
from LightGCN.code.batch_dataloader import Loader
from LightGCN.code import utils

def save_embeddings(user_embs, item_embs, **context):
    # Save to mounted volume

    date_str = datetime.now().strftime('%y-%m-%d')
    save_dir = os.path.join(Directory.LIGHTGCN_DIR, f"embeddings/{date_str}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save embeddings
    torch.save(user_embs, os.path.join(save_dir, 'user_embeddings.pt'))
    torch.save(item_embs, os.path.join(save_dir, 'item_embeddings.pt'))
    print("save the embeddings successfully!")
    # Push only the paths to XCom
    context['task_instance'].xcom_push(
        key='embeddings_path', 
        value={'user': os.path.join(save_dir, 'user_embeddings.pt'),
               'item': os.path.join(save_dir, 'item_embeddings.pt')}
    )
    print("push embeddings to XCom successfully!")
    
def load_embeddings(**context):
    # Get paths from XCom
    paths = context['task_instance'].xcom_pull(task_ids='get_user_item_embedding_task', key='embeddings_path')
    
    # Load embeddings
    user_embs = torch.load(paths['user'])
    item_embs = torch.load(paths['item'])

    return user_embs, item_embs

def get_user_item_embedding(**context):
    config_path = os.path.join(Directory.LIGHTGCN_DIR, 'config.yaml')
    config = OmegaConf.load(config_path)
    print(f"config: {OmegaConf.to_yaml(config)}")

    weight_file_dir = os.path.join(Directory.LIGHTGCN_DIR, f'checkpoints/best_model.pth')
    dataset = Loader(config=config, path=os.path.join(Directory.LIGHTGCN_DIR, config.path.DATA))
    model = LightGCN(config, dataset)
    checkpoint = torch.load(weight_file_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("Load the model successfully!")
    
    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight

    save_embeddings(user_embs, item_embs, **context)

# User Embedding 임시 테이블 저장
def load_to_user_temp_table(**context):
    user_embs, _ = load_embeddings(**context)
    
    pg_hook = PostgresHook(postgres_conn_id='vector_db_postgres_connection')
    
    with pg_hook.get_conn() as conn:
        cur = conn.cursor()
        try:
            # 임시 테이블 생성 
            cur.execute("""
                CREATE TABLE temp_user_embeddings (
                    user_id SERIAL PRIMARY KEY,
                    user_emb vector(64)
                );
            """)

            print("Start to insert data to tmp database")
            print(f"total length : {len(user_embs)}")

            # 배치 삽입을 위한 데이터 준비
            values = [(str(emb.tolist()),) for emb in user_embs]
            
            # 배치 삽입 실행
            cur.executemany(
                """
                INSERT INTO temp_user_embeddings (user_emb)
                VALUES (%s::vector)
                """,
                values
            )

            conn.commit()
            print("Data insertion completed successfully")

        except Exception as e:
            conn.rollback()
            print(f"Error occurred: {str(e)}")
            raise e
        finally:
            cur.close()

# Track Embedding 임시 테이블 저장
def load_to_track_temp_table(**context):
   BATCH_SIZE = 10000  # 적절한 배치 크기 설정
   
   _, item_embs = load_embeddings(**context)
   item_id_list = pd.read_csv(os.path.join(Directory.AIRFLOW_HOME, "dags/data/gcn_track_id.csv"))['track_id'].tolist()
   pg_hook = PostgresHook(postgres_conn_id='vector_db_postgres_connection')
   
   with pg_hook.get_conn() as conn:
       cur = conn.cursor()
       try:
           # 임시 테이블 생성
           cur.execute("""
               CREATE TABLE temp_track_embeddings (
                   track_id INT PRIMARY KEY,
                   track_emb vector(64)
               );
           """)
           
           print("Start to insert data to tmp database")
           print(f"total length : {len(item_embs)}")
           
           # 전체 데이터 준비
           values = [(idx, str(emb.tolist())) for idx, emb in zip(item_id_list, item_embs)]
           
           # 배치 단위로 분할하여 삽입
           total_batches = (len(values) + BATCH_SIZE - 1) // BATCH_SIZE  # 총 배치 수 계산
           
           for i in range(0, len(values), BATCH_SIZE):
               batch = values[i:i + BATCH_SIZE]
               cur.executemany(
                   """
                   INSERT INTO temp_track_embeddings (track_id, track_emb)
                   VALUES (%s, %s::vector)
                   """,
                   batch
               )
               current_batch = i // BATCH_SIZE + 1
               print(f"Inserted batch {current_batch}/{total_batches}, "
                     f"rows {i} to {min(i + BATCH_SIZE, len(values))}")
               
               conn.commit()  # 각 배치마다 커밋
           
           print("Data insertion completed successfully")
           
       except Exception as e:
           conn.rollback()
           print(f"Error occurred: {str(e)}")
           raise e
       finally:
           cur.close()
           

def delete_xcoms_for_dags(dag_ids, **kwargs):
    with create_session() as session:
        session.query(XCom).filter(
            XCom.dag_id.in_(dag_ids)
        ).delete(synchronize_session=False)
        session.commit()



default_args = {
    'owner': 'airflow',
    'depends_on_past': False
}


with DAG('lightgcn_embedding_update_dag',
        default_args=default_args,
        schedule=None,
        start_date=datetime(2024, 2, 6),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_user_item_embedding_task = PythonOperator(
        task_id='get_user_item_embedding_task',
        python_callable=get_user_item_embedding,
        provide_context=True
    )

    load_to_user_temp_table_task = PythonOperator(
        task_id='load_to_user_temp_table_task',
        python_callable=load_to_user_temp_table,
        provide_context=True
    )

    upsert_user_embeddings_task = PostgresOperator(
        task_id='upsert_user_embeddings_task',
        postgres_conn_id='vector_db_postgres_connection',
        sql=user_embedding.upsert_sql
    )

    load_to_track_temp_table_task = PythonOperator(
        task_id='load_to_track_temp_table_task',
        python_callable=load_to_track_temp_table,
        provide_context=True
    )

    upsert_track_embeddings_task = PostgresOperator(
        task_id='upsert_track_embeddings_task',
        postgres_conn_id='vector_db_postgres_connection',
        sql=track_embedding.upsert_sql
    )

    delete_xcom_task = PythonOperator(
            task_id="delete_xcom_task",
            python_callable=delete_xcoms_for_dags,
            op_kwargs={'dag_ids': ['generate_user_embeddings_task', 'generate_track_embeddings_task']}
        )
    
    end_task = EmptyOperator(
        task_id = "end_task"
    )

    start_task >> get_user_item_embedding_task
    start_task >> get_user_item_embedding_task

    get_user_item_embedding_task >> load_to_user_temp_table_task >> upsert_user_embeddings_task
    get_user_item_embedding_task >> load_to_track_temp_table_task >> upsert_track_embeddings_task

    upsert_user_embeddings_task >> delete_xcom_task
    upsert_track_embeddings_task >> delete_xcom_task

    delete_xcom_task >> end_task