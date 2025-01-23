from datetime import datetime, timedelta
import numpy as np
import torch

# 3. Airflow 관련 import
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import XCom
from airflow.utils.session import create_session

# 4. 로컬 모듈 import
from dags.sql import user_embedding, track_embedding
from LightGCN.code.model import LightGCN
from LightGCN.code.dataloader import Loader


def get_user_item_embeddding(**context):
    ### config ###
    config = {}
    # config['batch_size'] = 4096
    config['bpr_batch_size'] = 2048
    config['latent_dim_rec'] = 64
    config['lightGCN_n_layers']= 3
    config['dropout'] = 0.001
    config['keep_prob']  = 0.6
    config['A_n_fold'] = 100
    config['test_u_batch_size'] = 1000
    config['multicore'] = 0
    config['lr'] = 0.001
    config['decay'] = 1e-4
    config['pretrain'] = 0
    config['A_split'] = False
    config['bigdata'] = False
    config['shuffle'] = 'shuffle'

    print("Load the dataset")
    dataset = Loader(path="../data/spotify")
    print("Load the model")
    model = LightGCN(config, dataset)
    checkpoint = torch.load('/Users/mac/level4-recsys-finalproject-hackathon-recsys-02-lv3/LightGCN/code/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight

    context['task_instance'].xcom_push(key='user_embeddings_data', value=user_embs)
    context['task_instance'].xcom_push(key='track_embeddings_data', value=item_embs)

# User Embedding 임시 테이블 저장
def load_to_user_temp_table(**context):
    embeddings_data = context['task_instance'].xcom_pull(
        task_ids='get_user_item_embeddding_task',
        key='user_embeddings_data'
    )
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_connection')
    
    with pg_hook.get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE temp_user_embeddings (
                    user_id INT PRIMARY KEY
                    user_emb vector(64)
                );
            """)
            
            for data in embeddings_data:
                cur.execute(
                    """
                    INSERT INTO temp_user_embeddings (user_id, user_emb)
                    VALUES (%s, %s, %s::vector)
                    """,
                    (data['user_id'], str(data['user_emb']))
                )
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()

# Track Embedding 임시 테이블 저장
def load_to_track_temp_table(**context):
    embeddings_data = context['task_instance'].xcom_pull(
        task_ids='get_user_item_embeddding_task',
        key='track_embeddings_data'
    )

    pg_hook = PostgresHook(postgres_conn_id='postgres_connection')
    
    with pg_hook.get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE temp_track_embeddings (
                    track_id INT PRIMARY KEY,
                    track_emb vector(64)
                );
            """)
            
            for data in embeddings_data:
                cur.execute(
                    """
                    INSERT INTO temp_track_embeddings (track_id, track_emb)
                    VALUES (%s, %s::vector)
                    """,
                    (data['track_id'], str(data['track_emb']))
                )
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
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
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG('user_embedding_update_dag',
        default_args=default_args,
        schedule='0 0 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_user_item_embeddding_task = PythonOperator(
        task_id='get_user_item_embeddding_task',
        python_callable=get_user_item_embeddding,
        provide_context=True
    )

    load_to_user_temp_table_task = PythonOperator(
        task_id='load_to_user_temp_table_task',
        python_callable=load_to_user_temp_table,
        provide_context=True
    )

    upsert_user_embeddings_task = PostgresOperator(
        task_id='upsert_user_embeddings_task',
        postgres_conn_id='postgres_connection',
        sql=user_embedding.upsert_sql
    )

    load_to_track_temp_table_task = PythonOperator(
        task_id='load_to_track_temp_table_task',
        python_callable=load_to_track_temp_table,
        provide_context=True
    )

    upsert_track_embeddings_task = PostgresOperator(
        task_id='upsert_track_embeddings_task',
        postgres_conn_id='postgres_connection',
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

    start_task >> get_user_item_embeddding_task
    start_task >> get_user_item_embeddding_task

    get_user_item_embeddding_task >> load_to_user_temp_table_task >> upsert_user_embeddings_task
    get_user_item_embeddding_task >> load_to_track_temp_table_task >> upsert_track_embeddings_task

    upsert_user_embeddings_task >> delete_xcom_task
    upsert_track_embeddings_task >> delete_xcom_task

    delete_xcom_task >> end_task