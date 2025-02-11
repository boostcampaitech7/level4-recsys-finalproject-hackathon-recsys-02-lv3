from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import XCom
from airflow.utils.session import create_session

from datetime import timedelta, datetime
import os
from omegaconf import OmegaConf
import psycopg2

from LightGCN.code.batch_Procedure import get_inter_data, generate_temporary_user
from dags.utils import Directory, Config

def get_embedding_vector(**context):
    batch_data = get_inter_data(Config.MONGO_DB_NAME, Config.MONGO_DB_PASSWORD)
    print("Getting batch data successfully!")
    config_path = os.path.join(Directory.LIGHTGCN_DIR, 'config.yaml')
    config = OmegaConf.load(config_path)

    ### 전일에 적재된 유저들에 대한 임시 임베딩 생성 ###
    update_list = generate_temporary_user(config, Directory.LIGHTGCN_DIR, batch_data)
    update_list = [
        {
            'uid': item['uid'],
            'u_emb': item['u_emb'].tolist()  # tensor를 numpy로 변환
        }
        for item in update_list
    ]

    print("Getting update_list successfully!")
    context['task_instance'].xcom_push(key = "update_list", value = update_list)

def load_embedding_to_table(**context):
    update_list = context['task_instance'].xcom_pull(task_ids = 'get_embedding_vector_task', key = "update_list")

    conn = psycopg2.connect(
        dbname=Config.VECTOR_DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST,
        port=Config.DB_PORT
    )
    cur = conn.cursor()

    for user in update_list:
        user_id = user['uid']
        new_embedding = user['u_emb']
        update_query = """
            UPDATE user_embedding 
            SET user_emb = %s
            WHERE user_id = %s
        """
        cur.execute(update_query, (new_embedding, user_id))
    conn.commit()
    print("update embedding vector successfully!")

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


with DAG('lightgcn_daily_train_dag',
        default_args=default_args,
        schedule='0 15 * * *',  # 하루에 한번 한국시간(KST, UTC+9)00시에 배치학습
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_embedding_vector_task = PythonOperator(
        task_id = 'get_embedding_vector_task',
        python_callable = get_embedding_vector,
        provide_context = True
    )

    load_embedding_to_table_task = PythonOperator(
        task_id = 'load_embedding_to_table_task',
        python_callable = load_embedding_to_table,
        provide_context = True
    )

    delete_xcom_task = PythonOperator(
            task_id="delete_xcom_task",
            python_callable=delete_xcoms_for_dags,
            op_kwargs={'dag_ids': ['get_embedding_vector_task']}
        )

    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> get_embedding_vector_task >> load_embedding_to_table_task >> delete_xcom_task >> end_task