import os
from datetime import datetime
import pickle

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import XCom
from airflow.utils.session import create_session
from airflow.providers.google.cloud.hooks.gcs import GCSHook

from BiEncoder.src.preprocess import preprocess_data
from BiEncoder.src.models import SongEncoder, GenreEncoder
from BiEncoder.src.train import train_model
from dags.utils import Directory, Config

def save_data(data_songs, artist_list, **context):
    # Save to mounted volume
    data_songs_dir = os.path.join(Directory.BIENCODER_DIR, f"data/data_songs.pkl")
    artist_list_dir = os.path.join(Directory.BIENCODER_DIR, f"data/artist_list.pkl")
    
    with open(data_songs_dir, "wb") as f:
        pickle.dump(data_songs, f)

    with open(artist_list_dir, "wb") as f:
        pickle.dump(artist_list, f)
    print("save the embeddings successfully!")

    # Push only the paths to XCom
    context['task_instance'].xcom_push(
        key='data_path', 
        value={'data_songs': data_songs_dir,
               'artist_list': artist_list_dir}
    )
    print("push embeddings to XCom successfully!")
    
def load_embeddings(**context):
    # Get paths from XCom
    paths = context['task_instance'].xcom_pull(task_ids='get_data_preprocessing_task', key='data_path')
    
    with open(paths['data_songs'], 'rb') as f:
        data_songs = pickle.load(f)

    with open(paths['artist_list'], 'rb') as f:
        artist_list = pickle.load(f)

    return data_songs, artist_list

def get_data_preprocessing(**context):
    track_path = os.path.join(Directory.BIENCODER_DIR, "data/data_songs.pkl")
    artist_path = os.path.join(Directory.BIENCODER_DIR, "data/artist_list.pkl")
    if os.path.exists(track_path) and os.path.exists(artist_path):
        print("Loading existing preprocessed data...")
        with open(track_path, 'rb') as f:
            data_songs = pickle.load(f)

        with open(artist_path, 'rb') as f:
            artist_list = pickle.load(f)
        save_data(data_songs, artist_list, **context)
        print("push the data successfully!")
    else:
        print("Start to data preprocessing")
        # Configuration
        config_path = os.path.join(Directory.BIENCODER_DIR, 'config.yaml')
        directory_path = os.path.join(Directory.BIENCODER_DIR, 'checkpoints')

        save_path = os.path.join(directory_path, 'batch_model.pt')
        print(f"load and save to {save_path}")

        # Preprocess
        print("Start Data Preprocessing!")
        data_songs, artist_list = preprocess_data(config_path)

        save_data(data_songs, artist_list, **context)
        print("push the data successfully!")

def get_model_train(batch_size, num_epochs, margin, **context):
    data_songs, artist_list = load_embeddings(**context)
    save_path = os.path.join(Directory.BIENCODER_DIR, "checkpoints/batch_model.pt")

    print("Start model initialization")
    # Model initialization
    song_encoder = SongEncoder(
        artist_list=artist_list,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder(
        pretrained_name="distilbert-base-uncased",
        embed_dim=32
    )

    print("Start model training!")
    # Train with batch processing
    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        margin=margin, 
        save_path=save_path
    )

    print("start evaluation!")
    # Evaluation
    #evaluate_model(song_encoder, genre_encoder, data_songs)

def upload_file_to_gcs(bucket_name: str, replace: bool = True) -> None:
    """ Airflow Hook으로 GCS에 파일을 업로드하는 메서드
    Args:
        source_path (str): 업로드할 로컬 파일 경로
        destination_path (str): GCS에 저장될 객체 이름/경로
        bucket_name (str): GCS 버킷 이름
        replace (bool): 덮어쓰기 여부 (기본값: True)
    """
    hook = GCSHook(gcp_conn_id="google_cloud_default")
    
    if not replace and hook.exists(bucket_name=bucket_name, object_name=destination_path):
        print(f"Object {destination_path} already exists in bucket {bucket_name}")
        return

    today_dir = datetime.now().strftime('%y-%m-%d')
    save_path = os.path.join(Directory.BIENCODER_DIR, "checkpoints/batch_model.pt")

    destination_path = f'model/BiEncoder/{today_dir}/{today_dir}.pt'    
    hook.upload(
        bucket_name=bucket_name,
        object_name=destination_path,
        filename=save_path
    )

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


with DAG('biencoder_batch_train_dag',
        default_args=default_args,
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_data_preprocessing_task = PythonOperator(
        task_id='get_data_preprocessing_task',
        python_callable=get_data_preprocessing,
        provide_context=True
    )

    get_model_train_task = PythonOperator(
        task_id='get_model_train_task',
        python_callable=get_model_train,
        op_kwargs= {
            "batch_size": 32,
            "num_epochs": 1,
            "margin": 0.2
        },
        executor_config={
            "cpu_limit": "1000"
        },
        provide_context=True
    )

    upload_file_to_gcs_task = PythonOperator(
        task_id='upload_file_to_gcs_task',
        python_callable=upload_file_to_gcs,
        op_kwargs= {
            "bucket_name": "itsmerecsys"
        },
        provide_context=True
    )

    delete_xcom_task = PythonOperator(
            task_id="delete_xcom_task",
            python_callable=delete_xcoms_for_dags,
            op_kwargs={'dag_ids': ['get_data_preprocessing_task']}
        )
    
    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> get_data_preprocessing_task >> get_model_train_task >> upload_file_to_gcs_task >> delete_xcom_task >> end_task