from datetime import datetime, timedelta
import numpy as np
import torch
from omegaconf import OmegaConf
import psycopg2
import logging
import os
import pickle
from tqdm import tqdm

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import XCom
from airflow.utils.session import create_session

from torch.utils.data import Dataset, DataLoader
from BiEncoder.src.preprocess import preprocess_data, load_config, connect_db
from BiEncoder.src.models import SongEncoder, GenreEncoder
from BiEncoder.src.train import load_model
from BiEncoder.src.eval import SongDataset
from dags.utils import Directory

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

def generate_and_save_embeddings(song_encoder, data_songs, config):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        song_encoder.to(device)
        song_encoder.eval()

        dataset = SongDataset(data_songs)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 진행상황 추적을 위한 변수 추가
        total_batches = len(dataloader)
        processed_batches = 0

        # DB 연결
        conn = psycopg2.connect(
            dbname=config['database_emb']['dbname'],
            user=config['database_emb']['user'],
            password=config['database_emb']['password'],
            host=config['database_emb']['host'],
            port=config['database_emb']['port']
        )
        cur = conn.cursor()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    artists = batch["artist"]
                    tracks = batch["track"]
                    playlists = batch["playlist"]
                    listeners = batch["listeners"].to(device)
                    lengths = batch["length"].to(device)
                    genres = batch["genres"]

                    # 임베딩 생성
                    embeddings = song_encoder(artists, tracks, playlists, listeners, lengths, genres)
                    
                    # DB 업데이트 준비
                    update_data = [
                        (batch_idx * dataloader.batch_size + i + 1, emb.cpu().numpy().tolist())
                        for i, emb in enumerate(embeddings)
                    ]

                    # 배치 업데이트 실행
                    cur.executemany(
                        """
                        INSERT INTO tmp_track_meta_embedding (track_id, track_emb) 
                        VALUES (%s, %s)
                        ON CONFLICT (track_id) 
                        DO UPDATE SET track_emb = EXCLUDED.track_emb
                        """,
                        update_data
                    )
                    
                    # 각 배치마다 커밋
                    conn.commit()
                    
                    processed_batches += 1
                    if processed_batches % 10 == 0:  # 10배치마다 로그
                        logging.info(f"Processed {processed_batches}/{total_batches} batches")
                        
                except Exception as batch_error:
                    logging.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                    continue

    except Exception as e:
        logging.error(f"Error in generate_and_save_embeddings: {str(e)}")
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
        logging.info(f"Completed processing {processed_batches}/{total_batches} batches")

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
        
def load_trained_model(**context):
    model_path = os.path.join(Directory.BIENCODER_DIR, "checkpoints/batch_model.pt")
    config_path = os.path.join(Directory.BIENCODER_DIR, 'config.yaml')

    config = load_config(config_path)
    data_songs, artist_list = load_embeddings(**context)

    # Initialize models
    song_encoder = SongEncoder(
        artist_list=artist_list,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder()

    # Load trained models
    load_model(song_encoder, genre_encoder, model_path)

    generate_and_save_embeddings(song_encoder, data_songs, config)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}


with DAG('biencoder_embedding_update_dag',
        default_args=default_args,
        schedule='0 0 * * *',
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

    load_trained_model_task = PythonOperator(
        task_id='load_trained_model_task',
        python_callable=load_trained_model,
        execution_timeout=timedelta(hours=2),
        provide_context=True
    )

    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> get_data_preprocessing_task >> load_trained_model_task >> end_task