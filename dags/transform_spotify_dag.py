from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from airflow.models.variable import Variable
from airflow.models import XCom
from airflow.utils.session import create_session
from airflow.providers.google.cloud.hooks.gcs import GCSHook

import pandas as pd
import ast
import os
from datetime import datetime, timedelta
import time

from dags.utils import Directory, Config

def get_total_data():
    date = datetime.now().strftime("%Y-%m-%d")
    src_pth = os.path.join(Directory.DOWNLOADS_DIR, f"spotify/{date}.csv")
    df = pd.read_csv(src_pth)
    return df

def transform_track(**context):
    df = get_total_data()
    track_table = df[['track', 'last_fm_url', 'listeners', 'length', 'introduction']]
    track_table.drop_duplicates(subset = 'last_fm_url', inplace = True)
    track_table.insert(0, 'track_id', range(1, len(track_table) + 1))
    track_table.reset_index(drop=True)
    path = os.path.join(Directory.TRANSFORM_DIR, 'spotify/track_table.csv')
    track_table.to_csv(path, index=False)
    context['task_instance'].xcom_push(key='track_table', value=track_table.to_dict())
    print("Transform track successfully!")

    return track_table


def transform_artist(**context):
    df = get_total_data()
    tmp_artist_table = df[['artist']]
    tmp_artist_table['artist'] = tmp_artist_table['artist'].str.split('[&]').apply(lambda x: [name.strip() for name in x])
    tmp_artist_table = tmp_artist_table.explode("artist")
    artist_list = tmp_artist_table.artist.unique()
    artist_table = pd.DataFrame({"artist_id" : range(1, len(artist_list) + 1), "artist" : artist_list})
    
    path = os.path.join(Directory.TRANSFORM_DIR, 'spotify/artist_table.csv')
    artist_table.to_csv(path, index=False)
    context['task_instance'].xcom_push(key=f'artist_table', value=artist_table.to_dict())
    print("Transform artist successfully!")

    return artist_table


def transform_genre(**context):
    df = get_total_data()
    tmp_genre_table = df[['genres']]
    tmp_genre_table.dropna(inplace=True)
    tmp_genre_table['genres'] = tmp_genre_table['genres'].apply(ast.literal_eval)
    tmp_genre_table = tmp_genre_table.explode("genres")

    genre_list = tmp_genre_table['genres'].unique().tolist()
    genre_table = pd.DataFrame({'genre_id' : range(1, len(genre_list)+ 1), 'genre' : genre_list})

    path = os.path.join(Directory.TRANSFORM_DIR, 'spotify/genre_table.csv')
    genre_table.to_csv(path, index=False)
    context['task_instance'].xcom_push(key=f'genre_table', value=genre_table.to_dict())
    print("Transform genre successfully!")

    return genre_table

def tranform_track_artist(**context):
    time.sleep(5)
    df = get_total_data()
    track_table = context['ti'].xcom_pull(key="track_table")
    artist_table = context['ti'].xcom_pull(key="artist_table")
    track_table = pd.DataFrame.from_dict(track_table)
    artist_table = pd.DataFrame.from_dict(artist_table)
    print("HI!")
    print(track_table.columns)
    print(artist_table.columns)

    track_artist_table = df[['track', 'artist', 'last_fm_url']]
    track_artist_table['artist'] = track_artist_table['artist'].str.split('[&]').apply(lambda x: [name.strip() for name in x])

    track_artist_table = track_artist_table.explode("artist")

    track_artist_table['artist_id'] = track_artist_table['artist'].map(artist_table.set_index('artist')['artist_id'])
    track_artist_table['track_id'] = track_artist_table['last_fm_url'].map(track_table.set_index('last_fm_url')['track_id'])

    track_artist_table.drop(['track', 'artist', 'last_fm_url'], axis = 1, inplace = True)
    track_artist_table.insert(0, 'track_artist_id', range(1, len(track_artist_table) + 1))
    track_artist_table.reset_index(drop=True)
    path = os.path.join(Directory.TRANSFORM_DIR, 'spotify/track_artist_table.csv')
    track_artist_table.to_csv(path, index=False)
    print("Transform track_artist successfully!")

def tranform_track_genre(**context):
    time.sleep(5)
    df = get_total_data()
    track_table = context['ti'].xcom_pull(key="track_table")
    genre_table = context['ti'].xcom_pull(key="genre_table")
    track_table = pd.DataFrame.from_dict(track_table)
    genre_table = pd.DataFrame.from_dict(genre_table)


    track_genre_table = df[['track', 'last_fm_url', 'genres']]
    track_genre_table.rename(columns={'genres' : 'genre'}, inplace=True)
    track_genre_table = track_genre_table.dropna(subset=['genre'])

    track_genre_table['genre'] = track_genre_table['genre'].apply(ast.literal_eval)
    track_genre_table = track_genre_table.explode("genre")

    track_genre_table['genre_id'] = track_genre_table['genre'].map(genre_table.set_index("genre")['genre_id'])
    track_genre_table['track_id'] = track_genre_table['last_fm_url'].map(track_table.set_index("last_fm_url")['track_id'])

    track_genre_table.drop(['track', 'last_fm_url'], axis = 1, inplace = True)
    track_genre_table.insert(0, "track_genre_id", range(1, len(track_genre_table) + 1))

    path = os.path.join(Directory.TRANSFORM_DIR, 'spotify/track_genre_table.csv')
    track_genre_table.to_csv(path, index=False)
    print("Transform track_genre successfully!")

def upload_file_to_gcs(bucket_name: str, replace: bool = True) -> None:
    hook = GCSHook(gcp_conn_id="google_cloud_default")

    tables = ['track_table', 'artist_table', 'track_artist_table', 'genre_table', 'track_genre_table']
    for table in tables:
        filename = os.path.join(Directory.TRANSFORM_DIR, f'spotify/{table}.csv')
        destination_path = f'transform/spotify/{table}.csv'    
        
        hook.upload(
            bucket_name=bucket_name,
            object_name=destination_path,
            filename=filename
        )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG('transform_spotify_dag',
        default_args=default_args,
        schedule='0 0 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    transform_track_task = PythonOperator(
        task_id='transform_track_task',
        python_callable=transform_track,
        provide_context=True
    )

    transform_artist_task = PythonOperator(
        task_id='transform_artist_task',
        python_callable=transform_artist,
        provide_context=True
    )

    transform_genre_task = PythonOperator(
        task_id='transform_genre_task',
        python_callable=transform_genre,
        provide_context=True
    )

    tranform_track_artist_task = PythonOperator(
        task_id='tranform_track_artist_task',
        python_callable=tranform_track_artist,
        provide_context=True
    )

    tranform_track_genre_task = PythonOperator(
        task_id='tranform_track_genre_task',
        python_callable=tranform_track_genre,
        provide_context=True
    )

    upload_file_to_gcs_task = PythonOperator(
        task_id='upload_file_to_gcs_task',
        python_callable=upload_file_to_gcs,
        op_kwargs= {
            "bucket_name": Config.BUCKET_NAME
        },
        provide_context=True
    )

    load_to_db_trigger_task = TriggerDagRunOperator(
        task_id='load_to_db_trigger_task',
        trigger_dag_id='load_to_db_dag',
        trigger_run_id=None,
        execution_date=None,
        reset_dag_run=False,
        wait_for_completion=False,
        poke_interval=60,
        allowed_states=["success"],
        failed_states=None,
    )

    end_task = EmptyOperator(
        task_id="end_task"
    )


    start_task >> transform_track_task >> upload_file_to_gcs_task
    start_task >> transform_artist_task >> upload_file_to_gcs_task
    start_task >> transform_genre_task >> upload_file_to_gcs_task
    start_task >> tranform_track_artist_task >> upload_file_to_gcs_task
    start_task >> tranform_track_genre_task >> upload_file_to_gcs_task

    upload_file_to_gcs_task >> end_task