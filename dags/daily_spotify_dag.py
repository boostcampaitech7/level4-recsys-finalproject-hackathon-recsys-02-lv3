from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from airflow.models.variable import Variable
from airflow.models import XCom
from airflow.utils.session import create_session
from airflow.providers.google.cloud.hooks.gcs import GCSHook

from datetime import datetime, timedelta
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup 
import os
from datetime import datetime
import pandas as pd
import psycopg2
import math
import logging

from dags.utils import Config, Directory

def get_soup(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_last_fm_url_from_db(**context):
    conn = psycopg2.connect(
            dbname=Config.RDB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
            )

    query = "SELECT * FROM track;"
    df = pd.read_sql(query, conn)
    total_last_fm_url_list = df['last_fm_url'].unique().tolist()
    print(f"the total number of url = {len(total_last_fm_url_list)}")
    context["ti"].xcom_push(key='total_last_fm_url_list', value=total_last_fm_url_list)

# ray 사용
def get_track_artist_list(**context):
    artist_list = []
    track_list = []

    country_daily_chart_url = Config.REGIONS

    print("Crawling Start!")
    for url in tqdm(country_daily_chart_url[:2]):
        try:
            soup = get_soup(url)
            rows = soup.select('div.subcontainer>table')[0].find_all('td', class_='text mp')
            for row in rows:
                artist_track = row.select("a")

                artist = artist_track[0].text
                artist_list.append(artist)

                track = artist_track[1].text
                track_list.append(track)
        except:
            print(url)

    basic_url = 'https://www.last.fm/music/'
    url_list = []
    for track_name, artist_name in zip(track_list, artist_list):
        url = basic_url + artist_name.replace(' ', '+') + '/_/' + track_name.replace(' ', '+')
        url_list.append(url)

    total_last_fm_url_list = context['ti'].xcom_pull(key="total_last_fm_url_list")
    last_fm_url_list = list(set(url_list) - set(total_last_fm_url_list))

    print(f"the number of url to be searched from last fm : {len(last_fm_url_list)}")
    context["ti"].xcom_push(key='last_fm_url_list', value=last_fm_url_list)


def get_listeners(url):
    try:
        soup = get_soup(url)
        listeners = soup.select('div.header-new-info-desktop>ul.header-metadata-tnew>li.header-metadata-tnew-item>div.header-metadata-tnew-display')[0].text.strip()
        if listeners[-1] == 'K':
            listeners = float(listeners[:-1]) * 1000
        elif listeners[-1] == 'M':
            listeners = float(listeners[:-1]) * 1000000
        else:
            listeners = float(listeners.replace(',',''))
    except:
        listeners = '0'
    return int(listeners)


def get_length(url):
    try:
        soup = get_soup(url)
        length = soup.select('div.container.page-content>div.row')[0].find('div', class_='col-main buffer-standard buffer-reset@sm').select('div.metadata-column>dl.catalogue-metadata>dd.catalogue-metadata-description')[0].text.strip()
        length = int(length[0]) * 60 + int(length[2:])
    except:
        length = ''
    return length


def get_genres(url):
    try:
        soup = get_soup(url)
        group = soup.select('div.container.page-content>div.row')[0].find('div', class_='row buffer-3 buffer-4@sm').select('div.col-sm-8>div.section-with-separator.section-with-separator--xs-only>section.catalogue-tags>ul.tags-list.tags-list--global')[0].find_all('li', class_='tag')
        genres = [genre.text for genre in group]
    except:
        genres = ''
    return genres


def get_img_url(url):
    try:
        soup = get_soup(url)
        img_url = soup.select('div.source-album-art>span.cover-art>img')[0]['src']
    except:
        img_url = 'https://lastfm.freetls.fastly.net/i/u/300x300/c6f59c1e5e7240a4c0d427abd71f3dbb.jpg'
    return img_url


def get_introduction(url):
    try:
        soup = get_soup(url)
        introduction = soup.find_all('div',class_='wiki-content')[0].text.strip()
    except:
        introduction = ''
    return introduction

def get_info(num_partition: int, idx:int, **context):
    last_fm_url_list = context['ti'].xcom_pull(key="last_fm_url_list")[:500]
    print(f"Today's the number of new tracks : {len(last_fm_url_list)}")

    partition = math.ceil(len(last_fm_url_list) / num_partition)
    start, end = idx * partition, (idx + 1) * partition
    last_fm_url_list = last_fm_url_list[start:end]
    print(f"start : {start}")
    print(f"end : {end}")


    print("Start crawling")
    results = []
    batch_size = math.ceil(len(last_fm_url_list) / 20)
    for i in tqdm(range(0, len(last_fm_url_list), batch_size)):
        batch_urls = last_fm_url_list[i:i + batch_size]
        batch_results = {
            'last_fm_url': batch_urls,
            'listeners': [get_listeners(url) for url in batch_urls],
            'length': [get_length(url) for url in batch_urls],
            'genres': [get_genres(url) for url in batch_urls],
            'img_url': [get_img_url(url) for url in batch_urls],
            'introduction': [get_introduction(url + '/+wiki') for url in batch_urls]
        }

        batch_df = pd.DataFrame(batch_results)
        results.append(batch_df)

    print("Crawling completed!")

    final_df = pd.concat(results, ignore_index=True)
    context['task_instance'].xcom_push(key=f'partition_result_{idx}', value=final_df.to_dict())

def combine_results(**context):
    """모든 파티션의 결과를 합치는 함수"""
    all_results = []
    
    # 각 파티션의 결과를 수집
    for i in range(NUM_PARTITION):
        partition_result = context['task_instance'].xcom_pull(
            task_ids=f'get_info_group_task.get_info_{i+1}',
            key=f'partition_result_{i}'
        )
        if partition_result:
            df = pd.DataFrame.from_dict(partition_result)
            all_results.append(df)
    
    # 모든 결과를 하나의 DataFrame으로 합치기
    final_df = pd.concat(all_results, ignore_index=True)
    
    # artist와 track 열 생성
    final_df['artist'] = final_df['last_fm_url'].apply(lambda x: x.split('/')[-3].replace('+', ' '))
    final_df['track'] = final_df['last_fm_url'].apply(lambda x: x.split('/')[-1].replace('+', ' '))

    # 최종 결과 저장
    date = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(Directory.DOWNLOADS_DIR, f"spotify/{date}.csv")
    final_df.to_csv(path, index = False)

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

    date = datetime.now().strftime("%Y-%m-%d")
    source_path = os.path.join(Directory.DOWNLOADS_DIR, f"{date}.csv")
    destination_path = f'download/spotify/{date}.csv'    
    hook.upload(
        bucket_name=bucket_name,
        object_name=destination_path,
        filename=source_path
    )

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

NUM_PARTITION = 8

with DAG('daily_spotify_dag',
        default_args=default_args,
        schedule='0 0 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_last_fm_url_from_db_task = PythonOperator(
        task_id='get_last_fm_url_from_db_task',
        python_callable=get_last_fm_url_from_db,
        provide_context=True
    )

    get_track_artist_list_task = PythonOperator(
        task_id='get_track_artist_list_task',
        python_callable=get_track_artist_list,
        provide_context=True
    )

    with TaskGroup("get_info_group_task") as get_info_group_task:
        for i in range(NUM_PARTITION):
            get_info_task = PythonOperator(
                task_id=f"get_info_{i+1}",
                python_callable=get_info,
                op_kwargs={
                    "num_partition": NUM_PARTITION,
                    "idx": i
                }
            )

    combine_results_task = PythonOperator(
        task_id='combine_results_task',
        python_callable=combine_results
    )

    upload_file_to_gcs_task = PythonOperator(
        task_id='upload_file_to_gcs_task',
        python_callable=upload_file_to_gcs,
        op_kwargs= {
            "bucket_name": Config.BUCKET_NAME
        },
        provide_context=True
    )

    delete_xcom_task = PythonOperator(
            task_id="delete_xcom_task",
            python_callable=delete_xcoms_for_dags,
            op_kwargs={'dag_ids': ['get_last_fm_url_from_db_task', 'get_track_artist_list_task', 'get_info_group_task']}
        )
    
    end_task = EmptyOperator(
        task_id="end_task"
    )
    
    start_task >> get_last_fm_url_from_db_task >> get_track_artist_list_task >> get_info_group_task >> combine_results_task >>  upload_file_to_gcs_task >> delete_xcom_task >> end_task
