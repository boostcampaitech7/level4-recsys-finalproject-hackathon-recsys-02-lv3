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
import ray
from datetime import datetime
import pandas as pd
import psycopg2
from utils import Config, Directory

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
    context["ti"].xcom_push(key='total_last_fm_url_list', value=total_last_fm_url_list)


def get_track_artist_list(**context):
    artist_list = []
    track_list = []

    country_daily_chart_url = Config.REGIONS

    for url in tqdm(country_daily_chart_url):
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

    context["ti"].xcom_push(key='last_fm_url_list', value=last_fm_url_list)

@ray.remote
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

@ray.remote
def get_length(url):
    try:
        soup = get_soup(url)
        length = soup.select('div.container.page-content>div.row')[0].find('div', class_='col-main buffer-standard buffer-reset@sm').select('div.metadata-column>dl.catalogue-metadata>dd.catalogue-metadata-description')[0].text.strip()
        length = int(length[0]) * 60 + int(length[2:])
    except:
        length = ''
    return length

@ray.remote
def get_genres(url):
    try:
        soup = get_soup(url)
        group = soup.select('div.container.page-content>div.row')[0].find('div', class_='row buffer-3 buffer-4@sm').select('div.col-sm-8>div.section-with-separator.section-with-separator--xs-only>section.catalogue-tags>ul.tags-list.tags-list--global')[0].find_all('li', class_='tag')
        genres = [genre.text for genre in group]
    except:
        genres = ''
    return genres

@ray.remote
def get_img_url(url):
    try:
        soup = get_soup(url)
        img_url = soup.select('div.source-album-art>span.cover-art>img')[0]['src']
    except:
        img_url = 'https://lastfm.freetls.fastly.net/i/u/300x300/c6f59c1e5e7240a4c0d427abd71f3dbb.jpg'
    return img_url

@ray.remote
def get_introduction(url):
    try:
        soup = get_soup(url)
        introduction = soup.find_all('div',class_='wiki-content')[0].text.strip()
    except:
        introduction = ''
    return introduction

def get_info(**context):
    num_cpus = 8
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    
    last_fm_url_list = context['ti'].xcom_pull(key="last_fm_url_list")
    print(f"Today's the number of new tracks : {len(last_fm_url_list)}")

    batch_size = 10  # CPU 코어 수를 고려한 배치 크기
    results = []
    
    print("Start crawling")
    for i in tqdm(range(len(last_fm_url_list), batch_size)):
        batch_urls = last_fm_url_list[i:i + batch_size]
        
        # 배치 단위로 작업 제출
        batch_refs = {
            'url': batch_urls,
            'listeners': [get_listeners.remote(url) for url in batch_urls],
            'length': [get_length.remote(url) for url in batch_urls],
            'genres': [get_genres.remote(url) for url in batch_urls],
            'img_url': [get_img_url.remote(url) for url in batch_urls],
            'introduction': [get_introduction.remote(url + '/+wiki') for url in batch_urls]
        }
        
        # 배치 결과 수집
        batch_results = {
            'url': batch_refs['url'],
            'listeners': ray.get(batch_refs['listeners']),
            'length': ray.get(batch_refs['length']),
            'genres': ray.get(batch_refs['genres']),
            'img_url': ray.get(batch_refs['img_url']),
            'introduction': ray.get(batch_refs['introduction'])
        }
        
        # 배치 결과를 DataFrame으로 변환하여 저장
        batch_df = pd.DataFrame(batch_results)
        results.append(batch_df)

    print("Crawling completed!")

    final_df = pd.concat(results, ignore_index=True)

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    path = os.path.join(Directory.DOWNLOADS_DIR, f"spotify/{date}.csv")
    final_df.to_csv(path)


def upload_file_to_gcs(filename: str, object_name: str, bucket_name: str, replace: bool = True) -> None:
    """ Airflow Hook으로 GCS에 파일을 업로드하는 메서드
    Args:
        filename (str): 업로드할 로컬 파일 경로
        object_name (str): GCS에 저장될 객체 이름/경로
        bucket_name (str): GCS 버킷 이름
        replace (bool): 덮어쓰기 여부 (기본값: True)
    """
    hook = GCSHook(gcp_conn_id="google_cloud_default")
    
    if not replace and hook.exists(bucket_name=bucket_name, object_name=object_name):
        print(f"Object {object_name} already exists in bucket {bucket_name}")
        return
        
    hook.upload(
        bucket_name=bucket_name,
        object_name=object_name,
        filename=filename,
        gzip=False  # 필요시 True로 설정하여 gzip 압축 가능
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


with DAG('user_embedding_update_dag',
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

    get_info_task = PythonOperator(
        task_id='get_info_task',
        python_callable=get_info,
        provide_context=True
    )

    upload_file_to_gcs_task = PythonOperator(
        task_id='upload_file_to_gcs_task',
        python_callable=upload_file_to_gcs,
        provide_context=True
    )