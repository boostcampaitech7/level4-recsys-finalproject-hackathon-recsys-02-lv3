from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
import boto3
from selenium import webdriver

import ray
import math
import os
import time
from datetime import datetime
import pandas as pd
from typing import List, Tuple
import logging
import os
#import pendulum
import glob
import requests
import json
from urllib.parse import quote
from datetime import datetime
from tqdm import tqdm
import pickle
import random

from utils import Directory

def get_soup(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    return soup

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

def get_list():
    # 불러오기
    path = os.path.join(Directory.ROOT_DIR, "crawler/my_list.pkl")
    with open(path, 'rb') as f:
        loaded_list = pickle.load(f)
    return loaded_list

def get_info():
    num_cpus = 8
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    
    unique_url_list = get_list()
    batch_size = 10  # CPU 코어 수를 고려한 배치 크기
    results = []
    
    print("Start crawling")
    for i in tqdm(range(0, len(unique_url_list), batch_size)):
        batch_urls = unique_url_list[i:i + batch_size]
        
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
    
    # 모든 배치 결과 합치기
    final_df = pd.concat(results, ignore_index=True)

    benchmark_data_path = os.path.join(Directory.DOWNLODAD_DIR, 'benchmark')
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(benchmark_data_path, f'data_{current_time}.csv')

    final_df.to_csv(filename, index=False)

def merge_with_benchmark_data():
    benchmark_data_path = os.path.join(Directory.DOWNLODAD_DIR, 'benchmark')

    spotify_data_path = os.path.join(benchmark_data_path, "spotify_dataset.csv")
    data = pd.read_csv(spotify_data_path, on_bad_lines = "skip")

    meta_data_path = os.path.join(benchmark_data_path, '550000_.csv')
    meta_data = pd.read_csv(meta_data_path)
    meta_data.rename(columns = {'url' : "last_fm_url"}, inplace = True)
    meta_data['genres'].fillna('[]', inplace=True)
    meta_data['introduction'].fillna('', inplace=True)

    df = data[data['last_fm_url'].isin(meta_data['last_fm_url'].unique().tolist())]
    df = pd.merge(df, meta_data, on='last_fm_url', how = 'left')

    total_data_path = os.path.join(benchmark_data_path, 'total_data.csv')
    df.to_csv(total_data_path, index=False)
