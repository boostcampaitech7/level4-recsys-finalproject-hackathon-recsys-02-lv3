import pandas as pd
import psycopg2
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
import torch


def load_config(config_path: str) -> Dict:
    '''
    YAML 구성 파일을 로드
    '''
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def connect_db(config: Dict):
    '''
    데이터베이스 연결을 생성
    '''
    db_config = config['database']
    conn = psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"]
    )
    return conn

def fetch_data(conn) -> pd.DataFrame:
    '''
    데이터베이스에서 트랙 데이터를 조회하여 DataFrame으로 반환
    '''
    sql = """
    SELECT 
        t.track_id,
        t.track,
        t.listeners,
        t.length,
        array_agg(DISTINCT a.artist) AS artist,  
        string_agg(DISTINCT g.genre, ', ') AS genres,
        string_agg(DISTINCT p.playlist, ', ') AS playlist
    FROM track t
    LEFT JOIN track_artist ta    ON t.track_id = ta.track_id
    LEFT JOIN artist a           ON ta.artist_id = a.artist_id
    LEFT JOIN track_genre tg     ON t.track_id = tg.track_id
    LEFT JOIN genre g            ON tg.genre_id = g.genre_id
    LEFT JOIN track_playlist tp  ON t.track_id = tp.track_id
    LEFT JOIN playlist p         ON tp.playlist_id = p.playlist_id
    GROUP BY t.track_id, t.track, t.listeners, t.length
    """
    data = pd.read_sql(sql, conn)
    conn.close()
    data["genres"] = data["genres"].fillna("").apply(lambda x: x.split(", ") if x else [])
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    '''
    데이터셋에서 누락된 값을 처리
    '''    
    data['length'] = data['length'].fillna(data['length'].mean())
    data['listeners'] = data['listeners'].fillna(data['listeners'].mean())
    data['artist'] = data['artist'].fillna("<UNK>")
    data['track'] = data['track'].fillna("<UNK>")
    data['playlist'] = data['playlist'].fillna("<UNK>")
    data['genres'] = data['genres'].fillna("")
    return data

def scale_numeric_features(data: pd.DataFrame, scaler=None, 
                           scale_cols: List[str] = ['listeners', 'length']):
    '''
    수치형 데이터 정규화
    '''
    if scaler is None:
        scaler = MinMaxScaler()
        data[scale_cols] = scaler.fit_transform(data[scale_cols])
        return data, scaler
    else:
        data[scale_cols] = scaler.transform(data[scale_cols])
        return data

def dataframe_to_dict(data: pd.DataFrame) -> List[Dict]:
    '''
    데이터프레임을 딕셔너리 리스트 형식으로 변환
    '''
    data_songs = []
    for _, row in data.iterrows():
        genres_text = row["genres"]
        data_songs.append({
            "artist": str(row["artist"]),
            "track": str(row["track"]),
            "playlist": str(row["playlist"]),
            "listeners": float(row["listeners"]),
            "length": float(row["length"]),
            "genres": genres_text
        })
    return data_songs

@torch.no_grad()
def compute_cluster_embeddings(clusters_dict, encoder):
    '''
    장르 클러스터의 평균 임베딩 계산
    '''
    encoder.eval()
    cluster_embeddings = {}

    for cid, genre_list in clusters_dict.items():
        batch_texts_emb = encoder(genre_list)
        cluster_mean_emb = batch_texts_emb.mean(dim=0)
        cluster_embeddings[cid] = cluster_mean_emb

    return cluster_embeddings

def load_playlist(config_path):
    '''
    데이터베이스에서 플레이리스트 정보 로드
    '''
    config = load_config(config_path)
    conn = connect_db(config) 
    sql = """
    SELECT p.playlist, a.artist
    FROM playlist p
    JOIN track_playlist tp ON p.playlist_id = tp.playlist_id
    JOIN track_artist ta ON tp.track_id = ta.track_id
    JOIN artist a ON ta.artist_id = a.artist_id
    LIMIT 10000
    """
    df = pd.read_sql(sql, conn)
    playlist_info = df.groupby('playlist')['artist'].apply(list).to_dict()
    conn.close()

    return playlist_info

def preprocess_data(config_path, scaler=None):
    '''
    데이터 전처리 수행
    '''
    config = load_config(config_path)
    conn = connect_db(config)
    data = fetch_data(conn)
    data = handle_missing_values(data)
    clusters_dict = config['clusters']

    # train
    if scaler == None: 
        data, scaler = scale_numeric_features(data, scaler=scaler)
        data_songs = dataframe_to_dict(data)

        from models import DistilBertTextEncoder
        encoder = DistilBertTextEncoder(pretrained_name="distilbert-base-uncased", output_dim=64)
        cluster_embeds = compute_cluster_embeddings(clusters_dict, encoder)
        return data_songs, scaler, cluster_embeds, clusters_dict

    # eval 
    else: 
        data = scale_numeric_features(data, scaler=scaler)
        data_songs = dataframe_to_dict(data)

        from models import DistilBertTextEncoder
        encoder = DistilBertTextEncoder(pretrained_name="distilbert-base-uncased", output_dim=64)
        cluster_embeds = compute_cluster_embeddings(clusters_dict, encoder)
        return data_songs, cluster_embeds, clusters_dict