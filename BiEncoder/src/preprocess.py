import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
import ast
from typing import List, Dict

def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def connect_db(config: Dict):
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
    sql = """
    SELECT 
        t.track_id,
        t.track,
        t.listeners,
        t.length,
        array_agg(DISTINCT a.artist) AS artist,  
        string_agg(DISTINCT g.genre, ', ')   AS genres,
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
    data = data.head(100000) # 빠른 실험을 위해 간소화(수정요망)
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # fill nan(numeric)
    data['length'].fillna(data['length'].mean(), inplace=True)
    data['listeners'].fillna(data['listeners'].mean(), inplace=True)
    
    # fill nan(str)
    data['artist'].fillna("<UNK>", inplace=True)
    data['track'].fillna("<UNK>", inplace=True)
    data['playlist'].fillna("<UNK>", inplace=True)
    data['genres'].fillna("", inplace=True)  
    
    return data

def scale_numeric_features(data: pd.DataFrame, scale_cols: List[str] = ['listeners', 'length']) -> pd.DataFrame:
    scaler = MinMaxScaler()
    data[scale_cols] = scaler.fit_transform(data[scale_cols])
    return data

def dataframe_to_dict(data: pd.DataFrame) -> List[Dict]:
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

def extract_unique_artists(data_songs: List[Dict]) -> List[str]:
    all_artists = set()
    for song in data_songs:
        if song["artist"] is None:
            continue
        
        if isinstance(song["artist"], str) and song["artist"].startswith("[") and song["artist"].endswith("]"):
            artists = ast.literal_eval(song["artist"]) 
            all_artists.update(artists)
        elif isinstance(song["artist"], list):
            all_artists.update(song["artist"])
        elif isinstance(song["artist"], str):
            all_artists.add(song["artist"])
        else:
            continue  # 잘못된 형식 무시
        artist_list = ["<UNK>"] + sorted(filter(lambda x: x is not None, all_artists)) # 수정함
    return artist_list


def preprocess_data(config_path):
    config = load_config(config_path)
    conn = connect_db(config)
    data = fetch_data(conn)
    data = handle_missing_values(data)
    data = scale_numeric_features(data)
    data_songs = dataframe_to_dict(data)
    artist_list = extract_unique_artists(data_songs)
    return data_songs, artist_list