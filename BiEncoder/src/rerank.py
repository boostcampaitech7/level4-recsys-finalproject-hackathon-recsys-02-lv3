import psycopg2
from pgvector.psycopg2 import register_vector
import pandas as pd
import logging
from typing import List, Dict
from omegaconf import OmegaConf
import torch
from torch.nn.functional import normalize
from preprocess import handle_missing_values, scale_numeric_features, dataframe_to_dict
from train import load_model


def generate_embeddings(song_encoder, data_songs: List[Dict]) -> torch.Tensor:
    '''
    주어진 트랙 데이터를 기반으로 임베딩을 생성하는 함수

    Args:
        song_encoder (torch.nn.Module): 학습된 트랙 임베딩 모델
        data_songs (List[Dict]): 트랙 메타데이터 리스트

    Returns:
        torch.Tensor: 트랙 임베딩 텐서
    '''
    embeddings_list = []
    device = next(song_encoder.parameters()).device

    for idx, song_info in enumerate(data_songs):
        if 'artist' not in song_info:
            logging.warning(f"[generate_embeddings] Missing 'artist' key in index {idx}, skipping.")
            continue
        
        try:
            artists = [song_info["artist"]]
            tracks = [song_info["track"]]
            playlists = [song_info["playlist"]]
            listeners = [song_info["listeners"]]
            lengths = [song_info["length"]]
            genres = [song_info["genres"]]

            with torch.no_grad():
                emb = song_encoder(artists, tracks, playlists, listeners, lengths, genres)
            embeddings_list.append(emb)
        except KeyError as e:
            logging.error(f"[generate_embeddings] KeyError: {e} at index {idx}, skipping.")
            continue
        except Exception as e:
            logging.error(f"[generate_embeddings] Unexpected error at index {idx}: {e}, skipping.")
            continue
    
    if embeddings_list:
        return torch.cat(embeddings_list, dim=0).to(device)
    else:
        return torch.tensor([]).to(device)

def fetch_track_embeddings_from_db(track_ids: List[int], config) -> Dict[int, torch.Tensor]:
    '''
    데이터베이스에서 트랙 임베딩을 가져오는 함수

    Args:
        track_ids (List[int]): 가져올 트랙 ID 리스트
        config (OmegaConf): 데이터베이스 설정을 포함한 설정

    Returns:
        Dict[int, torch.Tensor]: 트랙 ID를 키로 하고 임베딩을 값으로 가지는 딕셔너리
    '''
    conn = None
    candidate_embeddings = {}
    try:
        conn = psycopg2.connect(
            dbname=config.database_emb.dbname,
            user=config.database_emb.user,
            password=config.database_emb.password,
            host=config.database_emb.host,
            port=config.database_emb.port
        )

        register_vector(conn)
        cur = conn.cursor()

        sql_query = """
        SELECT track_id, track_emb
        FROM track_meta_embedding
        WHERE track_id = ANY(%s);
        """
        cur.execute(sql_query, (track_ids,))
        results = cur.fetchall()

        for row in results:
            can_track_id, can_track_emb = row
            candidate_embeddings[can_track_id] = torch.from_numpy(can_track_emb)

    except Exception as e:
        logging.error(f"[fetch_track_embeddings_from_db] Error while fetching data: {e}")
    finally:
        if conn:
            conn.close()

    return candidate_embeddings

def recommend_songs(response, candidates_track_ids, config_path, model_path):
    '''
    사용자 플레이리스트와 후보 트랙 간의 유사도를 기반으로 추천 곡을 반환하는 함수

    Args:
        response (Dict): 사용자 플레이리스트 데이터
        candidates_track_ids (List[int]): 후보 트랙 ID 리스트
        config_path (str): 설정 파일 경로
        model_path (str): 모델 체크포인트 경로

    Returns:
        List[int]: 추천된 트랙 ID 리스트
    '''

    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained song encoder and query encoder
    song_encoder, query_encoder, scaler, data_songs = load_model(config_path, model_path)
    song_encoder.to(device)
    query_encoder.to(device)
    song_encoder.eval()
    query_encoder.eval()

    # Convert genre list to a single string if necessary
    for i in range(len(response['missing'])):
        if isinstance(response['missing'][i]['genres'], list):
            response['missing'][i]['genres'] = ", ".join(response['missing'][i]['genres'])

    # Convert response data to DataFrame
    data = pd.DataFrame(response['missing'])
    data.columns = ['track', 'artist', 'playlist', 'genres', 'length', 'listeners']
    data = handle_missing_values(data)
    data = scale_numeric_features(data, scaler=scaler)
    data_songs = dataframe_to_dict(data)

    # Generate embeddings for missing tracks
    playlist_embeddings_m = generate_embeddings(song_encoder, data_songs)

    # Fetch existing track embeddings from database
    exists_track_ids = response.get('exists', [])
    playlist_embeddings_e_dict = fetch_track_embeddings_from_db(exists_track_ids, config)

    # Collect valid existing track embeddings
    playlist_e_tensors = []
    for track_id in exists_track_ids:
        if track_id in playlist_embeddings_e_dict:
            playlist_e_tensors.append(playlist_embeddings_e_dict[track_id])

    # Convert existing embeddings to tensor
    if len(playlist_e_tensors) > 0:
        playlist_e_tensor = torch.stack(playlist_e_tensors).to(device)  # (N_e, dim)
    else:
        playlist_e_tensor = torch.tensor([]).view(0, playlist_embeddings_m.shape[1]).to(device)

    # Merge missing and existing track embeddings
    if playlist_e_tensor.shape[0] > 0 and playlist_embeddings_m.shape[0] > 0:
        playlist_tensor = torch.cat((playlist_e_tensor, playlist_embeddings_m), dim=0)
    elif playlist_e_tensor.shape[0] > 0:
        playlist_tensor = playlist_e_tensor
    else:
        playlist_tensor = playlist_embeddings_m 

    # Fetch candidate track embeddings from database
    candidate_embeddings_dict = fetch_track_embeddings_from_db(candidates_track_ids, config)

    candidate_track_ids_loaded = list(candidate_embeddings_dict.keys())
    candidate_embedding_list = [candidate_embeddings_dict[tid] for tid in candidate_track_ids_loaded]
    if len(candidate_embedding_list) == 0:
        logging.warning("[recommend_songs] No candidate embeddings found.")
        return []

    candidate_tensor = torch.stack(candidate_embedding_list).to(device)  # (C, dim)

    # Normalize embeddings for similarity calculation
    playlist_normalized = normalize(playlist_tensor, p=2, dim=1)  # (P, d)
    candidate_normalized = normalize(candidate_tensor, p=2, dim=1) # (C, d)

    # Compute cosine similarity
    similarity_matrix = torch.matmul(candidate_normalized, playlist_normalized.T)
    mean_similarities = similarity_matrix.mean(dim=1) 

    # Rank candidates by similarity
    sorted_indices = torch.argsort(mean_similarities, descending=True)
    sorted_candidate_ids = [candidate_track_ids_loaded[idx] for idx in sorted_indices.tolist()]

    return sorted_candidate_ids


if __name__ == "__main__":
    config_path = "../config.yaml"
    model_path = "song_query_model.pt"
    
    response_example = {
        "user_id": "user_123",
        "exists": [1951, 1511, 9383],  # 이미 가진 곡(트랙 ID)
        "missing": [
            {
                "track_name": "I'm Not the Only One",
                "artists_name": "Sam Smith",
                "playlist_name": "sad moood",
                "genres": ["soul", "pop", "synthpop", "melancholy", "2014"],
                "length": 205.0,
                "listeners": 1100000.0
            },
            {
                "track_name": "Adele",
                "artists_name": "Hello",
                "playlist_name": "silent",
                "genres": ["soul", "pop", "2015", "british"],
                "length": 352.0,
                "listeners": 884600.0
            }
        ]
    }

    candidates_track_ids_example = list(range(1, 100000))  # 가상의 후보 트랙ID들

    recommended_ids = recommend_songs(
        response_example,
        candidates_track_ids_example,
        config_path,
        model_path,
    )
    print("[TEST] Recommended Track IDs:", recommended_ids[:10])