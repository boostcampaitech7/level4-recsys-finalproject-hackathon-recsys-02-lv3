import psycopg2
from pgvector.psycopg2 import register_vector
import pandas as pd
import logging
from typing import List, Dict
from omegaconf import OmegaConf
import torch
from torch.nn.functional import normalize
from preprocess import handle_missing_values, scale_numeric_features, dataframe_to_dict, connect_db
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


def fetch_data_listeners(conn, candidates_track_ids) -> pd.DataFrame:
    '''
    후보곡들의 아이디를 이용해 해당 곡들의 청취자수를 DB에서 불러오는 함수

    Args:
        candidates_track_ids (List[int]): 후보 트랙 ID 리스트
    
    Returns:
        DataFrame: track_id과 listeners로 구성된 데이터프레임
    '''

    sql = """
    SELECT 
        t.track_id,
        t.listeners
    FROM track t
    WHERE t.track_id = ANY(%s)
    GROUP BY t.track_id, t.track, t.listeners
    """
    data = pd.read_sql(sql , conn, params=(candidates_track_ids,))
    conn.close()
    return data


def sort_by_score_popularity(data, similarity_scores):
    '''
    후보곡의 (사용자 플레이리스트와의)유사도, 청취자수를 이용해 해당 곡들 추천 순서를 재정렬하는 함수

    Args:
        data (pd.DataFrame): 청취자 수를 0~1 범위로 정규화한 'popularity' 칼럼이 포함된 DataFrame
        similarity_scores (array-like): 후보곡의 유사도 점수 (임베딩 기반 유사도)
    
    Returns:
        pd.DataFrame: 최종 정렬된 DataFrame
        상위 90개는 (0.9 * similarity + 0.1 * popularity)에 따라 내림차순 정렬되고,
        하위 10개(가장 낮은 popularity)는 별도로 추출하여 맨 아래에 배치

    '''

    # 최종 score = 0.9 * similarity + 0.1 * normalized popularity
    data['similarity'] = similarity_scores
    data['score'] = 0.9 * similarity_scores + 0.1 * data['popularity']

    # popularity 하위 10개, 상위 90개 
    bottom_10 = data.nsmallest(10, 'popularity')
    bottom_10_indices = bottom_10.index
    top_candidates = data.drop(bottom_10_indices)

    # 상위 90개는 score 내림차순으로 정렬, 하위 10개는 popularity 내림차순 정렬
    top_candidates_sorted = top_candidates.sort_values(by='score', ascending=False)
    bottom_candidates_sorted = data.loc[bottom_10_indices].sort_values(by='popularity', ascending=False)

    # 최종 정렬: 상위 90개 + 하위 10개(맨 아래 배치)
    data_sorted = pd.concat([top_candidates_sorted, bottom_candidates_sorted])

    return data_sorted


def recommend_songs(song_encoder, query_encoder, response, candidates_track_ids, config_path, model_path):
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
    playlist_e_tensors = []
    for track_id in exists_track_ids:
        if track_id in playlist_embeddings_e_dict:
            playlist_e_tensors.append(playlist_embeddings_e_dict[track_id])

    # Merge missing and existing track embeddings
    if len(playlist_e_tensors) > 0:
        playlist_e_tensor = torch.stack(playlist_e_tensors).to(device)  # (N_e, dim)
    else:
        playlist_e_tensor = torch.tensor([]).view(0, playlist_embeddings_m.shape[1]).to(device)
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

    # Compute Cosine Similarity
    playlist_normalized = normalize(playlist_tensor, p=2, dim=1)  # (P, d)
    candidate_normalized = normalize(candidate_tensor, p=2, dim=1) # (C, d)
    similarity_matrix = torch.matmul(candidate_normalized, playlist_normalized.T)
    mean_similarities = similarity_matrix.mean(dim=1) 
    similarity_scores = mean_similarities.cpu().numpy()  # shape: (C,)

    # Compute Popularity 
    conn = connect_db(config)
    data = fetch_data_listeners(conn, candidates_track_ids) 
    min_listeners = data['listeners'].min()
    max_listeners = data['listeners'].max() 
    if max_listeners > min_listeners: # listeners 칼럼을 0 ~ 1 사이로 스케일링 (min-max scaling)
        data['popularity'] = (data['listeners'] - min_listeners) / (max_listeners - min_listeners)
    else:
        data['popularity'] = 1.0

    # Sort by Score and popularity
    data_sorted = sort_by_score_popularity(data, similarity_scores)
    sorted_candidate_ids = data_sorted['track_id'].tolist()

    return sorted_candidate_ids    


if __name__ == "__main__":
    config_path = "../config.yaml"
    model_path = "song_query_model.pt"
    song_encoder, query_encoder, scaler, data_songs = load_model(config_path, model_path) # 모델 로드 
    
    response_example = {
        "user_id": "user_123",
        "exists": [647, 5918, 6760],  # 이미 가진 곡(트랙 ID)
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

    import random
    candidates_track_ids_example = random.sample(range(1, 2241631), 100) # 가상의 후보 트랙ID들

    recommended_ids = recommend_songs(
            song_encoder, 
            query_encoder,
            response_example,
            candidates_track_ids_example,
            config_path,
            model_path,
        )
    print("[TEST] Recommended Track IDs:", recommended_ids[:10])