import torch
import psycopg2
import numpy as np
import pandas as pd
import logging

from typing import List, Dict
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader

# --- 필요한 모듈 임포트 ---
from preprocess import (
    load_config,
    handle_missing_values,
    scale_numeric_features,
    dataframe_to_dict,
    extract_unique_artists
)
from models import SongEncoder, GenreEncoder
from train import load_model
from eval import SongDataset  # 필요 시 사용
from pgvector.psycopg2 import register_vector


def generate_embeddings(song_encoder, data_songs: List[Dict]) -> torch.Tensor:
    """
    data_songs 리스트(각 곡의 dict 형태)를 받아,
    song_encoder로 곡 임베딩을 만든 뒤 하나의 텐서로 반환.
    """
    embeddings_list = []
    device = next(song_encoder.parameters()).device

    for idx, song_info in enumerate(data_songs):
        # 필수 키가 없는 경우 건너뜀
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

            # 모델은 배치 형태 입력을 기대하므로 리스트로 감싸 처리
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


def fetch_track_embeddings_from_db(
    track_ids: List[int],
    config: Dict
) -> Dict[int, torch.Tensor]:
    """
    주어진 track_id 목록을 이용해,
    DB의 track_meta_embedding 테이블에서 (track_id, track_emb)를 조회하고,
    {track_id: tensor_embedding} 형태의 딕셔너리로 반환.
    """
    conn = None
    candidate_embeddings = {}
    try:
        conn = psycopg2.connect(
            dbname=config['database_emb']['dbname'],
            user=config['database_emb']['user'],
            password=config['database_emb']['password'],
            host=config['database_emb']['host'],
            port=config['database_emb']['port']
        )

        register_vector(conn)
        cur = conn.cursor()

        query = """
        SELECT track_id, track_emb
        FROM track_meta_embedding
        WHERE track_id = ANY(%s);
        """
        cur.execute(query, (track_ids,))
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


def recommend_songs(
    response: Dict,
    candidates_track_ids: List[int],
    config_path: str,
    model_path: str,
    # top_k: int = 10
) -> List[int]:
    """
    1) response(사용자 플레이리스트)와
    2) candidates_track_ids(추천 후보 트랙ID)를 입력받아
    3) song_encoder, genre_encoder 모델 로드 후
    4) 사용자의 플레이리스트 임베딩(존재하는 곡 + missing 곡) & 후보 임베딩을 각각 구함
    5) 평균 유사도 기준으로 내림차순 정렬하여 상위 track_id들을 반환
    """
    # 1. 설정 로드
    config = load_config(config_path)

    # 2. 모델 준비 (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    song_encoder = SongEncoder(
        initial_artist_vocab_size=1000,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder()  # 필요 없는 경우 불러만 두고 사용 X

    # 모델 파라미터 불러오기 (동적 artist 임베딩 사이즈 포함)
    song_encoder, genre_encoder = load_model(song_encoder, genre_encoder, model_path)
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.eval()
    genre_encoder.eval()

    # 3. response['missing'] 전처리 & 임베딩
    #    - genres 컬럼을 string으로 합치고, 데이터프레임 변환 후 scale
    for i in range(len(response['missing'])):
        # 이미 리스트 형태면 ", "로 join
        if isinstance(response['missing'][i]['genres'], list):
            response['missing'][i]['genres'] = ", ".join(response['missing'][i]['genres'])

    data = pd.DataFrame(response['missing'])
    data.columns = ['track', 'artist', 'playlist', 'genres', 'length', 'listeners']
    data = handle_missing_values(data)
    data = scale_numeric_features(data)
    data_songs = dataframe_to_dict(data)

    # missing 곡에 대한 임베딩
    playlist_embeddings_m = generate_embeddings(song_encoder, data_songs)

    # 4. 존재하는 곡 임베딩 (response['exists']에 대해 DB에서 꺼내옴)
    exists_track_ids = response.get('exists', [])
    playlist_embeddings_e_dict = fetch_track_embeddings_from_db(exists_track_ids, config)

    # DB에서 가져온 임베딩들을 하나의 텐서로 합치기
    playlist_e_tensors = []
    for track_id in exists_track_ids:
        if track_id in playlist_embeddings_e_dict:
            playlist_e_tensors.append(playlist_embeddings_e_dict[track_id])

    if len(playlist_e_tensors) > 0:
        playlist_e_tensor = torch.stack(playlist_e_tensors).to(device)  # (N_e, dim)
    else:
        # 존재하는 곡이 없는 경우
        playlist_e_tensor = torch.tensor([]).view(0, playlist_embeddings_m.shape[1]).to(device)

    # 최종 사용자 플레이리스트 임베딩(존재 + missing) 합치기
    if playlist_e_tensor.shape[0] > 0 and playlist_embeddings_m.shape[0] > 0:
        playlist_tensor = torch.cat((playlist_e_tensor, playlist_embeddings_m), dim=0)
    elif playlist_e_tensor.shape[0] > 0:
        playlist_tensor = playlist_e_tensor
    else:
        playlist_tensor = playlist_embeddings_m  # 둘 다 없을 경우 빈 텐서일 수도 있음

    # 5. 후보 트랙 임베딩 조회
    candidate_embeddings_dict = fetch_track_embeddings_from_db(candidates_track_ids, config)

    # DB에서 가져온 후보 트랙 임베딩 리스트/키 추출
    candidate_track_ids_loaded = list(candidate_embeddings_dict.keys())
    candidate_embedding_list = [candidate_embeddings_dict[tid] for tid in candidate_track_ids_loaded]
    if len(candidate_embedding_list) == 0:
        logging.warning("[recommend_songs] No candidate embeddings found.")
        return []

    candidate_tensor = torch.stack(candidate_embedding_list).to(device)  # (C, dim)

    # 6. 임베딩 정규화
    playlist_normalized = normalize(playlist_tensor, p=2, dim=1)  # (P, d)
    candidate_normalized = normalize(candidate_tensor, p=2, dim=1) # (C, d)

    # 7. 코사인 유사도 계산 (행렬 곱)
    # similarity_matrix: (C, P)
    similarity_matrix = torch.matmul(candidate_normalized, playlist_normalized.T)
    # 플레이리스트 전체 임베딩에 대한 평균 유사도
    mean_similarities = similarity_matrix.mean(dim=1)  # (C,)

    # 8. 유사도 기반 정렬
    sorted_indices = torch.argsort(mean_similarities, descending=True)
    sorted_candidate_ids = [candidate_track_ids_loaded[idx] for idx in sorted_indices.tolist()]

    # # 9. 최종 top_k 추출 (top_k=None 경우 전체 반환 가능)
    # if top_k and top_k > 0:
    #     sorted_candidate_ids = sorted_candidate_ids[:top_k]

    return sorted_candidate_ids


# 모듈 테스트용 (직접 실행 시)
if __name__ == "__main__":
    # 임의 데이터/경로 (실제 환경에 맞게 수정)
    config_path = "../config.yaml"
    model_path = "song_genre_model.pt"
    
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
    # top_k = 10

    recommended_ids = recommend_songs(
        response_example,
        candidates_track_ids_example,
        config_path,
        model_path,
        # top_k=top_k
    )
    print("[TEST] Recommended Track IDs:", recommended_ids[:10])
