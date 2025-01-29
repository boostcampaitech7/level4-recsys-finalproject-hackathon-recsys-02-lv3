import torch
import psycopg2
import numpy as np
import pandas as pd
import logging

from typing import List, Dict
from torch.nn.functional import normalize

from preprocess import (
    load_config,
    handle_missing_values,
    scale_numeric_features,
    dataframe_to_dict)
from models import SongEncoder, GenreEncoder
from train import load_model
from pgvector.psycopg2 import register_vector


def generate_embeddings(song_encoder, data_songs: List[Dict]) -> torch.Tensor:
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


def fetch_track_embeddings_from_db(
    track_ids: List[int],
    config: Dict
) -> Dict[int, torch.Tensor]:
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
    response,
    candidates_track_ids,
    config_path,
    model_path):

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    song_encoder = SongEncoder(
        initial_artist_vocab_size=1000,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder()

    song_encoder, genre_encoder = load_model(song_encoder, genre_encoder, model_path)
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.eval()
    genre_encoder.eval()

    for i in range(len(response['missing'])):
        if isinstance(response['missing'][i]['genres'], list):
            response['missing'][i]['genres'] = ", ".join(response['missing'][i]['genres'])

    data = pd.DataFrame(response['missing'])
    data.columns = ['track', 'artist', 'playlist', 'genres', 'length', 'listeners']
    data = handle_missing_values(data)
    data = scale_numeric_features(data)
    data_songs = dataframe_to_dict(data)

    playlist_embeddings_m = generate_embeddings(song_encoder, data_songs)

    exists_track_ids = response.get('exists', [])
    playlist_embeddings_e_dict = fetch_track_embeddings_from_db(exists_track_ids, config)

    playlist_e_tensors = []
    for track_id in exists_track_ids:
        if track_id in playlist_embeddings_e_dict:
            playlist_e_tensors.append(playlist_embeddings_e_dict[track_id])

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

    candidate_embeddings_dict = fetch_track_embeddings_from_db(candidates_track_ids, config)

    candidate_track_ids_loaded = list(candidate_embeddings_dict.keys())
    candidate_embedding_list = [candidate_embeddings_dict[tid] for tid in candidate_track_ids_loaded]
    if len(candidate_embedding_list) == 0:
        logging.warning("[recommend_songs] No candidate embeddings found.")
        return []

    candidate_tensor = torch.stack(candidate_embedding_list).to(device)  # (C, dim)

    playlist_normalized = normalize(playlist_tensor, p=2, dim=1)  # (P, d)
    candidate_normalized = normalize(candidate_tensor, p=2, dim=1) # (C, d)

    similarity_matrix = torch.matmul(candidate_normalized, playlist_normalized.T)
    mean_similarities = similarity_matrix.mean(dim=1) 

    sorted_indices = torch.argsort(mean_similarities, descending=True)
    sorted_candidate_ids = [candidate_track_ids_loaded[idx] for idx in sorted_indices.tolist()]

    return sorted_candidate_ids


if __name__ == "__main__":
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

    recommended_ids = recommend_songs(
        response_example,
        candidates_track_ids_example,
        config_path,
        model_path,
    )
    print("[TEST] Recommended Track IDs:", recommended_ids[:10])
