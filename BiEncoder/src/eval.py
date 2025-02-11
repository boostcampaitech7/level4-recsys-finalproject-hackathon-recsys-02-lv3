import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from omegaconf import OmegaConf
from train import custom_collate_fn, SongDataset


def evaluate_model(song_encoder, query_encoder, data_songs: List[Dict], config) -> float:
    '''
    SongEncoder와 QueryEncoder 성능을 평가
    SongEncoder의 임베딩과 해당 QueryEncoder의 임베딩 간의 코사인 유사도를 계산

    Args:
        song_encoder (torch.nn.Module): 트랙 임베딩 생성 모델
        query_encoder (torch.nn.Module): 쿼리 임베딩 생성 모델
        data_songs (List[Dict]): 평가에 사용할 곡 메타데이터 리스트
        config (OmegaConf): 학습 및 평가 환경 설정

    Returns:
        float: 모델의 정확도 (올바른 임베딩 매칭 비율)

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    query_encoder.to(device)
    song_encoder.eval()
    query_encoder.eval()

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size,
        shuffle=False, 
        collate_fn = custom_collate_fn
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            artists = batch["artist"]
            tracks = batch["track"]
            playlists = batch["playlist"]
            listeners = batch["listeners"]
            lengths = batch["length"]
            genres = batch["genres"]

            # Compute song embeddings
            anchor_embs = song_encoder(
                artists, tracks, playlists, 
                listeners, lengths, genres
            )

            # Compute embeddings
            pos_embs = query_encoder(genres)

            # Negative sampling
            neg_embs = query_encoder(genres[1:] + [genres[0]])

            # Cosine similarity
            pos_sim = F.cosine_similarity(anchor_embs, pos_embs)
            neg_sim = F.cosine_similarity(anchor_embs, neg_embs)

            # Count correct predictions
            correct += torch.sum(pos_sim > neg_sim).item()
            total += len(pos_sim)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy*100:.2f}%")
    return accuracy