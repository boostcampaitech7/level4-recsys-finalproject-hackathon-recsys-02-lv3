import ast 
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import SongEncoder, GenreEncoder
from preprocess import preprocess_data, load_playlist
from utils import cosine_triplet_margin_loss, EarlyStopping


class SongDataset(Dataset):
    '''
    트랙 메타 데이터를 처리하는 PyTorch Dataset 클래스
    '''
    def __init__(self, data_songs):
        self.data = data_songs

    def __len__(self):
        # Number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves a single sample from the dataset.
        item = self.data[idx]
        artist_value = item["artist"]
        
        if isinstance(artist_value, str):
            if artist_value.startswith("[") and artist_value.endswith("]"):
                artist_value = ast.literal_eval(artist_value)  # -> list
            else:
                artist_value = [artist_value]

        return {
            "artist": artist_value,
            "track": item["track"],
            "playlist": item["playlist"],
            "listeners": item["listeners"],
            "length": item["length"],
            "genres": item["genres"]
        }


def custom_collate_fn(batch):
    '''
    배치 데이터를 PyTorch 텐서 및 리스트로 변환

    Args:
        batch (List[Dict]): 배치 데이터 리스트

    Returns:
        Dict: PyTorch 텐서 및 리스트로 변환된 배치 데이터
    '''
    artists, tracks, playlists, genres = [], [], [], []
    listeners_list, lengths_list = [], []
    
    for item in batch:
        artists.append(item["artist"])      
        tracks.append(item["track"])
        playlists.append(item["playlist"])
        genres.append(item["genres"])
        
        listeners_list.append(item["listeners"])
        lengths_list.append(item["length"])

    # Convert numerical values to PyTorch tensors
    listeners_tensor = torch.tensor(listeners_list, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths_list, dtype=torch.float32)

    return {
        "artist": artists,       
        "track": tracks,         
        "playlist": playlists,  
        "listeners": listeners_tensor, 
        "length": lengths_tensor,      
        "genres": genres   
    }


def train_model(song_encoder, genre_encoder, data_songs: List[Dict], 
                scaler, config):
    '''
    트랙 및 장르 인코더 모델을 학습

    Args:
        song_encoder (torch.nn.Module): 학습할 트랙 인코더 모델
        genre_encoder (torch.nn.Module): 학습할 장르 인코더 모델
        data_songs (List[Dict]): 트랙 메타데이터 리스트
        scaler (MinMaxScaler): 데이터 정규화를 위한 스케일러
        config (OmegaConf): 학습 환경 설정

    Returns:
        None
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        collate_fn = custom_collate_fn)

    optimizer = optim.Adam(
        list(song_encoder.parameters()) + list(genre_encoder.parameters()), 
        lr=config.training.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.training.scheduler_patience)
    early_stopping = EarlyStopping(config)

    best_loss = float("inf")  # Initialize best loss
    ###
    num_batches = len(dataloader)

    # 에포크 진행바
    epoch_pbar = tqdm(range(config.training.num_epochs), desc="Training", position=0)
    
    # 러닝 메트릭스 저장
    metrics_history = {
        'loss': [],
        'lr': []
    }
    ###

    for epoch in epoch_pbar: ###
        song_encoder.train()
        genre_encoder.train()
        total_loss = 0.0

        ###
        # 배치 진행바
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", 
                         leave=False, position=1, 
                         total=num_batches)
        
        # 배치별 손실값 저장
        batch_losses = [] # Store batch loss values
        ###

        for batch in batch_pbar:###
            artists = batch["artist"]
            tracks = batch["track"]
            playlists = batch["playlist"]
            listeners = batch["listeners"]
            lengths = batch["length"]
            genres = batch["genres"]

            # Compute embeddings
            anchor_embs = song_encoder(
                artists, tracks, playlists, 
                listeners, lengths, genres
            )

            pos_embs = genre_encoder(genres)

            neg_embs = genre_encoder(genres[1:] + [genres[0]])

            loss = cosine_triplet_margin_loss(anchor_embs, pos_embs, neg_embs, config)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ###
            # total_loss += loss.item()
            current_loss = loss.item()
            total_loss += current_loss
            batch_losses.append(current_loss)
            
            # 배치 진행바 업데이트
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{np.mean(batch_losses):.4f}'
            })
            ###

        avg_loss = total_loss / num_batches

        ###
        current_lr = optimizer.param_groups[0]['lr']
        
        # 메트릭스 저장
        metrics_history['loss'].append(avg_loss)
        metrics_history['lr'].append(current_lr)
        ###
                
        scheduler.step(avg_loss)
        ###
        # 에포크 진행바 업데이트
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'best_loss': f'{best_loss:.4f}'
        })
        ###
        # print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
        best_loss = save_best_model(
            song_encoder, genre_encoder, scaler, 
            config.training.save_path, best_loss, avg_loss
        )

        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    save_model(song_encoder, genre_encoder, scaler, config.training.save_path)
    ###
    # 학습 완료 메시지
    print("\n✅ Training completed!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Learning Rate: {current_lr:.6f}")
    ###


def save_model(song_encoder, genre_encoder, scaler, save_path="song_genre_model.pt"):
    '''
    학습된 모델을 저장

    Args:
        song_encoder (torch.nn.Module): 학습된 트랙 인코더
        genre_encoder (torch.nn.Module): 학습된 장르 인코더
        scaler (MinMaxScaler): 데이터 정규화를 위한 스케일러
        save_path (str, optional): 저장할 모델 경로

    Returns:
        None
    '''

    checkpoint = {
        "song_encoder_state": song_encoder.state_dict(),
        "genre_encoder_state": genre_encoder.state_dict(),
        "artist_vocab": {
            "artist2id": song_encoder.artist_encoder.artist2id,
            "id2artist": song_encoder.artist_encoder.id2artist
        },
        "scaler": scaler
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(config_path, model_path="song_genre_model.pt"):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    data_songs, cluster_embeds, clusters_dict = preprocess_data(config_path, checkpoint.get("scaler"))
    playlist_info = load_playlist(config_path)

    config = OmegaConf.load(config_path)
    
    # Model initialization
    song_encoder = SongEncoder(
        config,
        playlist_info=playlist_info,
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    genre_encoder = GenreEncoder(config)

    old_embedding_size = checkpoint["song_encoder_state"]["artist_encoder.embedding.weight"].shape[0]
    if old_embedding_size != song_encoder.artist_encoder.embedding.num_embeddings:
        new_embedding = nn.Embedding(old_embedding_size, 
                                        song_encoder.artist_encoder.output_dim)
        with torch.no_grad():
            new_embedding.weight.data = checkpoint["song_encoder_state"]["artist_encoder.embedding.weight"].data

        song_encoder.artist_encoder.embedding = new_embedding

    song_encoder.load_state_dict(checkpoint["song_encoder_state"])
    genre_encoder.load_state_dict(checkpoint["genre_encoder_state"])

    if "artist_vocab" in checkpoint:
        song_encoder.artist_encoder.artist2id = checkpoint["artist_vocab"]["artist2id"]
        song_encoder.artist_encoder.id2artist = checkpoint["artist_vocab"]["id2artist"]

    scaler = checkpoint.get("scaler", None)
    print(f"Model loaded from {model_path}")
    return song_encoder, genre_encoder, scaler, data_songs


def save_best_model(song_encoder, genre_encoder, scaler, save_path, best_loss, current_loss):
    """ 
    현재 Loss가 더 낮을 때 최상의 모델을 저장

    Args:
        song_encoder (torch.nn.Module): 학습된 트랙 인코더 모델
        genre_encoder (torch.nn.Module): 학습된 장르 인코더 모델
        scaler (MinMaxScaler): 데이터 정규화를 위한 스케일러
        save_path (str): 저장할 모델 경로
        best_loss (float): 현재까지의 최소 손실 값
        current_loss (float): 현재 에포크의 손실 값

    Returns:
        float: 갱신된 최상의 손실 값.
    '''
    """
    if current_loss < best_loss:
        checkpoint = {
            "song_encoder_state": song_encoder.state_dict(),
            "genre_encoder_state": genre_encoder.state_dict(),
            "artist_vocab": {
                "artist2id": song_encoder.artist_encoder.artist2id,
                "id2artist": song_encoder.artist_encoder.id2artist
            },
            "scaler": scaler
        }
        torch.save(checkpoint, save_path)
        print(f"Best model saved to {save_path}")
        return current_loss 
    return best_loss