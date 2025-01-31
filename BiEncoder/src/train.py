import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SongEncoder, GenreEncoder
from preprocess import preprocess_data, load_playlist
from utils import cosine_triplet_margin_loss

from typing import List, Dict
import ast 

class SongDataset(Dataset):
    def __init__(self, data_songs):
        self.data = data_songs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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

    artists, tracks, playlists, genres = [], [], [], []
    listeners_list, lengths_list = [], []
    
    for item in batch:
        artists.append(item["artist"])      
        tracks.append(item["track"])
        playlists.append(item["playlist"])
        genres.append(item["genres"])
        
        listeners_list.append(item["listeners"])
        lengths_list.append(item["length"])

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
                scaler, 
                num_epochs=10, batch_size=32, margin=0.2, 
                save_path="song_genre_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    pin_memory=True, 
                    collate_fn = custom_collate_fn)

    optimizer = optim.Adam(
        list(song_encoder.parameters()) + list(genre_encoder.parameters()), 
        lr=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(num_epochs):
        song_encoder.train()
        genre_encoder.train()
        total_loss = 0.0

        for batch in dataloader:
            # Prepare batch data
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

            # Compute genre embeddings
            pos_embs = genre_encoder(genres)

            # Simple negative sampling (circular shift)
            neg_embs = genre_encoder(genres[1:] + [genres[0]])

            # Compute loss
            loss = cosine_triplet_margin_loss(anchor_embs, pos_embs, neg_embs, margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    save_model(song_encoder, genre_encoder, scaler, save_path)

def save_model(song_encoder, genre_encoder, scaler, save_path="song_genre_model.pt"):
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

    song_encoder = SongEncoder(
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32,
        playlist_info = playlist_info, 
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    genre_encoder = GenreEncoder()

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
