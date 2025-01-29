import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import cosine_triplet_margin_loss
from typing import List, Dict

class SongDataset(Dataset):
    def __init__(self, data_songs):
        self.data = data_songs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "artist": self.data[idx]["artist"],
            "track": self.data[idx]["track"],
            "playlist": self.data[idx]["playlist"],
            "listeners": self.data[idx]["listeners"],
            "length": self.data[idx]["length"],
            "genres": self.data[idx]["genres"]
        }

def train_model(song_encoder, genre_encoder, data_songs: List[Dict], 
                num_epochs=10, batch_size=32, margin=0.2, 
                save_path="song_genre_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

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

    save_model(song_encoder, genre_encoder, save_path)
    print(f"Model saved to {save_path}")

def save_model(song_encoder, genre_encoder, save_path="song_genre_model.pt"):
    checkpoint = {
        "song_encoder_state": song_encoder.state_dict(),
        "genre_encoder_state": genre_encoder.state_dict(),
        "artist_vocab": {
            "artist2id": song_encoder.artist_encoder.artist2id,
            "id2artist": song_encoder.artist_encoder.id2artist
        }
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(song_encoder, genre_encoder, load_path="song_genre_model.pt"):
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    
    # control artist encoder embedding size
    old_embedding_size = checkpoint["song_encoder_state"]["artist_encoder.embedding.weight"].shape[0]
    if old_embedding_size != song_encoder.artist_encoder.embedding.num_embeddings:
        new_embedding = nn.EmbeddingBag(old_embedding_size, 
                                      song_encoder.artist_encoder.embed_dim, 
                                      mode='mean')
        with torch.no_grad():
            new_embedding.weight.data = checkpoint["song_encoder_state"]["artist_encoder.embedding.weight"].data
        
        song_encoder.artist_encoder.embedding = new_embedding
    
    song_encoder.load_state_dict(checkpoint["song_encoder_state"])
    genre_encoder.load_state_dict(checkpoint["genre_encoder_state"])
    
    if "artist_vocab" in checkpoint:
        song_encoder.artist_encoder.artist2id = checkpoint["artist_vocab"]["artist2id"]
        song_encoder.artist_encoder.id2artist = checkpoint["artist_vocab"]["id2artist"]
    
    print(f"Model loaded from {load_path}")
    return song_encoder, genre_encoder