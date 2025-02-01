import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from BiEncoder.src.utils import cosine_triplet_margin_loss
#from utils import cosine_triplet_margin_loss
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

    torch.cuda.empty_cache()
    dataset = SongDataset(data_songs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = optim.Adam(
        list(song_encoder.parameters()) + list(genre_encoder.parameters()), 
        lr=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in tqdm(range(num_epochs)):
        song_encoder.train()
        genre_encoder.train()
        total_loss = 0.0
        batch_count = 0
        print(len(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            # if batch_idx >= 4:
            #     break
            
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
            print(5)
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save model
    torch.save({
        "song_encoder_state": song_encoder.state_dict(),
        "genre_encoder_state": genre_encoder.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")

def load_model(song_encoder, genre_encoder, load_path="song_genre_model.pt"):
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    song_encoder.load_state_dict(checkpoint["song_encoder_state"])
    genre_encoder.load_state_dict(checkpoint["genre_encoder_state"])
    print(f"Model loaded from {load_path}")