import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

def evaluate_model(song_encoder, genre_encoder, data_songs: List[Dict], batch_size=32) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.eval()
    genre_encoder.eval()

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

            # Compute genre embeddings
            pos_embs = genre_encoder(genres)

            # Negative sampling (circular shift)
            neg_embs = genre_encoder(genres[1:] + [genres[0]])

            # Cosine similarity
            pos_sim = F.cosine_similarity(anchor_embs, pos_embs)
            neg_sim = F.cosine_similarity(anchor_embs, neg_embs)

            # Count correct predictions
            correct += torch.sum(pos_sim > neg_sim).item()
            total += len(pos_sim)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy*100:.2f}%")
    return accuracy