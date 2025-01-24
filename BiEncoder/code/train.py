import torch
import torch.optim as optim
from utils import cosine_triplet_margin_loss
from typing import List, Dict

def train_model(song_encoder, genre_encoder, data_songs: List[Dict], num_epochs=3, margin=0.2, save_path="song_genre_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.train()
    genre_encoder.train()

    optimizer = optim.Adam(list(song_encoder.parameters()) + list(genre_encoder.parameters()), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, song_info in enumerate(data_songs):
            anchor_emb = song_encoder(
                song_info["artist"],
                song_info["track"],
                song_info["playlist"],
                song_info["listeners"],
                song_info["length"],
                song_info["genres"]
            ).unsqueeze(0).to(device)  # (1, final_dim)

            pos_emb = genre_encoder(song_info["genres"]).unsqueeze(0).to(device)

            neg_idx = (i + 1) % len(data_songs)
            neg_emb = genre_encoder(data_songs[neg_idx]["genres"]).unsqueeze(0).to(device)

            loss = cosine_triplet_margin_loss(anchor_emb, pos_emb, neg_emb, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_songs)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # save model
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
