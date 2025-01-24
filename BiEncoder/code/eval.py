import torch
import torch.nn.functional as F
from typing import List, Dict

def evaluate_model(song_encoder, genre_encoder, data_songs: List[Dict]) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.eval()
    genre_encoder.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, song_info in enumerate(data_songs):
            anchor_emb = song_encoder(
                song_info["artist"],
                song_info["track"],
                song_info["playlist"],
                song_info["listeners"],
                song_info["length"],
                song_info["genres"]
            ).unsqueeze(0).to(device)

            pos_emb = genre_encoder(song_info["genres"]).unsqueeze(0).to(device)

            neg_idx = (i + 1) % len(data_songs)
            neg_emb = genre_encoder(data_songs[neg_idx]["genres"]).unsqueeze(0).to(device)

            pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=1)
            neg_sim = F.cosine_similarity(anchor_emb, neg_emb, dim=1)

            total += 1
            if pos_sim.item() > neg_sim.item():
                correct += 1

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy*100:.2f}%")
    return accuracy
