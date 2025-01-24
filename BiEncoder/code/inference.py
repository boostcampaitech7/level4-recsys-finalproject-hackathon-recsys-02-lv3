import torch
import torch.nn.functional as F
from typing import List, Dict

def infer_similarity(song_encoder, genre_encoder, song_info: Dict, test_genres: List[str]) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    genre_encoder.to(device)
    song_encoder.eval()
    genre_encoder.eval()

    with torch.no_grad():
        anchor_emb = song_encoder(
            song_info["artist"],
            song_info["track"],
            song_info["playlist"],
            song_info["listeners"],
            song_info["length"],
            song_info["genres"]
        ).unsqueeze(0).to(device)  # (1, final_dim)

        test_genre_emb = genre_encoder(test_genres).unsqueeze(0).to(device)  # (1, embed_dim)

        cos_sim = F.cosine_similarity(anchor_emb, test_genre_emb, dim=1)
        print(f"Song '{song_info['track']}' vs {test_genres}, CosSim = {cos_sim.item():.4f}")
        return cos_sim.item()
