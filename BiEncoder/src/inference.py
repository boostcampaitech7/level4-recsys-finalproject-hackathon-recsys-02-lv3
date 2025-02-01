import torch
import psycopg2
from torch.utils.data import DataLoader
from preprocess import load_config
from train import load_model, custom_collate_fn
from eval import SongDataset
import logging
from omegaconf import OmegaConf

def generate_and_save_embeddings(song_encoder, data_songs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    song_encoder.eval()

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size,
        shuffle=False, 
        collate_fn = custom_collate_fn
    )

    conn = psycopg2.connect(
        dbname=config.database_emb.dbname,
        user=config.database_emb.user,
        password=config.database_emb.password,
        host=config.database_emb.host,
        port=config.database_emb.port
    )
    cur = conn.cursor()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            artists = batch["artist"]
            tracks = batch["track"]
            
            playlists = batch["playlist"]
            listeners = batch["listeners"].to(device)
            lengths = batch["length"].to(device)
            genres = batch["genres"]

            embeddings = song_encoder(artists, tracks, playlists, listeners, lengths, genres)
            
            for i, emb in enumerate(embeddings):
                track_id = batch_idx * dataloader.batch_size + i + 1
                emb_array = emb.cpu().numpy()
                
                cur.execute(
                    "UPDATE track_meta_embedding SET track_emb = %s WHERE track_id = %s", 
                    (emb_array.tolist(), track_id)
                )

    conn.commit()
    cur.close()
    conn.close()
    logging.info("Embeddings saved to database successfully")

def main():
    config_path = "../config.yaml"
    model_path = "song_genre_model.pt"

    config = OmegaConf.load(config_path)

    song_encoder, genre_encoder, scaler, data_songs = load_model(config_path, model_path)

    generate_and_save_embeddings(song_encoder, data_songs, config)

if __name__ == "__main__":
    main()
