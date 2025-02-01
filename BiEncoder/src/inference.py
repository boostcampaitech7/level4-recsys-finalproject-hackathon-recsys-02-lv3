import torch
import psycopg2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess_data, load_config, connect_db
from models import SongEncoder, GenreEncoder
from train import load_model
from eval import SongDataset
import logging

def generate_and_save_embeddings(song_encoder, data_songs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    song_encoder.eval()

    dataset = SongDataset(data_songs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Database connection
    conn = psycopg2.connect(
        dbname=config['database_emb']['dbname'],
        user=config['database_emb']['user'],
        password=config['database_emb']['password'],
        host=config['database_emb']['host'],
        port=config['database_emb']['port']
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

            # Generate embeddings
            embeddings = song_encoder(artists, tracks, playlists, listeners, lengths, genres)
            
            # Save to database
            for i, emb in enumerate(embeddings):
                track_id = batch_idx * dataloader.batch_size + i + 1
                emb_array = emb.cpu().numpy()
                print(track_id)
                print(emb_array)
    #             # Insert embedding into PostgreSQL
    #             cur.execute(
    #                 "UPDATE track_meta_embedding SET track_emb = %s WHERE track_id = %s", 
    #                 (emb_array.tolist(), track_id)
    #             )

    # conn.commit()
    # cur.close()
    # conn.close()
    # logging.info("Embeddings saved to database successfully")


def main():
    config_path = "../config.yaml"
    config_path = '/Users/mac/Documents/level4-recsys-finalproject-hackathon-recsys-02-lv3/BiEncoder/config.yaml'
    model_path = "song_genre_model.pt"
    model_path = "/Users/mac/Documents/level4-recsys-finalproject-hackathon-recsys-02-lv3/BiEncoder/checkpoints/25-01-28/25-01-28.pt"

    # Load configuration
    config = load_config(config_path)

    # Preprocess data
    data_songs, artist_list = preprocess_data(config_path)

    # Initialize models
    song_encoder = SongEncoder(
        artist_list=artist_list,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder()

    # Load trained models
    load_model(song_encoder, genre_encoder, model_path)

    # Generate and save embeddings
    generate_and_save_embeddings(song_encoder, data_songs, config)

if __name__ == "__main__":
    main()
