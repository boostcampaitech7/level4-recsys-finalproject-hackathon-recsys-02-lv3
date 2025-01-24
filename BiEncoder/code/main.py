from preprocess import preprocess_data
from models import SongEncoder, GenreEncoder
from train import train_model, load_model
from eval import evaluate_model
from inference import infer_similarity
import torch

def main():
    # preprocess
    data_songs, artist_list = preprocess_data(config_path="../config.yaml")
    
    # model initialization
    song_encoder = SongEncoder(
        artist_list=artist_list,
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32
    )
    genre_encoder = GenreEncoder(
        pretrained_name="distilbert-base-uncased",
        embed_dim=32
    )

    # train
    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        num_epochs=3, 
        margin=0.2, 
        save_path="song_genre_model.pt"
    )

    # evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs)

    # inference
    test_song = data_songs[0]
    test_genres = ["instrumental", "piano", "covers"]
    infer_similarity(song_encoder, genre_encoder, test_song, test_genres)

if __name__ == "__main__":
    main()
