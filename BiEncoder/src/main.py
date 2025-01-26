import torch
from preprocess import preprocess_data
from models import SongEncoder, GenreEncoder
from train import train_model, load_model
from eval import evaluate_model
from inference import infer_similarity

def main():
    # Configuration
    config_path = "../config.yaml"
    batch_size = 32
    num_epochs = 3
    margin = 0.2
    save_path = "song_genre_model.pt"

    # Preprocess
    data_songs, artist_list = preprocess_data(config_path)
    
    # Model initialization
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

    # Train with batch processing
    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        margin=margin, 
        save_path=save_path
    )

    # Evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs)

    # # Inference
    # print("inference start")
    # test_song = data_songs[0]
    # test_genres = ["instrumental", "piano", "covers"]
    # infer_similarity(song_encoder, genre_encoder, test_song, test_genres)

if __name__ == "__main__":
    main()