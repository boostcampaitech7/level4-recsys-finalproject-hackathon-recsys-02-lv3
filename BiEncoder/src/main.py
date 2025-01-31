import torch
from preprocess import preprocess_data
from models import SongEncoder, GenreEncoder
from train import train_model, load_model
from eval import evaluate_model

def main():
    # Configuration
    config_path = "../config.yaml"
    batch_size = 32
    num_epochs = 10
    margin = 0.2
    save_path = "song_genre_model.pt"

    # Preprocess
    data_songs, artist_list, scaler, cluster_embeds, clusters_dict = preprocess_data(config_path, scaler=None)

    # Model initialization
    song_encoder = SongEncoder(
        bert_pretrained="distilbert-base-uncased",
        mha_embed_dim=64,
        mha_heads=4,
        final_dim=32, 
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
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
        scaler=scaler, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        margin=margin, 
        save_path=save_path
    )

    # Evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs)

if __name__ == "__main__":
    main()