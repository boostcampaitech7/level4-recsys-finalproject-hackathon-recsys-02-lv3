import torch
from preprocess import load_config, preprocess_data
from models import SongEncoder, GenreEncoder
from train import train_model, load_model
from eval import evaluate_model

def main():
    # Configuration
    config_path = "../config.yaml"
    config = load_config(config_path)
    config_training = config['training']
    config_model = config['model']

    # Preprocess
    data_songs, artist_list, scaler, cluster_embeds, clusters_dict = preprocess_data(config_path, scaler=None)

    # Model initialization
    song_encoder = SongEncoder(
        bert_pretrained=config_model['bert_pretrained'],
        mha_embed_dim=config_model['mha_embed_dim'],
        mha_heads=config_model['mha_heads'],
        final_dim=config_model['final_dim'], 
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    genre_encoder = GenreEncoder(
        pretrained_name=config_model['bert_pretrained'],
        embed_dim=config_model['genre_embed_dim']
    )

    # Train with batch processing
    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        scaler=scaler, 
        num_epochs=config_training['num_epochs'],
        batch_size=config_training['batch_size'],
        margin=config_model['margin'],
        save_path=config_training['save_path']
    )

    # Evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs)

if __name__ == "__main__":
    main()