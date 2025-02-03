from omegaconf import OmegaConf
from preprocess import preprocess_data, load_playlist
from models import SongEncoder, QueryEncoder
from train import train_model
from eval import evaluate_model


def main():
    # Configuration
    print(1)
    config_path = "./config.yaml"
    config = OmegaConf.load(config_path)

    # Preprocess
    print(2)
    data_songs, scaler, cluster_embeds, clusters_dict = preprocess_data(config_path, scaler=None)
    playlist_info = load_playlist(config_path)

    # Model initialization
    print(3)
    song_encoder = SongEncoder(
        config,
        playlist_info=playlist_info,
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    query_encoder = QueryEncoder(config)

    # Train
    print(4)
    train_model(
        song_encoder, 
        query_encoder, 
        data_songs, 
        scaler=scaler, 
        config=config
    )
    print(5)
    # Evaluation
    evaluate_model(song_encoder, query_encoder, data_songs, config)

if __name__ == "__main__":
    main()