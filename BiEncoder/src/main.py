from preprocess import preprocess_data, load_playlist
from models import SongEncoder, GenreEncoder
from train import train_model
from eval import evaluate_model
from omegaconf import OmegaConf


def main():
    # Configuration
    config_path = "../config.yaml"
    config = OmegaConf.load(config_path)

    # Preprocess
    data_songs, scaler, cluster_embeds, clusters_dict = preprocess_data(config_path, scaler=None)
    playlist_info = load_playlist(config_path)

    # Model initialization
    song_encoder = SongEncoder(
        config,
        playlist_info=playlist_info,
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    genre_encoder = GenreEncoder(config)

    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        scaler=scaler, 
        config=config
    )

    # Evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs, config)

if __name__ == "__main__":
    main()