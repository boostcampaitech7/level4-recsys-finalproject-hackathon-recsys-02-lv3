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
        bert_pretrained=config.model.bert_pretrained,
        mha_embed_dim=config.model.mha_embed_dim,
        mha_heads=config.model.mha_heads,
        final_dim=config.model.final_dim, 
        playlist_info=playlist_info,
        cluster_embeds=cluster_embeds,
        clusters_dict=clusters_dict
    )
    genre_encoder = GenreEncoder(
        pretrained_name=config.model.bert_pretrained,
        embed_dim=config.model.genre_embed_dim
    )

    train_model(
        song_encoder, 
        genre_encoder, 
        data_songs, 
        scaler=scaler, 
        config=config
    )

    # Evaluation
    evaluate_model(song_encoder, genre_encoder, data_songs)

if __name__ == "__main__":
    main()