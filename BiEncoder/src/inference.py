import torch
from torch.utils.data import DataLoader
import logging
import psycopg2
from omegaconf import OmegaConf
from tqdm import tqdm
from train import load_model, custom_collate_fn
from eval import SongDataset


def generate_and_save_embeddings(
    song_encoder: torch.nn.Module,
    data_songs: list,
    config: OmegaConf,
    batch_commit_size: int = 100
):
    '''
    트랙 메타 데이터를 기반으로 트랙 임베딩을 생성하고 데이터베이스에 저장하는 함수
    존재하는 track_id는 업데이트하고, 없는 track_id는 새로 생성
    
    Args:
        song_encoder (torch.nn.Module): 학습된 트랙 임베딩 모델
        data_songs (list): 트랙 메타데이터 리스트
        config (OmegaConf): 데이터베이스 및 학습 환경 설정
        batch_commit_size (int): 데이터베이스 커밋 단위
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    song_encoder.to(device)
    song_encoder.eval()

    # Check GPU usage
    print(f"Using device: {device}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Create dataset and dataloader
    dataset = SongDataset(data_songs)
    dataloader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Connect to PostgreSQL Database
    conn = psycopg2.connect(
        dbname=config.database_emb.dbname,
        user=config.database_emb.user,
        password=config.database_emb.password,
        host=config.database_emb.host,
        port=config.database_emb.port
    )
    cur = conn.cursor()

    # SQL Query
    upsert_query = """
        INSERT INTO track_meta_embedding (track_id, track_emb)
        VALUES (%s, %s)
        ON CONFLICT (track_id)
        DO UPDATE SET track_emb = EXCLUDED.track_emb;
    """

    pbar = tqdm(total=len(dataloader), desc="Generating embeddings")
    batch_counter = 0
    
    try:
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
                    
                    # UPSERT 
                    cur.execute(upsert_query, (track_id, emb_array.tolist()))
                    batch_counter += 1
                    
                    # Commit by batch_commit_size
                    if batch_counter >= batch_commit_size:
                        conn.commit()
                        logging.info(f"Committed {batch_commit_size} upserts to database")
                        batch_counter = 0
                
                pbar.update(1)
        
        # Final Commit 
        if batch_counter > 0:
            conn.commit()
            logging.info(f"Committed final {batch_counter} upserts to database")
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        conn.rollback()
        raise e
    
    finally:
        pbar.close()
        cur.close()
        conn.close()
        logging.info("Embeddings saved to database successfully")

def main():
    '''
    설정 파일을 불러와 모델을 로드한 후, 트랙 임베딩을 생성하여 저장하는 메인 함수
    '''
    # Basic Setting
    config_path = "../config.yaml"
    model_path = "song_query_model.pt"
    config = OmegaConf.load(config_path)

    # Load model and dataset
    song_encoder, query_encoder, scaler, data_songs = load_model(config_path, model_path)

    # Generate and save embeddings with progress tracking
    generate_and_save_embeddings(song_encoder, data_songs, config)

if __name__ == "__main__":
    main()