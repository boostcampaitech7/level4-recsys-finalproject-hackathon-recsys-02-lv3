import os
import torch
from LightGCN.code.batch_dataloader import Loader
from LightGCN.code.model import LightGCN
import numpy as np
from omegaconf import OmegaConf

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LightGCN", "config.yaml")
CONFIG = OmegaConf.load(config_path)


def get_model():
    """
    모델 로드 및 아이템 임베딩 반환
    """
    dataset = Loader(config=CONFIG, path="./data")
    model = LightGCN(CONFIG, dataset)
    checkpoint = torch.load('./LightGCN/checkpoints/best_model.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)

    # Item embeddings만 반환 (사용자 임베딩은 동적으로 계산)
    item_embs = model.embedding_item.weight
    return model, item_embs

def compute_ratings(user_embedding: np.ndarray, item_embs: torch.Tensor) -> torch.Tensor:
    """
    사용자 임베딩과 아이템 임베딩으로 점수 계산
    :param user_embedding: 사용자 임베딩 (numpy 배열)
    :param item_embs: 아이템 임베딩 (torch.Tensor)
    :return: 사용자-아이템 점수 (torch.Tensor)
    """
    user_embedding_tensor = torch.tensor(user_embedding, dtype=torch.float32).to(item_embs.device)
    ratings = torch.matmul(user_embedding_tensor, item_embs.T)
    return ratings

# 여러 user에 대한 ratings 중 uid에 맞는 topk 추출
# def inference(ratings, uid, top_k=10) -> list:
#     _, indices = torch.topk(ratings[uid], top_k)
#     return indices

def inference(ratings, top_k=100) -> list:
    """
    단일 사용자의 ratings에서 상위 top_k 아이템 인덱스를 반환
    :param ratings: 단일 사용자의 아이템 점수 
    :param top_k: 상위 k개의 아이템
    :return: top_k 아이템의 인덱스
    """
    _, indices = torch.topk(ratings, top_k)
    print(f"Top {top_k} indices: {indices}")
    return indices.tolist()
