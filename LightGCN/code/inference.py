import os
from omegaconf import OmegaConf
import torch
from dataloader import Loader
from model import LightGCN

from functools import wraps

def load_data_model(func):
    @wraps(func) 
    def wrapper(config, root_path, *args, **kwargs):
        dataset = Loader(config=config, path=os.path.join(root_path, config.path.DATA))
        model = LightGCN(config, dataset)
        checkpoint = torch.load(os.path.join(root_path, config.path.FILE, 'best_model.pth'), map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint)
        return func(dataset, model, *args, **kwargs)

    return wrapper


@load_data_model
def inference(dataset, model, uid, top_k=100) -> list:
    '''
    Parameters:
        uid: top-k개의 추천 리스트를 fetch할 user id
    Return:
        top-k개의 추천 리스트
    '''
    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight
    print(f"graph shape is users: {user_embs.shape}, items: {item_embs.shape}")
    ratings = torch.matmul(user_embs, item_embs.T)

    _, indices = torch.topk(ratings[uid], top_k)

    return indices


def get_item_node(model, iid) -> torch.embedding:
    item_nodes = model.embedding_item.weight
    return item_nodes.detach()[iid]


def get_all_user_node(dataset, model) -> list:
    user_nodes = model.embedding_user.weight
    return user_nodes


if __name__=='__main__':
    config = OmegaConf.load('config.yaml')
    print(f"config: {OmegaConf.to_yaml(config)}")
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    
    for uid in range(config.num_users):
        print(inference(uid))