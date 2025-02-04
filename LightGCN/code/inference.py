import torch
import os
from omegaconf import OmegaConf
from batch_dataloader import Loader
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
    user_embs, item_embs: 학습된 유저와 아이템 임베딩 (emb_dim:64)
    uid: 유저 아이디 (int)
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
