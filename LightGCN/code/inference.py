import torch
import os
from omegaconf import OmegaConf
from batch_dataloader import Loader
from model import LightGCN

def inference(rating, uid, top_k=100) -> list:
    '''
    user_embs, item_embs: 학습된 유저와 아이템 임베딩 (emb_dim:64)
    uid: 유저 아이디 (int)
    '''
    _, indices = torch.topk(ratings[uid], top_k)

    return indices

if __name__=='__main__':
    config = OmegaConf.load('config.yaml')
    print(f"config: {OmegaConf.to_yaml(config)}")

    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'spotify']
    all_models  = ['mf', 'lgn']

    dataset = Loader(config=config, path=os.path.join(ROOT_PATH,config.path.DATA))
    model = LightGCN(config, dataset)
    checkpoint = torch.load(os.path.join(ROOT_PATH, config.path.FILE, 'best_model.pth'), map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    
    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight
    print(user_embs.shape, item_embs.shape)
    ratings = torch.matmul(user_embs, item_embs.T)
    
    for uid in range(len(user_embs)):
        print(inference(ratings, uid))