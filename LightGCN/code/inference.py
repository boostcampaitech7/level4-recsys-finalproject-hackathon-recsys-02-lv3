import torch
import os
from omegaconf import OmegaConf
from batch_dataloader import Loader
from model import LightGCN
import utils

def inference(rating, uid, top_k=100) -> list:
    '''
    user_embs, item_embs: 학습된 유저와 아이템 임베딩 (emb_dim:64)
    uid: 유저 아이디 (int)
    '''
    _, indices = torch.topk(ratings[uid], top_k)

    return indices

if __name__=='__main__':
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(ROOT_PATH, 'config.yaml')
    config = OmegaConf.load(config_path)
    print(f"config: {OmegaConf.to_yaml(config)}")

    weight_file = utils.getFileName(ROOT_PATH, config)
    dataset = Loader(config=config, path=os.path.join(ROOT_PATH,config.path.DATA))
    model = LightGCN(config, dataset)
    checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("Load the model successfully!")
    
    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight
    print(user_embs.shape, item_embs.shape)
    ratings = torch.matmul(user_embs, item_embs.T)
    
    # for uid in range(len(user_embs)):
    #     print(inference(ratings, uid))