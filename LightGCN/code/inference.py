import torch
import argparse

from dataloader import Loader
from model import LightGCN

def inference(rating, uid, top_k=10000) -> list:
    '''
    user_embs, item_embs: 기학습된 유저와 아이템 임베딩(emb_dim:64)
    uid: 유저 아이디 (int)
    '''
    # 이미 본 아이템 제외하는 로직 필요
    _, indices = torch.topk(ratings[uid], top_k)

    return indices

if __name__=='__main__':
    ### config ###
    config = {}
    all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'spotify']
    all_models  = ['mf', 'lgn']
    # config['batch_size'] = 4096
    config['bpr_batch_size'] = 2048
    config['latent_dim_rec'] = 64
    config['lightGCN_n_layers']= 3
    config['dropout'] = 0.001
    config['keep_prob']  = 0.6
    config['A_n_fold'] = 100
    config['test_u_batch_size'] = 1000
    config['multicore'] = 0
    config['lr'] = 0.001
    config['decay'] = 1e-4
    config['pretrain'] = 0
    config['A_split'] = False
    config['bigdata'] = False
    config['shuffle'] = 'shuffle'

    dataset = Loader(path="../data/spotify")
    model = LightGCN(config, dataset)
    checkpoint = torch.load('./checkpoints/lgn-spotify-3-64-shuffle.pth.tar', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)

    user_embs, item_embs = model.embedding_user.weight, model.embedding_item.weight
    ratings = torch.matmul(user_embs, item_embs.T)
    
    for uid in range(len(user_embs)):
        print(inference(ratings, uid))