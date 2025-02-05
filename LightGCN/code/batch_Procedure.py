import argparse
from omegaconf import OmegaConf
from pymongo import MongoClient
import pandas as pd
import os

import torch

from inference import load_data_model, get_item_node, get_all_user_node


def get_inter_data(id, pw)-> list:
    '''
    Parameters: id, pw
        MongoDB 접근을 위한 id, pw입니다.
    Return: list
        gcn 모델에 존재하는 track들에 대해(모든 트랙에 해당하지 않음) 전일에 누적된 유저, pos item 상호작용 배치 데이터를 불러옵니다.
    '''
    uri = f"mongodb+srv://{id}:{pw}@cluster0.octqq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    
    interaction = client['spotify']['interaction']

    pipeline = [{
        "$group": {
            "_id": "$user_id",
            "records": {"$push": "$$ROOT"}
        }
    }]

    grouped_results = list(interaction.aggregate(pipeline))
    gcn_id = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sources/gcn_track_id.csv'))
    batch = []

    for group in grouped_results:
        uid, batch_pos = group['_id'], []
        for r in group['records']:
            try:
                track_id = int(gcn_id.loc[gcn_id['track_id']==r['track_id'], 'gcn_track_id'])
            except: continue
            if r['action'] == 'positive': batch_pos.append(track_id)
        batch.append({"uid":uid, "batch_pos": batch_pos})
    
    return batch

@load_data_model
def generate_temporary_user(dataset, model, batch_data, alpha=0.2) -> list:
    '''
    배치 데이터에 존재하는 유저들의 임시 임베딩을 생성합니다.
    Parameters:
        alpha: 근접한 유저 임베딩과 pos item 임베딩 조합의 영향을 조절하는 가중치
    Return:
        batch_data: 각 유저 아이디와 유저 임시 임베딩
    '''
    e_users = get_all_user_node(dataset, model)
    for batch_user in batch_data:
        user_emb = torch.mean(get_item_node(model, torch.tensor(batch_user['batch_pos'], dtype=torch.long)), dim=0)
        sim = torch.matmul(e_users, user_emb) / \
            (torch.norm(e_users, dim=1) * torch.norm(user_emb))
        sim_user = torch.topk(sim, k=1, largest=True)[1]
        batch_user['u_emb'] = alpha * e_users.detach()[sim_user].squeeze() + (1-alpha) * user_emb
        batch_user.pop('batch_pos', None)
    return batch_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--pw', type=str, required=True)
    args = parser.parse_args()

    batch_data = get_inter_data(args.id, args.pw)
    config = OmegaConf.load('config.yaml')
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

    ### 전일에 적재된 유저들에 대한 임시 임베딩 생성 ###
    print(generate_temporary_user(config, ROOT_PATH, batch_data))