# 1. MongoDB 연결해 uid, pos, neg fetch
# 2. KNN Logic ~.~
# 3. fetch했던 uid / user_emb 반환
from omegaconf import OmegaConf
from pymongo import MongoClient
import pandas as pd
import os

import torch

from inference import load_data_model, get_item_node, get_all_user_node

def get_inter_data():
    username = "jaypark"
    password = 'IOWABB9HnR0ew7Bf'  # 특수문자가 있는 경우 인코딩
    uri = f"mongodb+srv://{username}:{password}@cluster0.octqq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    
    interaction = client['spotify']['interaction']

    pipeline = [{
        "$group": {
            "_id": "$user_id",
            "records": {"$push": "$$ROOT"}
        }
    }]

    grouped_results = list(interaction.aggregate(pipeline))
    gcn_id = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gcn_track_id.csv'))
    batch = []

    for group in grouped_results:
        uid, batch_pos, batch_neg = group['_id'], [], []
        for r in group['records']:
            try:
                track_id = int(gcn_id.loc[gcn_id['track_id']==r['track_id'], 'gcn_track_id'])
            except: continue
            if r['action'] == 'positive': batch_pos.append(track_id)
        batch.append({"uid":uid, "batch_pos": batch_pos})
    
    return batch

@load_data_model
def generate_temporary_user(dataset, model, batch_data, alpha=0.2) -> list:
    e_users = get_all_user_node(dataset, model)
    for batch_user in batch_data:
        user_emb = torch.mean(get_item_node(model, torch.tensor(batch_user['batch_pos'], dtype=torch.long)), dim=0)
        sim = torch.matmul(e_users, user_emb) / \
            (torch.norm(e_users, dim=1) * torch.norm(user_emb))
        sim_user = torch.topk(sim, k=1, largest=True)[1]
        batch_user['u_emb'] = alpha * user_emb + (1-alpha) * e_users.detach()[sim_user].squeeze()
        batch_user.pop('batch_pos', None)
    return batch_data

if __name__ == '__main__':
    batch_data = get_inter_data()
    config = OmegaConf.load('config.yaml')
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

    ### 전일에 적재된 유저들에 대한 임시 임베딩 생성 ###
    print(generate_temporary_user(config, ROOT_PATH, batch_data))