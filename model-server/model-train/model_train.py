import utils
import torch
import numpy as np
import time
from omegaconf import OmegaConf
import os
import Procedure
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from airflow import DAG
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from .utils import Config, Directory
from LightGCN.code.model import LightGCN
from LightGCN.code.batch_dataloader import Loader
from LightGCN.code.utils import EarlyStopping


def get_user_pos_neg_data():
    username = Config.MONGO_DB_NAME
    password = Config.MONGO_DB_PASSWORD  # 특수문자가 있는 경우 인코딩

    uri = f"mongodb+srv://{username}:{password}@cluster0.octqq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client['spotify']
    collection = db['interaction']
    print("connect to db successfully!")

    unique_user_ids = collection.distinct('user_id')
    dic = {}
    for user_id in tqdm(unique_user_ids):
        pos_list = []
        neg_list = []
        query = {"user_id": user_id}
        user_list = list(collection.find(query))
        for x in user_list:
            if x['action'] == 'positive':
                pos_list.append(x['track_id'])
            elif x['action'] == 'negative':
                neg_list.append(x['track_id'])
        dic[user_id] = {'pos' : pos_list, "neg" : neg_list}

    pos_path = os.path.join(Directory.LIGHTGCN_DIR, 'data/pos.txt')
    neg_path = os.path.join(Directory.LIGHTGCN_DIR, 'data/neg.txt')

    # 기존 파일 내용 읽기
    pos_existing_data = {}
    with open(pos_path, 'r') as pos_file:
        for line in pos_file:
            user_id, tracks = line.strip().split('\t')
            pos_existing_data[int(user_id)] = set(map(int, tracks.split()))


    neg_existing_data = {}
    with open(neg_path, 'r') as neg_file:
        for line in neg_file:
            user_id, tracks = line.strip().split('\t')
            neg_existing_data[int(user_id)] = set(map(int, tracks.split()))

    # 새로운 데이터 병합
    for user_id, action_list in dic.items():
        if user_id in pos_existing_data and user_id in neg_existing_data:
            # 기존 트랙에 새로운 트랙 추가
            pos_existing_data[user_id].update(map(int, action_list['pos']))
            neg_existing_data[user_id].update(map(int, action_list['neg']))
        else:
            # 새로운 유저 데이터 추가
            pos_existing_data[user_id] = set(map(int, action_list['pos']))
            neg_existing_data[user_id] = set(map(int, action_list['neg']))
    print(f"pos_existing_data : {len(pos_existing_data)}")
    print(f"neg_existing_data : {len(neg_existing_data)}")
    
    # 수정된 데이터 저장
    with open(pos_path, 'w') as pos_file:
        for user_id, tracks in pos_existing_data.items():
            # 트랙 ID를 정렬하여 저장
            tracks_str = ' '.join(map(str, sorted(tracks)))
            pos_file.write(f"{user_id}\t{tracks_str}\n")

    with open(neg_path, 'w') as neg_file:
        for user_id, tracks in neg_existing_data.items():
            # 트랙 ID를 정렬하여 저장
            tracks_str = ' '.join(map(str, sorted(tracks)))
            neg_file.write(f"{user_id}\t{tracks_str}\n")

def get_model_train():
    mlflow.set_experiment("lightgcn_experiment")

    config_path = os.path.join(Directory.LIGHTGCN_DIR, 'config.yaml')
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))

    directory_path = os.path.join(Directory.LIGHTGCN_DIR, config.path.FILE)
    print(directory_path)
    os.makedirs(directory_path, exist_ok=True)

    weight_file_dir = os.path.join(directory_path, f'best_model.pth')
    print(f"load and save to {weight_file_dir}")

    dataloader = Loader(config=config, path=os.path.join(Directory.LIGHTGCN_DIR, config.path.DATA))
    model = LightGCN(config, dataloader)
    model = model.to(torch.device(config.device))
    bpr = utils.BPRLoss(model, config)

    if config.finetune:
        try:
            model.load_state_dict(torch.load(weight_file_dir,map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file_dir}")
        except FileNotFoundError:
            print(f"{weight_file_dir} not exists, start from beginning")

    es = EarlyStopping(model=model,
                    patience=10, 
                    delta=0, 
                    mode='min', 
                    verbose=True,
                    path=os.path.join(Directory.LIGHTGCN_DIR, config.path.FILE, 'best_model.pth')
                    )

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        for epoch in tqdm(range(config.epochs)):
            if config.test and epoch % 10 == 0:
                print(f"Testing at epoch {epoch}")
                Procedure.Test(config, dataloader, model, epoch, w)
            output_info, avr_loss = Procedure.BPR_train_original(config, dataloader, model, bpr, epoch, 1, w)
            if epoch % 5 == 0:
                print(f'EPOCH[{epoch+1}/{config.epochs}] {output_info}')
            torch.save(model.state_dict(), weight_file_dir)
            es(avr_loss)
            mlflow.log_metric("loss", avr_loss, step=epoch)
            if es.early_stop:
                print(f'Early stopping at epoch {epoch+1}, avr_loss: {avr_loss}')
                break

        mlflow.pytorch.log_model(model, artifact_path="model")
        model_name = "LightGCN"
        client = MlflowClient()
    
        model_uri = f"runs:/{run.info.run_id}/model"
        model_version = client.create_model_version(name=model_name, source=model_uri, run_id=run.info.run_id).version
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        print("Model logged to MLflow")


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'depends_on_past': False
}


with DAG('lightgcn_batch_train_dag',
        default_args=default_args,
        schedule='0 15 */4 * *',  # 4일에 한번 한국시간(KST, UTC+9)00시에 배치학습
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_user_pos_neg_data_task = PythonOperator(
        task_id='get_user_pos_neg_data_task',
        python_callable=get_user_pos_neg_data
    )

    get_model_train_task = PythonOperator(
        task_id='get_model_train_task',
        python_callable=get_model_train,
        trigger_rule=TriggerRule.ONE_FAILED
    )
    
    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> get_user_pos_neg_data_task >> get_model_train_task >> end_task
