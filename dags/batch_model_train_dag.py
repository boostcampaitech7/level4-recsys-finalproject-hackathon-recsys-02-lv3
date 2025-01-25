import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from omegaconf import OmegaConf
import os
import Procedure
from model import LightGCN
from batch_dataloader import Loader
from utils import EarlyStopping
from tqdm import tqdm
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import XCom
from airflow.utils.session import create_session
from airflow.providers.google.cloud.hooks.gcs import GCSHook

from LightGCN.code.model import LightGCN
from LightGCN.code.batch_dataloader import Loader
from LightGCN.code.utils import EarlyStopping


def get_model_train():
    ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LightGCN')
    config_path = os.path.join(ROOT_PATH, 'config.yaml')
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))

    today_dir = datetime.now().strftime('%y-%m-%d')
    directory_path = os.path.join(ROOT_PATH, config.path.FILE, today_dir)
    print(directory_path)
    os.makedirs(directory_path, exist_ok=True)

    weight_file_dir = os.path.join(directory_path, f'{today_dir}.pth.tar')
    print(f"load and save to {weight_file_dir}")

    dataloader = Loader(config=config, path=os.path.join(ROOT_PATH,config.path.DATA))
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
                    path=os.path.join(ROOT_PATH, config.path.FILE, 'best_model.pth')
                    )

    # init tensorboard
    if config.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        os.path.join(ROOT_PATH, config.path.BOARD, time.strftime("%m-%d-%Hh%Mm%Ss-"))
                                        )
    else:
        w = None
        print("not enable tensorflowboard")

    try:
        stopping_step = 0
        cur_best_pre_0 = 0
        should_stop = False
        for epoch in tqdm(range(config.epochs)):
            start = time.time()
            if config.test and epoch %10 == 0:
                print(f"test at epoch {epoch}")
                Procedure.Test(config, dataloader, model, epoch, w)
            output_information, avr_loss = Procedure.BPR_train_original(config, dataloader, model, bpr, epoch, 1, w)
            if epoch % 5 == 0:
                print(f'EPOCH[{epoch+1}/{config.epochs}] {output_information}')
            torch.save(model.state_dict(), weight_file_dir)
            # early stopping -> 10 epoch 동안 loss 값이 줄어들지 않을 경우 학습 종료
            es(avr_loss)
            if es.early_stop:
                print(f'Early Stopping at {epoch+1}, avr_loss:{avr_loss}')
                break
    finally:
        if config.tensorboard:
            w.close()


def upload_file_to_gcs(bucket_name: str, replace: bool = True) -> None:
    """ Airflow Hook으로 GCS에 파일을 업로드하는 메서드
    Args:
        source_path (str): 업로드할 로컬 파일 경로
        destination_path (str): GCS에 저장될 객체 이름/경로
        bucket_name (str): GCS 버킷 이름
        replace (bool): 덮어쓰기 여부 (기본값: True)
    """
    hook = GCSHook(gcp_conn_id="google_cloud_default")
    
    if not replace and hook.exists(bucket_name=bucket_name, object_name=destination_path):
        print(f"Object {destination_path} already exists in bucket {bucket_name}")
        return
    ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LightGCN')
    today_dir = datetime.now().strftime('%y-%m-%d')
    directory_path = os.path.join(ROOT_PATH, 'checkpoints', today_dir)
    weight_file_dir = os.path.join(directory_path, f'{today_dir}.pth.tar')

    destination_path = f'model/{today_dir}/{today_dir}.pth.tar'    
    hook.upload(
        bucket_name=bucket_name,
        object_name=destination_path,
        filename=weight_file_dir
    )


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG('batch_model_train_dag',
        default_args=default_args,
        schedule='0 0 * * *',
        start_date=datetime(2024, 1, 1),
        catchup=False
    ):

    start_task = EmptyOperator(
        task_id="start_task"
    )

    get_model_train_task = PythonOperator(
        task_id='get_model_train_task',
        python_callable=get_model_train,
        provide_context=True
    )

    upload_file_to_gcs_task = PythonOperator(
        task_id='upload_file_to_gcs_task',
        python_callable=upload_file_to_gcs,
        op_kwargs= {
            "bucket_name": "model"
        },
        provide_context=True
    )
    
    end_task = EmptyOperator(
        task_id="end_task"
    )

    start_task >> get_model_train_task >> upload_file_to_gcs_task >> end_task