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


config = OmegaConf.load('config.yaml')
print(OmegaConf.to_yaml(config))

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
weight_file = utils.getFileName(ROOT_PATH, config)
print(f"load and save to {weight_file}")

dataloader = Loader(config=config, path=os.path.join(ROOT_PATH,config.path.DATA))
model = LightGCN(config, dataloader)
model = model.to(torch.device(config.device))
bpr = utils.BPRLoss(model, config)

if config.finetune:
    try:
        model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

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
    for epoch in range(config.epochs):
        start = time.time()
        if config.test and epoch %10 == 0:
            print(f"test at epoch {epoch}")
            Procedure.Test(config, dataloader, model, epoch, w)
        output_information, avr_loss = Procedure.BPR_train_original(config, dataloader, model, bpr, epoch, 1, w)
        if epoch % 5 == 0:
            print(f'EPOCH[{epoch+1}/{config.epochs}] {output_information}')
        torch.save(model.state_dict(), weight_file)
        # early stopping -> 10 epoch 동안 loss 값이 줄어들지 않을 경우 학습 종료
        es(avr_loss)
        if es.early_stop:
            print(f'Early Stopping at {epoch+1}, avr_loss:{avr_loss}')
            break
finally:
    if config.tensorboard:
        w.close()