import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from utils import EarlyStopping

Recmodel = register.MODELS[world.model_name](world.config, register.DATA)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

es = EarlyStopping(model=Recmodel,
                   patience=10, 
                   delta=0, 
                   mode='min', 
                   verbose=True
                  )

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(register.DATA, Recmodel, epoch, w, world.config['multicore'])
        output_information, avr_loss = Procedure.BPR_train_original(register.DATA, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        if epoch % 5 == 0:
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
        # early stopping -> 10 epoch 동안 loss 값이 줄어들지 않을 경우 학습 종료
        es(avr_loss)
        if es.early_stop:
            print(f'Early Stopping at {epoch+1}, avr_loss:{avr_loss}')
            break
finally:
    if world.tensorboard:
        w.close()