'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import numpy as np
import torch
import utils
from utils import timer
import model


def BPR_train_original(config, dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        # 이 부분을 다른 Sampler로 대체
        S = utils.UniformSample_original_python(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(config.device)
    posItems = posItems.to(config.device)
    negItems = negItems.to(config.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config.train.bpr_batch_size + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config.train.bpr_batch_size)):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if config.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / config.train.bpr_batch_size) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}", aver_loss
    

            
def Test(config, dataset, Recmodel, epoch, w=None):
    def test_one_batch(X, topks):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in topks:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}
            
    u_batch_size = 100
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config.topks)

    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            
            # 이미 존재하는 값들 exclude
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            # rating_list, groundTrue는 각각 len==batch인 리스트.
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, config.topks))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        if config.tensorboard:
            w.add_scalars(f'Test/Recall@{config.topks}',
                          {str(config.topks[i]): results['recall'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{config.topks}',
                          {str(config.topks[i]): results['precision'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{config.topks}',
                          {str(config.topks[i]): results['ndcg'][i] for i in range(len(config.topks))}, epoch)
        

        print(results)
        return results
