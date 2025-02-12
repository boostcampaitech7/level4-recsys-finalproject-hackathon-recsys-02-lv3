"""
사용자의 positive/negative item을 함께 모델링하는 데이터셋
"""
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sp
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    @property
    def allNeg(self):
        raise NotImplementedError
    
    @property
    def valSet(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config, path):
        self.config = config
        self.split = config.A_split
        self.folds = config.A_n_fold
        self.n_user = 0
        self.m_item = 0
        #train_file = path + '/train.txt'
        pos_file = os.path.join(path, 'pos.txt')
        neg_file = os.path.join(path, 'neg.txt')
        self.path = path
        trainUniqueUsers, trainItem, trainUser, testItem, testUser = [], [], [], [], []
        self.traindataSize = 0

        with open(pos_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    uid, items = l.strip('\n').split('\t')
                    uid = int(uid)
                    items = list(set(list(map(int, items.split()))))
                    train_items = items
                    random.shuffle(items)
                    if config.test:
                        
                        split = int(len(items)*0.8)
                        train_items, test_items = items[:split], items[split:]
                        testUser.extend([uid]*len(test_items))
                        testItem.extend(test_items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(train_items))
                    trainItem.extend(train_items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(train_items)
        
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        assert len(self.testUser) == len(self.testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"Data Sparsity : {(self.trainDataSize) / self.n_users / self.m_items}")
        
        with open(neg_file) as f:
            neg_list = []
            for l in f.readlines():
                if len(l) > 0:
                    uid, items = l.strip('\n').split('\t')
                    items = list(map(int, items.split()))
                    uid = int(uid)
                    trainUniqueUsers.append(uid)
                    self.m_item = max(self.m_item, max(items)) if len(items)!=0 else self.m_item
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                    for iid in items:
                        neg_list.append((uid,iid))
        self.m_item += 1
        self.n_user += 1
        neg_lil_net = lil_matrix((self.n_user, self.m_item))
        
        for pair in neg_list:
            neg_lil_net[pair] = -1
        neg_lil_net = neg_lil_net.tocoo()
        
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        print(f"user, item matrix shape is: {self.UserItemNet.shape}")

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosItems(users=list(range(self.n_user)))
        self._allNeg = self.getUserNegItems(users=list(range(self.n_user)), 
                                            neg_net=neg_lil_net)
        if config.test:
            self._valSet = self.buildValidSet()
        print(f"Data is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos
    
    @property
    def allNeg(self):
        return self._allNeg
    
    @property
    def valSet(self):
        return self._valSet

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.config.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        if self.Graph is None:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.config.device)
                print("don't split the matrix")
        return self.Graph

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users, neg_net):
        negItems = []
        negValues = neg_net.data
        mask = (negValues == -1)
        r, c = neg_net.row[mask], neg_net.col[mask]
        for user in users:
            negIdx = np.where(r==user)[0]
            negItems.append(c[negIdx])
        return negItems

    def buildValidSet(self):
        testDict = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if testDict.get(user):
                testDict[user].append(item)
            else:
                testDict[user] = [item]
        return testDict