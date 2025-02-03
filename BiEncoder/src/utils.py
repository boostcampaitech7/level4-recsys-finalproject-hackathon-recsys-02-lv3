import torch.nn.functional as F


def cosine_triplet_margin_loss(anchor, positive, negative, config):
    '''
    코사인 유사도를 기반으로 한 트리플렛 마진 손실 함수

    Args:
        anchor (torch.Tensor): 기준 벡터
        positive (torch.Tensor): 양성 벡터
        negative (torch.Tensor): 음성 벡터
        config (OmegaConf): 모델 설정 정보

    Returns:
        torch.Tensor: 평균 손실 값
    '''

    # Compute cosine distance (1 - cosine similarity)
    pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=1)
    neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=1)

    # Apply triplet margin loss
    losses = F.relu(pos_dist - neg_dist + config.model.margin)

    return losses.mean()


class EarlyStopping:
    '''
    모델 학습 과정에서 Early Stopping을 수행

    Args:
        config (OmegaConf): 학습 설정 정보를 포함한 객체

    Attributes:
        patience (int): 조기 종료를 결정하는 에포크 수
        min_delta (float): 손실 값의 최소 개선 정도
        best_loss (float, optional): 현재까지의 최소 손실
        counter (int): 개선되지 않은 에포크 수
        early_stop (bool): 학습을 중단해야 하는지 여부
    '''

    def __init__(self, config):
        self.patience = config.training.early_stopping_patience
        self.min_delta = config.training.early_stopping_min_delta  
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        '''
        현재 에포크의 손실 값을 기반으로 조기 종료 여부를 결정
        
        '''

        # Initialize best loss if it's the first calls
        if self.best_loss is None:
            self.best_loss = val_loss  

        # If validation loss improves (drops by min_delta), reset counter
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss  
            self.counter = 0 

        # If validation loss does not improve, increment counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
