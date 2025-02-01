import torch.nn.functional as F

def cosine_triplet_margin_loss(anchor, positive, negative, margin=0.2):
    pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=1)
    neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=1)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss  
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss  
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
