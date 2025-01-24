import torch.nn.functional as F
import torch

def cosine_triplet_margin_loss(anchor, positive, negative, margin=0.2):
    pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=1)
    neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=1)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()
