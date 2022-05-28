import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCE(nn.Module):
    '''
        CELoss for mixup, label smoothing, etc. Labels are mixed one hot
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        return loss
