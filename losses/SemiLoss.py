import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiLoss(nn.Module):
    def __init__(self):
        super(SemiLoss, self).__init__()

    def forward(self, outputs_x, targets_x, outputs_u, targets_u):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu
