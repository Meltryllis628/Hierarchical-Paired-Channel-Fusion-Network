import torch.nn as nn
import torch.nn.functional as F
import torch


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        outputs = F.log_softmax(outputs,1)
        n, c, w, h = outputs.shape
        label_sums = torch.sum(torch.sum(label, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        total = w * h
        weight = label_sums / total
        loss = label * outputs
        loss *= weight
        entire_loss = -torch.sum(loss)/(n*total)
        return entire_loss
