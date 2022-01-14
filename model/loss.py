import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output,target):
    return F.cross_entropy(output,target)

def BCELoss():
    return nn.BCELoss()