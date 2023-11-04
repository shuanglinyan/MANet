
import torch.nn as nn
import torch

# from .metric import *

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss















