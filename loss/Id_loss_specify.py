# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x


class Id_Loss_Specify(nn.Module):

    def __init__(self, opt, feature_length):
        super(Id_Loss_Specify, self).__init__()

        self.opt = opt

        W = []
        for i in range(2):
            W.append(classifier(feature_length, opt.class_num))
        self.W = nn.Sequential(*W)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_img = self.W[0](image_embedding)
        score_txt = self.W[1](text_embedding)

        id_loss_img = criterion(score_img, label)
        id_loss_txt = criterion(score_txt, label)

        loss = id_loss_img + id_loss_txt

        return loss

    def forward(self, image_embedding, text_embedding, label):

        loss = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return loss

