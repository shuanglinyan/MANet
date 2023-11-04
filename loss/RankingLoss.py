# -*- coding: utf-8 -*-
"""

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def calculate_similarity(image_embedding, text_embedding):
    image_embedding = image_embedding.view(image_embedding.size(0), -1)
    text_embedding = text_embedding.view(text_embedding.size(0), -1)
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    # similarity_match = torch.sum(image_embedding_norm * text_embedding_norm, dim=1)

    return similarity


def calculate_margin_cr(similarity_match_cr, similarity_match, auto_margin_flag, margin):
    if auto_margin_flag:
        lambda_cr = abs(similarity_match_cr.detach()) / abs(similarity_match.detach())
        ones = torch.ones_like(lambda_cr)
        data = torch.ge(ones, lambda_cr).float()
        data_2 = torch.ge(lambda_cr, ones).float()
        lambda_cr = data * lambda_cr + data_2

        lambda_cr = lambda_cr.detach().cpu().numpy()
        margin_cr = ((lambda_cr + 1) * margin) / 2.0
    else:
        margin_cr = margin / 2.0

    return margin_cr


class RankingLoss(nn.Module):

    def __init__(self, opt):
        super(RankingLoss, self).__init__()

        self.device = opt.device
        self.margin = np.array([opt.margin]).repeat(opt.batch_size)
        self.beta = opt.cr_beta

    def semi_hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None
    
    def hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss > margin, loss > 0))[0]
        return np.argmax(loss) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels, auto_margin_flag, margin, semi):

        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together
            if margin[idx] >= 0.16 or auto_margin_flag is False:
                negative = np.where(labels != label)[0]

                ap_sim = similarity[idx, idx]

                loss = similarity[idx, negative] - ap_sim + margin[idx]

                if semi:
                    negetive_index = self.semi_hard_negative(loss, margin[idx])
                else:
                    negetive_index = self.hard_negative(loss, margin[idx])

                if negetive_index is not None:
                    triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = torch.LongTensor(np.array(triplets))

        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).to(self.device)

        return triplets, return_margin

    def calculate_loss(self, similarity, label, auto_margin_flag, margin, semi):

        image_triplets, img_margin = self.get_triplets(similarity, label, auto_margin_flag, margin, semi)
        text_triplets, txt_margin = self.get_triplets(similarity.t(), label, auto_margin_flag, margin, semi)
        # print(img_margin[:10], img_margin.size())
        # print(txt_margin[:10], txt_margin.size())
        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        similarity = similarity.t()
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

    def forward(self, img, txt, labels, auto_margin_flag, semi):

        similarity = calculate_similarity(img, txt)

        cr_loss = self.calculate_loss(similarity, labels, auto_margin_flag, self.margin, semi)

        return cr_loss

