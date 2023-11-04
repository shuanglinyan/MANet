# -*- coding: utf-8 -*-
"""

@author: zifyloo
"""

from torch import nn
from torch._C import set_flush_denormal
from transformers.utils.dummy_pt_objects import RetriBertPreTrainedModel
from .text_feature_extract import TextExtract
from torchvision import models
import torch
from torch.nn import init
from torch.nn import functional as F
from .netvlad import NetVLAD, NetVLAD_V1
# from .netvlad_ori import NetVLAD_V1
import numpy as np
from collections import OrderedDict
from model.GCN_lib.Rs_GCN import Rs_GCN
from model.SEAttention import SEAttention
from .bert import Bert

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class pro_block(nn.Module):

    def __init__(self, in_planes=512, out_planes=512):

        super(pro_block, self).__init__()

        self.fc = nn.Linear(in_planes, out_planes)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.bn(x)
        x = self.relu(x)

        return x


class pro_block1(nn.Module):

    def __init__(self, in_planes=512, out_planes=512):

        super(pro_block1, self).__init__()

        self.fc = nn.Linear(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x).permute(0, 3, 1, 2)
        x = self.bn(x)
        x = self.relu(x)

        return x


class attention_generator(nn.Module):

    def __init__(self, in_planes=512, out_planes=512):

        super(attention_generator, self).__init__()

        self.fc = nn.Linear(in_planes, out_planes)
        self.bn = nn.BatchNorm1d(out_planes)
        self.sigmoid = nn.Sigmoid()
        self.fc.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):

        x = self.fc(x).permute(0, 2, 1)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x

class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x


class EncoderAttn(nn.Module):

    def __init__(self, in_channel, s_ratio, r_ratio, dim_spatial):
        super(EncoderAttn, self).__init__()

        self.theta = pro_block(in_channel, in_channel // s_ratio)
        self.phi = pro_block(in_channel, in_channel // s_ratio)
        self.re = pro_block1(in_channel // s_ratio, in_channel // r_ratio)
        inter_channel = dim_spatial * (in_channel // r_ratio) + in_channel
        self.att = attention_generator(inter_channel, in_channel)


    def forward(self, inputs, retain):
        """Extract image feature vectors."""
        # input: B, C, N
        B, C, N = inputs.size()
        theta_x = self.theta(inputs).permute(0, 2, 1)
        theta_x = theta_x.unsqueeze(2).expand(-1, -1, N, -1)
        phi_x = self.phi(inputs).permute(0, 2, 1)
        phi_x = phi_x.unsqueeze(1).expand(-1, N, -1, -1)
        diff = theta_x - phi_x
        rel = self.re(diff).permute(0, 2, 3, 1).view(B, N, -1)
        att = self.att(torch.cat([inputs.permute(0, 2, 1), rel], dim=2))
        features = torch.mul(att, inputs)
        if retain:
            return features
        else:
            features = torch.sum(features, dim=2)/torch.sum(att, dim=2)
        return features


class SNR_Block(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SNR_Block, self).__init__()

        self.IN = nn.InstanceNorm2d(in_channel, affine=True)
        self.se_block = SEAttention(channel=in_channel, reduction=reduction)

    def forward(self, x):
        # x: B, C, H, W
        x_norm = self.IN(x)
        R = x - x_norm
        mask = self.se_block(R)
        R_plus = R * mask.expand_as(R)
        output = x_norm + R_plus + x
        return output


class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 24 8
        )
        
    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        return x4


class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        # Backbone
        self.ImageExtract = ResNet_image_50()
        self.TextExtract = TextExtract(opt)

        # RAR module
        self.img_enc = EncoderAttn(2048, 32, 256, 192)
        # self.text_enc = EncoderAttn(2048, 32, 256, 100)

        # CAF module
        self.snr_block = SNR_Block(in_channel=2048, reduction=8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Global Alignment
        self.visual_embed_layer = nn.Linear(2048, 2048)
        self.textual_embed_layer = nn.Linear(2048, 2048)

        # Baseline-SSAN
        # self.conv_global = conv(2048, 2048)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Local Alignment
        dim = list(self.ImageExtract.parameters())[-1].shape[0]
        self.Net_VLAD = NetVLAD_V1(num_clusters=opt.num_clusters, dim=dim, alpha=1.0)

    def forward(self, image, tokens, segments, input_masks):

        # img_pre, img_aft, img_global, img_local
        img_pre, img_aft, img_global, img_local = self.img_embedding(image)
        txt_global, txt_local = self.txt_embedding(tokens, segments, input_masks)

        return img_pre, img_aft, img_global, img_local, txt_global, txt_local

    def img_embedding(self, image):

        image_feature = self.ImageExtract(image)

        # RAR module
        B, C, H, W = image_feature.size()
        image_feature = self.img_enc(image_feature.view(B, C, -1), retain=True).view(B, C, H, W)

        # CAF module
        image_feature_pre = self.avg_pool(image_feature).squeeze()
        image_feature = self.snr_block(image_feature)
        image_feature_aft = self.avg_pool(image_feature).squeeze()

        # global image feature
        image_feature_emb = image_feature.permute(0, 2, 3, 1)  # B, H, W, 2048
        image_feature_emb = self.visual_embed_layer(image_feature_emb).permute(0, 3, 1, 2)  # B, 2048, H, W
        image_global = self.global_maxpool(image_feature_emb)
        image_global = image_global.squeeze()

        # local image feature
        image_local = self.Net_VLAD(image_feature)   # B, part, 512

        if self.training:
            return image_feature_pre.unsqueeze(1), image_feature_aft.unsqueeze(1), image_global.unsqueeze(1), image_local
        else:
            return image_global.unsqueeze(1), image_local

    def txt_embedding(self, tokens, segments, input_masks):

        # Bert+LSTM
        text_feature_g = self.TextExtract(tokens, segments, input_masks)  # B, C, 100, 1
        
        # global text feature
        text_feature_global, _ = torch.max(text_feature_g, dim=2, keepdim=True)
        text_global = text_feature_global.squeeze()
        text_global = self.textual_embed_layer(text_global)
        
        # local text feature
        text_local = self.Net_VLAD(text_feature_g)  # B, part, 2048

        return text_global.unsqueeze(1), text_local

