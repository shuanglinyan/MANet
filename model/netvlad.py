from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d


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

class pro_block(nn.Module):

    def __init__(self, in_planes=512, out_planes=512):

        super(pro_block, self).__init__()

        self.fc = nn.Linear(in_planes, out_planes)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)


    def forward(self, x):
        if (len(list(x.shape))) > 3:
            x = self.fc(x).permute(0, 3, 1, 2)
            x = self.bn2(x).permute(0, 2, 1, 3)
        elif (len(list(x.shape))) > 2:
            x = self.fc(x).permute(0, 2, 1)
            x = self.bn1(x)
        else:
            x = self.fc(x)
            x = self.bn1(x)
        x = self.relu(x)

        return x

class attention_generator(nn.Module):

    def __init__(self, in_planes=512, out_planes=512):

        super(attention_generator, self).__init__()

        self.fc = nn.Linear(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.sigmoid = nn.Sigmoid()
        self.fc.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)


    def forward(self, x):

        x = self.fc(x).permute(0, 3, 1, 2)
        x = self.bn(x).permute(0, 2, 1, 3)
        x = self.sigmoid(x)
        # x = self.fc(x).permute(0, 1, 3, 2)

        return x

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
        # self.conv_img.weight = nn.Parameter(
        #     (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        # self.conv_img.bias = nn.Parameter(
        #     - self.alpha * self.centroids.norm(dim=1))
        # self.conv_txt.weight = nn.Parameter(
        #     (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        # self.conv_txt.bias = nn.Parameter(
        #     - self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C, H, _ = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        local_fea = vlad

        return local_fea


class NetVLAD_V1(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, dim_center=512, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD_V1, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.dim_center = dim_center
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, dim_center, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim_center))
        self._init_params()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 512))

        self.theta = pro_block(in_planes=dim_center, out_planes=dim_center//4)
        self.phi = pro_block(in_planes=dim_center, out_planes=dim_center//4)
        self.rm = pro_block(in_planes=dim_center//4, out_planes=dim_center)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, C, H, _ = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = self.conv(x).view(N, self.dim_center, -1)
        
        # calculate residuals to each clusters
        theta_x = self.theta(x_flatten.permute(0, 2, 1))  # N, 128, 48
        theta_y = self.phi(self.centroids)  # K, 128

        residual = theta_x.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            theta_y.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)     # 4， 6， 128， 48
        relation = self.rm(residual.permute(0, 1, 3, 2))  # 4， 6， 512， 48
        # relation = F.softmax(relation, dim=1)
        vlad = torch.mul(relation, x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)).sum(dim=-1)
        local_fea = vlad
        return local_fea

        # concat 
        # concat_fea = vlad.view(x.size(0), -1).unsqueeze(1)
        # return concat_fea

        # max pooling
        # pool_fea = self.max_pool(vlad)
        # return pool_fea, local_fea

class NetVLAD_V2(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, dim_center=512, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD_V2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.dim_center = dim_center
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, dim_center, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim_center))
        self._init_params()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 512))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
        # self.conv_img.weight = nn.Parameter(
        #     (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        # self.conv_img.bias = nn.Parameter(
        #     - self.alpha * self.centroids.norm(dim=1))
        # self.conv_txt.weight = nn.Parameter(
        #     (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        # self.conv_txt.bias = nn.Parameter(
        #     - self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C, H, _ = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = self.conv(x).view(N, self.dim_center, -1)

        # calculate residuals to each clusters
        temp = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
        temp1 = F.normalize(self.centroids, p=2, dim=1).expand(N, -1, -1).unsqueeze(2)
        weight = torch.matmul(temp1, temp)
        soft_assign = F.softmax(weight, dim=-1).squeeze().expand(x_flatten.size(1), -1, -1, -1).permute(1, 2, 0, 3)
        vlad = torch.mul(soft_assign, temp).sum(dim=-1)
        local_fea = vlad

        # concat
        # concat_fea = vlad.view(x.size(0), -1).unsqueeze(1)
        # return concat_fea, local_fea

        # max pooling
        # pool_fea = self.max_pool(vlad)
        # return pool_fea, local_fea

        return local_fea