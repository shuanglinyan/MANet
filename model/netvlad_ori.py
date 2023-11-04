import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.conv_txt = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 2048))

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

        # soft-assignment
        # if H > 1:
        #     soft_assign = self.conv_img(x).view(N, self.num_clusters, -1)
        # else:
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        local_fea = vlad

        # vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # concat 
        # global_fea = vlad.view(x.size(0), -1)  
        # max pooling
        global_fea = self.max_pool(vlad).squeeze()
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return global_fea, local_fea

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
        # self.conv_txt = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
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

        # soft-assignment
        # if H == 24:
        #     soft_assign = self.conv_img(x).view(N, self.num_clusters, -1)
        # else:
        # soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = self.conv(x).view(N, self.dim_center, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        soft_assign = F.softmax(residual, dim=-1)
        # residual *= soft_assign.unsqueeze(2)
        vlad = torch.mul(soft_assign, x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)).sum(dim=-1)
        local_fea = vlad

        # vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # concat 
        # global_fea = vlad.view(x.size(0), -1)
        # max pooling
        # global_fea = self.max_pool(vlad).squeeze()
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return local_fea

class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)
