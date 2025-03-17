import torch
import torch.nn as nn
from lib.pointops2.functions import pointops
from util.hooks import QueryGroup
from torch_points3d.modules.KPConv.kernels import KPConvLayer
from torch_points3d.core.common_modules import FastBatchNorm1d



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.norm = norm_layer(in_channels) if norm_layer else None
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, feats, xyz, offset):

        n_offset, count = [int(offset[0].item()*self.ratio)+1], int(offset[0].item()*self.ratio)+1
        for i in range(1, offset.shape[0]):
            count += ((offset[i].item() - offset[i-1].item())*self.ratio) + 1
            n_offset.append(count)
        n_offset = torch.cuda.IntTensor(n_offset)
        idx = pointops.furthestsampling(xyz, offset, n_offset)  # (m)
        n_xyz = xyz[idx.long(), :]  # (m, 3)

        feats = pointops.queryandgroup(self.k, xyz, n_xyz, feats, None, offset, n_offset, use_xyz=False)  # (m, nsample, 3+c)
        m, k, c = feats.shape
        feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)  # (m, c)

        return feats, n_xyz, n_offset

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.k = k
        self.norm = norm_layer(in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, pxo, pko):
        p, x, o = pxo
        n_p, knn_idx, n_o = pko

        feats, _ = QueryGroup(self.k, p, n_p, x, knn_idx, o, n_o, use_xyz=False)

        m, k, c = feats.shape
        feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)  # (m, c)

        return feats, n_p, n_o


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))

    def forward(self, feats, xyz, support_xyz, offset, support_offset, support_feats=None):

        feats = self.linear1(support_feats) + pointops.interpolation(xyz, support_xyz, self.linear2(feats), offset, support_offset)
        return feats, support_xyz, support_offset


class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats


class KPConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(nn.Linear(in_channels, d_2, bias=False), FastBatchNorm1d(d_2, momentum=bn_momentum), activation)
        self.unary_2 = torch.nn.Sequential(nn.Linear(d_2, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum), activation)
        self.kpconv = KPConvLayer(d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False)
        # self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum)
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        shortcut = feats
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats += shortcut
        return feats

