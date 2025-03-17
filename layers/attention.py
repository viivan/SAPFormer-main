import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum

from util.hooks import local_polar_representation
from timm.models.layers import DropPath

class PropagateLayer(nn.Module):
    def __init__(self, in_planes, share_planes, nsample):
        super().__init__()
        self.share_planes = share_planes
        self.nsample = nsample
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, in_planes, bias=False),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(inplace=True)
        )
        
        self.mixer = nn.Sequential(
            nn.Linear(2*in_planes, in_planes//share_planes),
            nn.ReLU(inplace=True)
        )
        
        self.value_proj = nn.Sequential(
            nn.Linear(in_planes, in_planes//share_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):

        x_center, x_neighbors, neighbor_idx, rel_pos = inputs

        pos_emb = self.pos_encoder(rel_pos.view(-1, 3)) 

        mixed_feat = torch.cat([
            pos_emb, 
            x_neighbors.view(-1, x_neighbors.size(-1))
        ], dim=1)  

        mixer_out = self.mixer(mixed_feat)  

        attn_weights = scatter_softmax(
            mixer_out, 
            neighbor_idx.view(-1, 1).expand(-1, mixer_out.size(-1)), 
            dim=0
        )  
        

        values = self.value_proj(x_neighbors.view(-1, x_neighbors.size(-1)))  
        

        weighted_values = values * attn_weights
        
        aggregated = scatter_sum(
            weighted_values,
            neighbor_idx.view(-1, 1).expand(-1, weighted_values.size(-1)),
            dim=0,
            dim_size=x_center.size(0)
        )  
        

        output = aggregated.repeat(1, self.share_planes)  
        
        return x_center + output


class EnhancedChannelAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=4, pos_dim=3):
        super().__init__()
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim)
        )
        
        self.channel_attn = nn.Sequential(
            nn.Linear(dim*2, dim // reduction_ratio), 
            nn.BatchNorm1d(dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )
        
        self.pos_gate = nn.Sequential(
            nn.Linear(pos_dim, dim),
            nn.Sigmoid()
        )
        self.deform_offset = nn.Sequential(
            nn.Linear(pos_dim, dim),
            nn.Tanh()  
        )

    def forward(self, feat, pos):
        pos_code = self.pos_encoder(pos)  
        
        gate = self.pos_gate(pos)  
        
        offset = self.deform_offset(pos)  
        enhanced_feat = feat * (gate + offset) + pos_code
        
        combined = torch.cat([enhanced_feat, pos_code], dim=1)  
        channel_weights = self.channel_attn(combined)
        
        return feat * channel_weights  


class LocalGeoEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.geo_encoder = nn.Sequential(
            nn.Linear(9, feat_dim//2),  
            nn.BatchNorm1d(feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, feat_dim)
        )
        self.dis_encoder = nn.Sequential(
            nn.Linear(1, feat_dim//2),  
            nn.BatchNorm1d(feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, feat_dim)
        )
        self.vol_encoder = nn.Sequential(
            nn.Linear(1, feat_dim//2),  
            nn.BatchNorm1d(feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, feat_dim)
        )
        
    def forward(self, xyz, neighbor_idx):

        local_rep, geometric_dis, vol_ratio = local_polar_representation(xyz, neighbor_idx)

        N, K, _ = local_rep.shape
        encoded_geo = self.geo_encoder(local_rep.view(-1, 9))  # [N*K, C]
        encoded_geo = encoded_geo.view(N, K, -1)

        encoded_dis = self.dis_encoder(geometric_dis.view(-1, 1))  # [N*K, C]
        encoded_dis = encoded_dis.view(N, K, -1)

        encoded_vol = self.vol_encoder(vol_ratio.view(-1, 1))
        encoded_vol = encoded_vol.view(N, 1, -1)


        
        return encoded_geo, encoded_dis, encoded_vol


class KeyPointAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(feat_dim, num_heads)
        
    def forward(self, dist_feat):
        # dist_feat: [N, M, C]
        dist_feat = dist_feat.permute(1, 0, 2)  
        attn_out, _ = self.attn(dist_feat, dist_feat, dist_feat)
        return attn_out.permute(1, 0, 2).mean(dim=1)  

class FeatureFusion(nn.Module):
    def __init__(self, feat_dim, num_features=4):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Linear(feat_dim * num_features, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, num_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, geo_rep, geo_dis, vol_ratio, dist_feat):
        combined = torch.cat([geo_rep, geo_dis, vol_ratio, dist_feat], dim=-1)

        weights = self.weights(combined)

        weighted_sum = (
            weights[:, 0].unsqueeze(-1) * geo_rep +
            weights[:, 1].unsqueeze(-1) * geo_dis +
            weights[:, 2].unsqueeze(-1) * vol_ratio +
            weights[:, 3].unsqueeze(-1) * dist_feat
        )
        
        return x + weighted_sum