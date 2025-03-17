import torch
import torch.nn as nn
from lib.pointops2.functions import pointops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .attention import EnhancedChannelAttention, KeyPointAttention,FeatureFusion,LocalGeoEncoder,PropagateLayer
from util.hooks import local_polar_representation,QueryGroup



class LGP(nn.Module):
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,num_ksp=16):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.local_geo_encoder = LocalGeoEncoder(dim)

        self.geo_adapter = nn.Sequential(
            nn.Linear(9, dim),  
        )
        self.geo_adapter2 = nn.Sequential(
            nn.Linear(1, dim),  
        )
        self.geo_adapter3 = nn.Sequential(
            nn.Linear(1, dim),  
        )



        self.posi_q = nn.Sequential(
            nn.Linear(self.nsample*3, dim),  
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(dim),
            Rearrange('b c n-> b n c'),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.posi_k = nn.Sequential(
            nn.Linear(self.nsample*3, dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(dim),
            Rearrange('b c n-> b n c'),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.posi_v = nn.Sequential(
            nn.Linear(self.nsample*3, dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(dim),
            Rearrange('b c n-> b n c'),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.point_propagate = PropagateLayer(dim, num_heads, nsample)

        self.kn = nn.Parameter(torch.randn(num_ksp, 3))  
        self.kp_attn = KeyPointAttention(dim)
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, dim//4), 
            nn.ReLU(),
            nn.Linear(dim//4, dim)
        )
        self.fusion = FeatureFusion(dim)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo

        knn_idx, _ = pointops.knnquery(self.nsample, p, p, o, o)
        knn_idx = knn_idx.long()
        
        n = x.size(0)
        knn_xyz = p[knn_idx.view(-1).long()].view(n, self.nsample, 3)
        p_r = knn_xyz - p.unsqueeze(1)
        x_knn = x[knn_idx.view(-1).long()].view(n, self.nsample, self.dim)
        

        x = self.point_propagate((x, x_knn, knn_idx, p_r))
        

        geo_rep, geo_dis, vol_ratio = local_polar_representation(p, knn_idx)  


        vol_ratio = vol_ratio.expand(p.size(0), 1)  
        dist = torch.cdist(p, self.kn)
        dist_feat = self.dist_encoder(dist.unsqueeze(-1)) 
        dist_feat = self.kp_attn(dist_feat)  

        geo_rep = geo_rep.mean(dim=1) 
        geo_dis = geo_dis.max(dim=1).values  
        geo_rep = self.geo_adapter(geo_rep) 
        geo_dis = self.geo_adapter2(geo_dis)  
        vol_ratio = self.geo_adapter3(vol_ratio)  
        x = self.fusion(x, geo_rep, geo_dis, vol_ratio, dist_feat)  
        pos_x, _ = QueryGroup(self.nsample, p, p, x, knn_idx, o, o, local=True)
        pos, x = pos_x
        pos = pos.reshape(pos.size(0), self.nsample, -1)  

        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)


        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))


        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))
        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))


        out = torch.matmul(attn, (v + pos_v))
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = out[:, 0, :]  
        out = self.proj_drop(self.proj(out))

        return out

class LGA(nn.Module):
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,is_proj=False):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.is_proj = is_proj

        self.posi_q = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_k = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_v = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, p_r) -> torch.Tensor:
        pos = p_r.reshape(p_r.shape[0], self.nsample, -1)

        # (n, nsample, c)
        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)


        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))

        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))

        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))

        out = torch.matmul(attn, (v + pos_v))

        out = rearrange(out, 'b h n d -> b n (h d)')

        if(self.is_proj):
            out = torch.mean(out, dim=1) # (n, c)

        out = self.proj_drop(self.proj(out))

        return out



class LGA_C(nn.Module):
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.channel_attn = EnhancedChannelAttention(dim, reduction_ratio=reduction_ratio)
        self.scale = qk_scale or head_dim ** -0.5
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.posi_q = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_k = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_v = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pxo, patch_pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        n_p, n_x, n_o = patch_pxo # (m, 3), (m, c), (b)

        knn_idx, _ = pointops.knnquery(self.nsample, n_p, p, n_o, o) # (n, nsample)
        pos_n_x, _ = QueryGroup(self.nsample, n_p, p, n_x, knn_idx, n_o, o, use_xyz=True)  # (n, nsample, 3+c)
        pos, n_x = pos_n_x[:, :, :3], pos_n_x[:, :, 3:] # (n, nsample, 3), (n, nsample, c)
        pos = repeat(pos, 'n m c -> n k m c', k = self.nsample)  # (n, nsample, nsample, 3)

        pos = pos.reshape(pos.shape[0], self.nsample, -1)

        # (n, nsample, c)
        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)

        q, k, v = self.linear_q(x), self.linear_k(n_x), self.linear_v(n_x)  # (n, c), (n, nsample, c), (n, nsample, c)
        q = repeat(q, 'n c -> n k c', k = self.nsample)  # (n, nsample, c)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))

        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))

        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))
        
        out = torch.matmul(attn, (v + pos_v))
        out = rearrange(out, 'b h n d -> b n (h d)') # (n, nsample, c)




        out = torch.mean(out, dim=1) 
        out = self.channel_attn(out, p)  
        out = self.proj_drop(self.proj(out))
        return out
