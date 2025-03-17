import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from layers.process import KPConvSimpleBlock, KPConvResBlock, TransitionDown, DownSample, Upsample
from lib.pointops2.functions import pointops
from util.hooks import Sampling,QueryGroup
from layers import TripleAttention
from layers import Mlp, TransitionDown, DownSample
from layers import LGA, LGP, LGA_C


class TripleAttention(nn.Module):
    def __init__(self, dim, num_heads, k=16, act_layer=nn.GELU, norm_layer=nn.LayerNorm, downscale=8,
                 aggre='max', mlp_ratio=4, attn_drop=0., proj_drop=0.1, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.aggre = aggre
        self.downscale = downscale
        self.nsample = k[0]

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # local
        self.l_norm1 = norm_layer(dim)
        self.l_attn = LGA(dim, num_heads, nsample=k[0], attn_drop=attn_drop, proj_drop=proj_drop)
        self.l_norm2 = norm_layer(dim)
        self.l_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        # global
        self.g_norm1 = norm_layer(dim)
        self.g_attn = LGP(dim, num_heads, nsample=k[1], attn_drop=attn_drop, proj_drop=proj_drop)
        self.g_norm2 = norm_layer(dim)
        self.g_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        # global-local
        self.c_norm1 = norm_layer(dim)
        self.c_norm2 = norm_layer(dim)
        self.c_attn = LGA_C(dim, num_heads, nsample=k[2], attn_drop=attn_drop, proj_drop=proj_drop)
        self.c_norm3 = norm_layer(dim)
        self.c_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def feats_aggre(self, feats):
        """
        feats: (m, nsample, c)
        """
        if self.aggre == 'max':
            patch_feats, _ = torch.max(feats, dim=1)
        elif self.aggre == 'mean':
            patch_feats = torch.mean(feats, dim=1)
        return patch_feats # (m, c)

    def forward(self, pxo, pko):
        p, x, o = pxo # (n, 3), (n, c), (b)
        n_p, knn_idx, n_o = pko

        identity = x.clone()

        px, _ = QueryGroup(self.nsample, p, n_p, x, knn_idx, o, n_o, local=True)  # (m, nsample, nsample, 3) (m, nsample, c)
        p_r, x  = px

        l_feat = x
        x = self.l_norm1(x)
        x = self.l_attn(x, p_r) # (m, nsample, c)
        x = l_feat + self.drop_path(x) # (m, nsample, c)
        x = x + self.drop_path(self.l_mlp(self.l_norm2(x))) # (m, nsample, c)

        
        x = self.feats_aggre(x) # (m, c)
        g_feat = x
        x = self.g_norm1(x)
        x = self.g_attn([n_p, x, n_o]) # (m, c)
        x = g_feat + self.drop_path(x)
        x = x + self.drop_path(self.g_mlp(self.g_norm2(x))) # (m, c)

        feat = self.c_norm1(identity) # (n, c)
        x = self.c_norm2(x)  # (m, c)
        x = self.c_attn([p, feat, o], [n_p, x, n_o]) # (n, c)
        x = identity + self.drop_path(x)
        x = x + self.drop_path(self.c_mlp(self.c_norm3(x))) # (n ,c)

        return x




class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, depth, num_heads, downsample=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, downscale=8, k=16,
                 aggre='max', mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.k = k
        self.nsample = k[0]
        self.downscale = downscale
        self.blocks = nn.ModuleList([
            TripleAttention(in_channel, num_heads, k=k, act_layer=act_layer, norm_layer=norm_layer, downscale=downscale,
                 aggre=aggre, mlp_ratio=4, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(depth)
            ])

        self.downsample = downsample(in_channel, out_channel, self.nsample) if downsample else None

    def forward(self, pxo):
        p, x, o = pxo # (n, 3), (n, c), (b)
        sample_idx, n_p, n_o = Sampling(p, o, downscale=self.downscale)  # (m,), (m, 3), (b)
        knn_idx, _ = pointops.knnquery(self.nsample, p, n_p, o, n_o) # (m, nsample)

        for i, blk in enumerate(self.blocks):
            x = blk([p, x, o], [n_p, knn_idx, n_o])

        if self.downsample:
            feats_down, xyz_down, offset_down = self.downsample([p, x, o], [n_p, knn_idx, n_o])
        else:
            feats_down, xyz_down, offset_down = None, None, None

        feats, xyz, offset = x, p, o

        return feats, xyz, offset, feats_down, xyz_down, offset_down
class SAPFormer(nn.Module):
    def __init__(self, downscale, depths, channels, up_k, num_heads=[1, 2, 4, 8], \
                 drop_path_rate=0.2, num_layers=4, concat_xyz=False,
                 num_classes=13, ratio=0.25, k=16, prev_grid_size=0.04, sigma=1.0, stem_transformer=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
        k = self.tolist(k)

        if stem_transformer:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma)
            ])
            self.layer_start = 0
        else:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(3 if not concat_xyz else 6, channels[0], prev_grid_size, sigma=sigma),
                KPConvResBlock(channels[0], channels[0], prev_grid_size, sigma=sigma)
            ])
            self.downsample = TransitionDown(channels[0], channels[1], ratio, k[0][0])
            self.layer_start = 1

        self.layers = nn.ModuleList([BasicBlock(channels[i], channels[i+1] if i < num_layers-1 else None, depths[i], num_heads[i], downscale=downscale,
            drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], downsample=DownSample if i < num_layers-1 else None, k=k[i]) for i in range(self.layer_start, num_layers)])

        self.upsamples = nn.ModuleList([Upsample(up_k, channels[i], channels[i-1]) for i in range(num_layers-1, 0, -1)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes)
        )

        self.init_weights()

    def forward(self, feats, xyz, offset, batch, neighbor_idx):

        feats_stack = []
        xyz_stack = []
        offset_stack = []

        for i, layer in enumerate(self.stem_layer):
            feats = layer(feats, xyz, batch, neighbor_idx)

        feats = feats.contiguous()

        if self.layer_start == 1:
            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)
            feats, xyz, offset = self.downsample(feats, xyz, offset)

        for i, layer in enumerate(self.layers):
            feats, xyz, offset, feats_down, xyz_down, offset_down = layer([xyz, feats, offset])

            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)

            feats = feats_down
            xyz = xyz_down
            offset = offset_down

        feats = feats_stack.pop()
        xyz = xyz_stack.pop()
        offset = offset_stack.pop()

        for i, upsample in enumerate(self.upsamples):
            feats, xyz, offset = upsample(feats, xyz, xyz_stack.pop(), offset, offset_stack.pop(), support_feats=feats_stack.pop())

        out = self.classifier(feats)

        return out

    def tolist(self, k):
        if type(k) != list:
            k = [[k]*3]*4
        return k

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)


