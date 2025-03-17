import torch
import torch.nn as nn
from lib.pointops2.functions import pointops




def QueryGroup(nsample, xyz, new_xyz, feats, idx, offset, new_offset, use_xyz=False, local=False):
    if idx is None:
        idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feats.shape[1]
    knn_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz = knn_xyz - new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feats[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)

    if use_xyz:
        grouped_feat = torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)

    if local:
        grouped_xyz = knn_xyz.unsqueeze(1) - knn_xyz.unsqueeze(2) # (m, nsample, nsample, 3)
        grouped_feat = [grouped_xyz, grouped_feat]

    return grouped_feat, idx


def Sampling(xyz, offset, downscale=8):
    count = int(offset[0].item()/downscale)+1
    n_offset = [count]

    for i in range(1, offset.shape[0]):
        count += ((offset[i].item() - offset[i-1].item())/downscale) + 1
        n_offset.append(count)
    n_offset = torch.tensor(n_offset, dtype=torch.int32, device=torch.device('cuda'))
    idx = pointops.furthestsampling(xyz, offset, n_offset)  # (m)
    n_xyz = xyz[idx.long(), :]  # (m, 3)
    return idx.long(), n_xyz, n_offset

def gather_neighbour(pc, neighbor_idx):
    batch_size, num_points, k = neighbor_idx.shape
    idx = neighbor_idx.reshape(batch_size, -1) 
    features = torch.gather(
        pc, 
        1, 
        idx.unsqueeze(-1).expand(-1, -1, pc.size(-1))
    )  
    return features.view(batch_size, num_points, k, -1)



def relative_pos_transforming(xyz, neighbor_xyz):

    batch_size, num_points, k, _ = neighbor_xyz.shape
    
    xyz_tile = xyz.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, K, 3]
    
    relative_xyz = xyz_tile - neighbor_xyz  # [B, N, K, 3]
    
    relative_alpha = torch.atan2(relative_xyz[...,1], relative_xyz[...,0]).unsqueeze(-1)
    relative_xydis = torch.sqrt(relative_xyz[...,:2].pow(2).sum(-1, keepdim=True))
    relative_beta = torch.atan2(relative_xyz[...,2], relative_xydis.squeeze(-1)).unsqueeze(-1)
    relative_dis = torch.sqrt(relative_xyz.pow(2).sum(-1, keepdim=True))
    
    relative_info = torch.cat([
        relative_dis,
        xyz_tile,
        neighbor_xyz
    ], dim=-1) 
    max_dis = relative_dis.amax(dim=(1,2)) 
    local_volume = max_dis.pow(3)
    
    return relative_info, relative_alpha, relative_beta, relative_dis, local_volume

def local_polar_representation(xyz, neighbor_idx):
    xyz = xyz.unsqueeze(0)  
    neighbor_idx = neighbor_idx.unsqueeze(0)  
    

    neighbor_xyz = gather_neighbour(xyz, neighbor_idx)  
    
    relative_info, rel_alpha, rel_beta,geometric_dis,local_volume = relative_pos_transforming(
        xyz, neighbor_xyz
    )
    
    neighbor_mean = neighbor_xyz.mean(dim=2) 
    direction = xyz - neighbor_mean  
    direction_tile = direction.unsqueeze(2).expand(-1, -1, neighbor_idx.size(2), -1)  
    
    dir_alpha = torch.atan2(direction_tile[...,1], direction_tile[...,0]).unsqueeze(-1)
    dir_xydis = torch.sqrt(direction_tile[...,:2].pow(2).sum(-1, keepdim=True))
    dir_beta = torch.atan2(direction_tile[...,2], dir_xydis.squeeze(-1)).unsqueeze(-1)
    
    angle_alpha = rel_alpha - dir_alpha  
    angle_beta = rel_beta - dir_beta    
    
    local_rep = torch.cat([
        angle_alpha,
        angle_beta,
        relative_info
    ], dim=-1).squeeze(0)  
    
    global_dis = torch.sqrt(xyz.squeeze(0).pow(2).sum(-1))  
    global_volume = global_dis.max().pow(3)
    vol_ratio = (local_volume / global_volume)  

    vol_ratio = vol_ratio.expand(xyz.size(0), 1)
    
    return local_rep, geometric_dis.squeeze(0), vol_ratio


