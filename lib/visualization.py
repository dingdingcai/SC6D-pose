import torch

def normalize_visulization(depth):
    
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().clone()
    else:
        depth = torch.tensor(depth).squeeze().clone()
        
    mask = torch.zeros_like(depth)
    mask[depth>0] = 1
    min_dep = depth[mask.bool()].min()
    max_dep = depth[mask.bool()].max()
    mean_depth = 0.5*(min_dep + max_dep)* mask
    depth = depth - mean_depth
    return depth
