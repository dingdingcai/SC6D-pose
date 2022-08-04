import torch
import math
import numpy as np
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix



def vec2skew(v):
    assert(v.dim() <= 2)
    squeeze = False
    if v.dim() == 1:
        v = v.unsqueeze(0)
        squeeze = True 
    N, C = v.shape
    assert(C==3)
    skew_v = torch.zeros((N, 3, 3)).to(v.device)
    skew_v[:, 1, 2] = -v[:, 0]
    skew_v[:, 0, 2] =  v[:, 1]
    skew_v[:, 0, 1] = -v[:, 2]
    skew_v -= skew_v.transpose(-1, -2)
    if squeeze:
        skew_v = skew_v.squeeze(0)
    return skew_v

def skew2vec(v):
    assert(v.dim() <= 3)
    squeeze = False
    if v.dim() == 2:
        v = v.unsqueeze(0)
        squeeze = True 
    
    tx = -v[:, 1, 2]
    ty =  v[:, 0, 2]
    tz = -v[:, 0, 1]
    T = torch.stack([tx, ty, tz], dim=1)
    if squeeze:
        T = T.squeeze(0)
    return T 
    
def EssentialMatrix(R1, R2, t1, t2, return_RT=False):
    R1.dim() == R2.dim() == 2
    t1.dim() == t2.dim() == 1
    rel_R = R2 @ R1.transpose(-1, -2)
    rel_t = t2 - rel_R @ t1
    skew_t = vec2skew(rel_t)
    E = skew_t @ rel_R
    if return_RT:
        return E, rel_R, rel_t
    return E

def decode_E(E):
    U, S, Vt = torch.linalg.svd(E)
    S_diag = torch.zeros_like(U)
    S_diag[0, 0] = (S[0] + S[1])/2.0
    S_diag[1, 1] = (S[0] + S[1])/2.0
    W = torch.tensor([[0.0, -1., 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Z = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rec_R1 = U @ W @ Vt
    if torch.linalg.det(rec_R1) < 0:
        rec_R1 = -rec_R1

    rec_R2 = U @ W.T @ Vt
    if torch.linalg.det(rec_R2) < 0:
        rec_R2 = -rec_R2
    
    rec_T = U @ W @ S_diag @ U.T
    rec_t1 = skew2vec(rec_T)
    
    return rec_R1, rec_R2, rec_t1, -rec_t1


def gen_essential_matrix(R, t):
    skew_t = vec2skew(t)
    E = skew_t @ R
    return E


def normalized_uv(im_hei, im_wid, cam_K=None):
    
    if cam_K is not None:
        if not isinstance(cam_K, torch.Tensor):
            cam_K = torch.as_tensor(cam_K)
        K_inv = torch.inverse(cam_K)
        yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid), indexing='ij')
        homo_uv = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).type(torch.float32)
        homo_uv_norm = (K_inv @ homo_uv.view(3, -1)).view(3, im_hei, im_wid) # 3xHxW
        homo_uv = homo_uv_norm[:2]# 2xHxW
    else:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, im_hei),torch.linspace(0, 1, im_wid), indexing='ij')
        homo_uv = torch.stack([xx, yy], dim=0).type(torch.float32) # 2xHxW

    return homo_uv


def random_z_rotation(n, eps_degree=0):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * math.pi # -pi, pi
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def random_xy_rotation(n, eps_degree=0, rang_degree=180):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    rang_rad = rang_degree / 180 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad  # -eps, eps

    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def random_xyz_rotation(n, eps_degree=180):
    """
    randomly sample N examples range from 0 to 360 
    """
    eps_rad = eps_degree / 180. * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -pi, pi
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -pi, pi
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps

    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def evenly_distributed_rotation(n, random_seed=None):
    """
    uniformly sample N examples on a sphere
    """
    def normalize(vector, dim: int = -1):
        return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)

    if random_seed is not None:
        torch.manual_seed(random_seed) # fix the sampling of viewpoints for reproducing evaluation

    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    points = torch.stack([
        torch.cos(theta) * torch.sin(phi), 
        torch.sin(theta) * torch.sin(phi), 
        torch.cos(phi),], dim=1)
    forward = -points
    
    down = normalize(torch.randn(n, 3), dim=1)
    right = normalize(torch.cross(down, forward))
    down = normalize(torch.cross(forward, right))
    R_mat = torch.stack([right, down, forward], dim=1)
    return R_mat



def recover_txtytz(delta_t, box_scale, box_center, cam_K, zoom_scale=256):
    """
    delta_t: Bx3
    box_scale: Bx1
    box_center: Bx2
    cam_K: Bx3x3
    """
    device = delta_t.device
    if box_center.device != device:
        box_center = box_center.to(device)
    if cam_K.device != device:
        cam_K = cam_K.to(device)
    if box_scale.device != device:
        box_scale = box_scale.to(device)
    
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)
    if box_scale.dim() == 0:
        box_scale = box_scale.unsqueeze(0)
    if box_center.dim() == 1:
        box_center = box_center.unsqueeze(0)

    if cam_K.dim() == 2:
        cam_K = cam_K.unsqueeze(0)

    assert(cam_K.dim() == 3)

    assert(delta_t.dim() == 2)
    assert(box_scale.dim() == 1)
    assert(box_center.dim() == 2)

    # delta_tx = delta_t[..., 0]
    # delta_ty = delta_t[..., 1]
    # delta_tz = delta_t[..., 2]

    # box_px = box_center[..., 0]
    # box_py = box_center[..., 1]
    
    new_bx = delta_t[..., 0] * box_scale + box_center[..., 0] # B
    new_by = delta_t[..., 1] * box_scale + box_center[..., 1] # B
    
    new_tz = delta_t[..., 2] * zoom_scale / box_scale  # B
    
    pixel_center = torch.stack([new_bx, new_by, torch.ones_like(new_bx)], dim=1).contiguous() # Bx3
    
    new_Kt = torch.inverse(cam_K).to(new_tz.dtype) @ pixel_center.unsqueeze(-1) # Bx3x1
    new_t = (new_Kt * new_tz[..., None, None]).squeeze(dim=2) # Bx3x1,  # Bx1x1 => Bx3x1->Bx3

    return new_t



