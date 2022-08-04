
import os
import sys
import cv2

import copy
import mmcv
import json
import math
import torch
import numpy as np
from PIL import Image 

import torch.nn.functional as F

import functools
import numpy as np
from itertools import groupby
import pycocotools.mask as cocomask
import random
from tqdm import tqdm
from pytorch3d import io as pyt3d_io
from pytorch3d import ops as pyt3d_ops
from pytorch3d import loss as pyt3d_loss
from pytorch3d import utils as pyt3d_util
from pytorch3d import structures as pyt3d_struct
from pytorch3d import renderer as pyt3d_renderer
from pytorch3d import transforms as pyt3d_transform
from pytorch3d.transforms import euler_angles_to_matrix

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

def aug_bbox_DZI(bbox_xyxy, im_H, im_W, scale_ratio, shift_ratio, pad_scale, dyn_pad=False):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
            center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    
    scale_ratio = 1 + scale_ratio * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
    shift_ratio = shift_ratio * (2 * np.random.random_sample(2) - 1)     # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
    if dyn_pad:
        dyn_pad_scale = 1 + np.random.random_sample() * 0.5 
        scale = max(y2 - y1, x2 - x1) * scale_ratio * dyn_pad_scale
    else:
        scale = max(y2 - y1, x2 - x1) * scale_ratio * pad_scale
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def read_obj_ply_as_mesh(filename, return_bbox=False, return_diameter=False):
    assert(filename.endswith('.ply')), 'can only read .ply file'
    verts, faces = pyt3d_io.load_ply(filename) # Nx3
    features = torch.ones_like(verts) # Nx3
    tex = pyt3d_renderer.TexturesVertex(verts_features=features[None, ...])
    obj_mesh_model = pyt3d_struct.Meshes(verts=verts[None, ...], 
                                        faces=faces[None, ...], 
                                        textures=tex)

    if return_diameter or return_bbox:
        # dx, dy, dz = verts.max(dim=0).values - verts.min(dim=0).values
        # obj_diameter = (dx**2 + dy**2 + dz**2)**0.5
        mesh_extent = verts.max(dim=0).values - verts.min(dim=0).values # 3
        return obj_mesh_model, mesh_extent                        

    return obj_mesh_model

    
def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im


def flat_dataset_dicts(dataset_dicts):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    new_dicts = []
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                rec = {"inst_id": inst_id, "inst_infos": anno}
                rec.update(img_infos)
                new_dicts.append(rec)
        else:
            rec = img_infos
            new_dicts.append(rec)
    return new_dicts


def lazy_property(function):
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def binary_mask_to_rle(mask, compressed=True):
    """
    encode mask image to save storage space
    """
    mask = mask.astype(np.uint8)
    if compressed:
        rle = cocomask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
    else:
        rle = {"counts": [], "size": list(mask.shape)}
        counts = rle.get("counts")
        for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):  # noqa: E501
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
    return rle

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask


def segmToRLE(segm, h, w):
    """Convert segmentation which can be polygons, uncompressed RLE to RLE.

    :return: binary mask (numpy 2D array)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask



# @jit
def calc_emb_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )

    return ProjEmb


def visualize_data(x):
    min_val = x.min()
    max_val = (x - min_val).max() + 1e-8
    return (x - min_val)/max_val




def BOP_REF_POSE(ref_R):
    unsqueeze = False
    if not isinstance(ref_R, torch.Tensor):
        ref_R = torch.tensor(ref_R, dtype=torch.float32)
    if ref_R.dim() == 2:
        ref_R = ref_R.unsqueeze(0)
        unsqueeze = True
    assert ref_R.dim() == 3 and ref_R.shape[-1] == 3, "rotation R dim must be B x 3 x 3"
    CAM_REF_POSE = torch.tensor((
                (-1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ), dtype=torch.float32)

    XR = euler_angles_to_matrix(torch.tensor([180/180*math.pi, 0, 0]), "XYZ")
    R = (XR[None, ...] @ ref_R.clone())
    R = CAM_REF_POSE.T[None, ...] @ R @ CAM_REF_POSE[None, ...]
    if unsqueeze:
        R = R.squeeze(0)
    return R

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

def uniform_z_rotation(n, eps_degree=0):
    """
    uniformly sample N examples range from 0 to 360
    """
    assert n > 0, "sample number must be nonzero"
    eps_rad = eps_degree / 180.0 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.arange(n) + 1)/(n + 1) * math.pi * 2
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_rotation_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_rotation_matrix

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


def spatial_transform_2D(x, theta, mode='nearest', padding_mode='border', align_corners=False):    
    assert(x.dim()==3 or x.dim()==4)
    assert(theta.dim()==2 or theta.dim()==3)
    assert(theta.shape[-2]==2 and theta.shape[-1]==3), "theta must be Nx2x3"
    if x.dim() == 3:
        x = x[None, ...]
    if theta.dim() == 2:
        theta = theta[None, ...].repeat(x.size(0), 1, 1)
    
    stn_theta = theta.clone()
    stn_theta[:, :2, :2] = theta[:, :2, :2].transpose(-1, -2)
    stn_theta[:, :2, 2:3] = -(stn_theta[:, :2, :2] @ stn_theta[:, :2, 2:3])
    
    grid = F.affine_grid(stn_theta.to(x.device), x.shape, align_corners=align_corners)
    new_x = F.grid_sample(x.type(grid.dtype), grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return new_x


def aug_bbox_DZI_dynamic_translation(K, gt_T, bbox_xyxy, 
                                     depth_range=(0.2, 3.0), 
                                     im_W=720, im_H=540,
                                     scale_ratio=0.25, shift_ratio=0.25, pad_scale=1.5):
    """
    generate dynamic online 3D translation (tx,ty,tz) and the corresponding 2D coordiate image (within bbox)
    return:
        new 3D translation [tx, ty, tz]
        the corresponding cropped 2D coordinate patch or
        the corresponding bounding box information [bbox_center, bbox_scale]
    """
    cam_K = K.squeeze()
    old_T = gt_T.squeeze()
    assert(cam_K.dim() == 2 and old_T.dim() == 1)
    fx = cam_K[0, 0]
    cx = cam_K[0, 2]
    fy = cam_K[1, 1]
    cy = cam_K[1, 2]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_xyxy
    bbox_bw = bbox_x2 - bbox_x1
    bbox_bh = bbox_y2 - bbox_y1
    bbox_cx = bbox_x1 + bbox_bw * 0.5
    bbox_cy = bbox_y1 + bbox_bh * 0.5
    
    ############ generate new translation ############ 
    new_tz = random.uniform(*depth_range)       # random tz, 500mm, 3000mm
    
    w_pad_pix = bbox_bw # / 2.0
    h_pad_pix = bbox_bh # / 2.0
    lower_tx = (w_pad_pix - cx) / fx * new_tz        # [pad_pix, ], ensure bbox corner within image
    upper_tx = (im_W - cx - w_pad_pix) / fx * new_tz # [, -pad_pix]
    new_tx = random.uniform(lower_tx, upper_tx)      # [pad_pix, W-pad_pix]
    
    lower_ty = (h_pad_pix - cy) / fy * new_tz        # [pad_pix, ]
    upper_ty = (im_H - cy - h_pad_pix) / fx * new_tz # [, -pad_pix]
    new_ty = random.uniform(lower_ty, upper_ty)      # [pad_pix, H-pad_pix]
    
    new_T = torch.as_tensor([new_tx, new_ty, new_tz])
    
    ############ relate the original and new transltion ############ 
    old_projection = cam_K @ old_T
    old_proj_center = old_projection[:2] / old_projection[2] # object center in pixel (px, py)
    
    new_projection = cam_K @ new_T
    new_proj_center = new_projection[:2] / new_projection[2] # object center in pixel (px, py)
    
    ############ Dynamic zoom-in ############ 
    shift_factor = shift_ratio * (2 * np.random.random_sample(2) - 1) # shift * [-1.0, 1.0]
    scale_factor = pad_scale * (1 + scale_ratio * (2 * np.random.random_sample() - 1)) #pad*(1 + scale*[-1.0, 1.0]
    
    bbox_wh = torch.tensor([bbox_bw, bbox_bh])
    bbox_center = torch.tensor([bbox_cx, bbox_cy])
    
    relative_scale = old_T[2] / new_T[2]
    shift_distance = bbox_wh * shift_factor

    proj_shift_dist = bbox_center - old_proj_center + shift_distance
    
    old_DZI_bbox_center = old_proj_center + proj_shift_dist
    new_DZI_bbox_center = new_proj_center + proj_shift_dist * relative_scale
    
    old_DZI_bbox_scale = bbox_wh.max() * scale_factor
    new_DZI_bbox_scale = old_DZI_bbox_scale * relative_scale
    if isinstance(new_T, torch.Tensor):
        new_T = new_T.numpy()
    if isinstance(new_DZI_bbox_scale, torch.Tensor):
        new_DZI_bbox_scale = new_DZI_bbox_scale.item()
        old_DZI_bbox_scale = old_DZI_bbox_scale.item()
    if isinstance(new_DZI_bbox_center, torch.Tensor):
        new_DZI_bbox_center = new_DZI_bbox_center.numpy()
        old_DZI_bbox_center = old_DZI_bbox_center.numpy()
    
    return (new_T, 
            new_DZI_bbox_center, new_DZI_bbox_scale, 
            old_DZI_bbox_center, old_DZI_bbox_scale)



def rotate_batch(batch: torch.Tensor):  # (..., H, H) -> (4, ..., H, H)
    assert batch.shape[-1] == batch.shape[-2]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch)
    return torch.stack([
        batch,  # 0 deg
        torch.flip(batch, [-2]).transpose(-1, -2),  # 90 deg
        torch.flip(batch, [-1, -2]),                # 180 deg
        torch.flip(batch, [-1]).transpose(-1, -2),  # 270 deg
    ])  # (4, ..., H, H)


def rotate_batch_back(batch: torch.Tensor):  # (4, ..., H, H) -> (4, ..., H, H)
    assert batch.shape[0] == 4
    assert batch.shape[-1] == batch.shape[-2]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch)
    return torch.stack([
        batch[0],  # 0 deg
        torch.flip(batch[1], [-1]).transpose(-1, -2),  # -90 deg
        torch.flip(batch[2], [-1, -2]),                # -180 deg
        torch.flip(batch[3], [-2]).transpose(-1, -2),  # -270 deg
    ])  # (4, ..., H, H


def Ego_T3_to_Allo_Rc_deltaT3(T3, camK, bbox_center, bbox_scale,
                                   zoom_scale=256, normalize=False,
                              cam_ray=torch.tensor([0.0, 0.0, 1.0])):
    if not isinstance(T3, torch.Tensor):
        T3 = torch.as_tensor(T3, dtype=torch.float32)

    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)
        
    assert (T3.dim() == 1 and camK.dim() == 2)
    assert (bbox_center.dim() == 1 and bbox_scale.dim() == 0)
    
    Kt = camK @ T3   # 3x3 @ 3 => 3
    homo_proj = Kt / Kt[2]
    
    obj_pxpy = homo_proj[:2]
    delta_pxpy = (obj_pxpy - bbox_center) / bbox_scale
    

    obj_ray = (torch.inverse(camK.squeeze()) @ homo_proj[..., None]).squeeze(1)
    obj_ray /= obj_ray.norm(p=2)
    r_vec = cam_ray.cross(obj_ray)
    scalar = cam_ray.dot(obj_ray)
    
    r_mat = torch.zeros((3, 3))
    r_mat[0, 1] = -r_vec[2]
    r_mat[0, 2] =  r_vec[1]
    r_mat[1, 0] =  r_vec[2]
    r_mat[1, 2] = -r_vec[0]
    r_mat[2, 0] = -r_vec[1]
    r_mat[2, 1] =  r_vec[0]
    
    Rc = torch.eye(3) + r_mat + r_mat @ r_mat / (1 + scalar)
    if normalize:
        Rc = F.normalize(Rc)
    allo_TxtTyTz = (torch.inverse(Rc) @ T3)
    delta_Td = allo_TxtTyTz[-1] / (zoom_scale / bbox_scale)
    return Rc, delta_pxpy, delta_Td

def AlloR_deltaT3_to_EgoR_EgoT3(delta_pxpy, delta_Td, camK, bbox_center, bbox_scale,
                                   zoom_scale=256, normalize=False,
                                cam_ray=torch.tensor([0.0, 0.0, 1.0])):
    if not isinstance(delta_pxpy, torch.Tensor):
        delta_pxpy = torch.as_tensor(delta_pxpy, dtype=torch.float32)
    
    if not isinstance(delta_Td, torch.Tensor):
        delta_Td = torch.as_tensor(delta_Td, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)
        
    assert (delta_Td.dim() == 0 and delta_pxpy.dim() == 1 and camK.dim() == 2)
    assert (bbox_center.dim() == 1 and bbox_scale.dim() == 0)
    
    homo_Td = torch.zeros((3,))
    homo_Td[2] = delta_Td * (zoom_scale / bbox_scale)
    
    
    homo_proj = torch.ones((3,))
    homo_proj[:2] = delta_pxpy * bbox_scale + bbox_center
        
    obj_ray = (torch.inverse(camK.squeeze()) @ homo_proj[..., None]).squeeze(1)
    obj_ray /= obj_ray.norm(p=2)
    r_vec = cam_ray.cross(obj_ray)
    scalar = cam_ray.dot(obj_ray)
    r_mat = torch.zeros((3, 3))
    r_mat[0, 1] = -r_vec[2]
    r_mat[0, 2] =  r_vec[1]
    r_mat[1, 0] =  r_vec[2]
    r_mat[1, 2] = -r_vec[0]
    r_mat[2, 0] = -r_vec[1]
    r_mat[2, 1] =  r_vec[0]
    
    Rc = torch.eye(3) + r_mat + r_mat @ r_mat / (1 + scalar)
    if normalize:
        Rc = F.normalize(Rc)
    
    T3 = Rc @ homo_Td
    return Rc, T3


def batch_Allo_delta_RT_Ego_RT(delta_pxpy, delta_Td, camK, bbox_center, bbox_scale, zoom_scale=256, 
                                cam_ray=torch.tensor([0.0, 0.0, 1.0])):
    if not isinstance(delta_pxpy, torch.Tensor):
        delta_pxpy = torch.as_tensor(delta_pxpy, dtype=torch.float32)
    
    if not isinstance(delta_Td, torch.Tensor):
        delta_Td = torch.as_tensor(delta_Td, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)
    

    unsqueeze = False
    if delta_Td.dim() == 0:
        unsqueeze = True
        delta_Td = delta_Td[None, ...]
    if delta_pxpy.dim() == 1:
        delta_pxpy = delta_pxpy[None, ...]
    if camK.dim() == 2:
        camK = camK[None, ...]
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...]

    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None, ...]

    if cam_ray.dim() == 1:
        cam_ray = cam_ray[None, ...]

    device = delta_pxpy.device
    cam_ray = cam_ray.to(device)
    dim_B = len(delta_Td)
    homo_Td = torch.zeros((dim_B, 3)).to(device)
    homo_Td[:, -1] = delta_Td * (zoom_scale / bbox_scale) # Bx3
        
    homo_proj = torch.ones((dim_B, 3)).to(device)
    homo_proj[:, :2] = delta_pxpy * bbox_scale[..., None] + bbox_center # Bx2, Bx1 + Bx2
        
    obj_ray = (torch.inverse(camK) @ homo_proj[..., None]).squeeze(2) # Bx3x3 @ Bx3x1 => Bx3
    # obj_ray /= obj_ray.norm(p=2)
    obj_ray = F.normalize(obj_ray, dim=1, p=2) # Bx3
    r_vec = torch.cross(cam_ray.repeat(dim_B, 1), obj_ray, dim=1) # Bx3

    scalar = torch.sum(cam_ray.repeat(dim_B, 1) * obj_ray, dim=1) # B
    r_mat = torch.zeros((dim_B, 3, 3)).to(device)

    r_mat[:, 0, 1] = -r_vec[:, 2]
    r_mat[:, 0, 2] =  r_vec[:, 1]
    r_mat[:, 1, 0] =  r_vec[:, 2]
    r_mat[:, 1, 2] = -r_vec[:, 0]
    r_mat[:, 2, 0] = -r_vec[:, 1]
    r_mat[:, 2, 1] =  r_vec[:, 0]

    norm_r_mat2 = r_mat @ r_mat / (1 + scalar[..., None, None].repeat(1, 3, 3).to(device))  # Bx3x3
    Rc = torch.eye(3)[None, ...].to(device).repeat(dim_B, 1, 1) + r_mat + norm_r_mat2

    T3 = (Rc @ homo_Td[..., None]).squeeze(dim=-1)

    return Rc, T3


def camK_to_boxK(cam_K, bbox_scale, bbox_center, zoom_scale=256):
    if not isinstance(cam_K, torch.Tensor):
        cam_K = torch.as_tensor(cam_K, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    bs = bbox_scale.squeeze()
    bx, by = bbox_center.squeeze()
    r = zoom_scale / bs
    
    S_2d = torch.tensor(([[r, 0, 0], 
                          [0, r, 0], 
                          [0, 0, 1]]))  # scaling matrix
    T_2d = torch.tensor(([[1, 0, bs/2 - bx], 
                          [0, 1, bs/2 - by],
                          [0, 0, 1]]))  # shifting matrix
    T_C2B = S_2d @ T_2d
    
    box_K = T_C2B.float() @ torch.as_tensor(cam_K).float()
    
    return box_K.float()


def convert_TxTyTz_to_delta_PxPyTz(T3, camK, bbox_center, bbox_scale, zoom_scale):
    """
    convert absolute 3D location to SITE (scale-invariant-translation-estimation)
    """
    if not isinstance(T3, torch.Tensor):
        T3 = torch.as_tensor(T3, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if T3.dim() == 1:
        unsqueeze = True
        T3 = T3[None, ...]
    if camK.dim() == 2:
        camK = camK[None, ...]
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...]

    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None, ...]

    assert (bbox_center.dim() == 2 and len(T3) == len(bbox_center))
    assert (T3.dim() == 2 and camK.dim() == 3 and len(T3) == len(camK))
    assert (bbox_scale.dim() == 1)

    Kt = (camK @ T3.unsqueeze(2)).squeeze(2) # 1x3x3 @ 1x3x1 => 1x3
    obj_pxpy = Kt[:, :2] / Kt[:, 2]

    delta_pxpy = (obj_pxpy - bbox_center) / bbox_scale
    delta_tz = T3[:, -1] / zoom_scale * bbox_scale

    if unsqueeze:
        delta_tz = delta_tz.squeeze(0)
        delta_pxpy = delta_pxpy.squeeze(0)
        
    return delta_pxpy, delta_tz


def recover_TxTyTz_from_delta_PxPyTz(delta_pxpy, delta_tz, camK, bbox_center, bbox_scale, zoom_scale):
    if not isinstance(delta_tz, torch.Tensor):
        delta_tz = torch.as_tensor(delta_tz, dtype=torch.float32)
    
    if not isinstance(delta_pxpy, torch.Tensor):
        delta_pxpy = torch.as_tensor(delta_pxpy, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if delta_tz.dim() == 0:
        unsqueeze = True
        delta_tz = delta_tz[None, ...]

    if delta_pxpy.dim() == 1:
        delta_pxpy = delta_pxpy[None, ...]
    
    if camK.dim() == 2:
        camK = camK[None, ...]
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...]

    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None, ...]

    Tz = delta_tz * zoom_scale / bbox_scale # V

    obj_pxpy = delta_pxpy * bbox_scale[..., None] + bbox_center # Vx2, Vx1, Vx2 => Vx2

    homo_pxpy = torch.cat([obj_pxpy, torch.ones_like(Tz)[..., None]], dim=1) # Vx2, Vx1 => Vx3
    T3 = Tz[..., None].float() * (torch.inverse(camK).float() @ homo_pxpy[..., None].float()).squeeze(dim=2) # Vx1 [Vx3x3, Vx3x1 => Vx3]
    
    if unsqueeze:
        T3 = T3.squeeze(0)
    
    return T3


  

def normalized_uv(im_hei, im_wid, cam_K=None):
    
    if cam_K is not None:
        if not isinstance(cam_K, torch.Tensor):
            cam_K = torch.as_tensor(cam_K)
        K_inv = torch.inverse(cam_K)
        try:
            yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid), indexing='ij')
        except:
            yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid))

        homo_uv = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).type(torch.float32)
        homo_uv_norm = (K_inv @ homo_uv.view(3, -1)).view(3, im_hei, im_wid) # 3xHxW
        homo_uv = homo_uv_norm[:2]# 2xHxW
    else:
        try:
            yy, xx = torch.meshgrid(torch.linspace(0, 1, im_hei),torch.linspace(0, 1, im_wid), indexing='ij')
        except:
            yy, xx = torch.meshgrid(torch.linspace(0, 1, im_hei),torch.linspace(0, 1, im_wid))

        homo_uv = torch.stack([xx, yy], dim=0).type(torch.float32) # 2xHxW

    return homo_uv


###### ROI-centric transformation #####



def transform_to_local_ROIcrop(bbox_center, bbox_scale, zoom_scale=256):
    """
    transformation from original image to the object-centric crop region
    """
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...] # Nx2
        bbox_scale = bbox_scale[None, ...]   # N
        unsqueeze = True
    assert(len(bbox_center) == len(bbox_scale))

    Ts_B2X = list()
    for bxby, bs in zip(bbox_center, bbox_scale):
        r = zoom_scale / bs
        T_b2x = torch.tensor([
            [r, 0, -r * bxby[0]], 
            [0, r, -r * bxby[1]], 
            [0, 0, 1]]) 
        Ts_B2X.append(T_b2x)
    Ts_B2X = torch.stack(Ts_B2X, dim=0)
    if unsqueeze:
        Ts_B2X = Ts_B2X.squeeze(0)    
    return Ts_B2X





