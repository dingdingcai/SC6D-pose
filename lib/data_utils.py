import cv2
import math
import torch
import functools
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import pycocotools.mask as cocomask
from pytorch3d import io as pyt3d_io
from pytorch3d import structures as pyt3d_struct
from pytorch3d import renderer as pyt3d_renderer
from pytorch3d.transforms import euler_angles_to_matrix

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

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

def evenly_distributed_rotation(n, random_seed=None):
    """
    uniformly sample N examples on a sphere
    """
    def normalize(vector, dim: int = -1):
        return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)
    
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    points = torch.stack([
        torch.cos(theta) * torch.sin(phi), 
        torch.sin(theta) * torch.sin(phi), 
        torch.cos(phi),], dim=1)
    forward = -points
    
    if random_seed is not None:
        torch.manual_seed(random_seed) # fix the sampling of viewpoints for reproducing evaluation
    
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

def generate_PEmap(im_hei, im_wid, cam_K):
    if not isinstance(cam_K, torch.Tensor):
        cam_K = torch.as_tensor(cam_K)
    K_inv = torch.inverse(cam_K)
    try:
        yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid), indexing='ij')
    except:
        yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid))

    homo_uv = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).type(torch.float32)
    homo_uvk = (K_inv @ homo_uv.view(3, -1)).view(3, im_hei, im_wid) # 3xHxW
    homo_uv = homo_uvk[:2] # 2xHxW

    return homo_uv

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

def rotation_from_Allo2Ego(obj_ray, cam_ray=torch.tensor([0.0, 0.0, 1.0])):
    if not isinstance(obj_ray, torch.Tensor):
        obj_ray = torch.as_tensor(obj_ray, dtype=torch.float32)
    
    unsqueeze = False
    if obj_ray.dim() == 1:
        unsqueeze = True
        obj_ray = obj_ray[None, ...]
    
    if obj_ray.dim() == 2 and cam_ray.dim() == 1:
        cam_ray = cam_ray[None, ...].repeat(len(obj_ray), 1)

    assert(obj_ray.shape == cam_ray.shape), ' {} vs {} are mismatched.'.format(obj_ray.shape, cam_ray.shape)

    dim_B = obj_ray.shape[0]
    device = obj_ray.device
    cam_ray = cam_ray.to(device)

    obj_ray = F.normalize(obj_ray, dim=1, p=2) 
    cam_ray = F.normalize(cam_ray, dim=1, p=2) 
    r_vec = torch.cross(cam_ray.repeat(dim_B, 1), obj_ray, dim=1) 
    scalar = torch.sum(cam_ray.repeat(dim_B, 1) * obj_ray, dim=1) 
    r_mat = torch.zeros((dim_B, 3, 3)).to(device)

    r_mat[:, 0, 1] = -r_vec[:, 2]
    r_mat[:, 0, 2] =  r_vec[:, 1]
    r_mat[:, 1, 0] =  r_vec[:, 2]
    r_mat[:, 1, 2] = -r_vec[:, 0]
    r_mat[:, 2, 0] = -r_vec[:, 1]
    r_mat[:, 2, 1] =  r_vec[:, 0]

    norm_r_mat2 = r_mat @ r_mat / (1 + scalar[..., None, None].repeat(1, 3, 3).to(device))  # Bx3x3
    Rc = torch.eye(3)[None, ...].to(device).repeat(dim_B, 1, 1) + r_mat + norm_r_mat2
    if unsqueeze:
        Rc = Rc.squeeze(0)
    return Rc


