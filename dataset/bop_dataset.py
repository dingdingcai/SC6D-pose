import os
import sys
import cv2
import copy
import math
import mmcv
import torch
import random
import hashlib
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from pytorch3d.transforms import euler_angles_to_matrix

CUR_FILE_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(CUR_FILE_DIR, '..'))
sys.path.append(PROJ_ROOT)

from lib import gdr_utils as gdr
from lib import data_utils as misc

from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                               Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                               Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                               GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                               DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                               MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                               Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)
COLOR_AUG_CODE = (
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=0.5)),"
        "Sometimes(0.5, CoarseDropout(p=0.2, size_percent=0.05)),"
        "Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))"
        "], random_order = False)"
        # aae
    )

class BOP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, cfg):
        self.dataset_name = dataset_name
        self.rgb_size = cfg.INPUT_IMG_SIZE
        self.mask_size = cfg.OUTPUT_MASK_SIZE
        self.data_dir = os.path.join(cfg.DATASET_ROOT, dataset_name)
        
        self.width = cfg.DATASET_CONFIG[dataset_name]['width']
        self.height = cfg.DATASET_CONFIG[dataset_name]['height']
        
        self.train_set = cfg.DATASET_CONFIG[dataset_name]['train_set']

        assert(isinstance(self.train_set, list)), 'train_set(s) must be a list' # ['train_pbr', 'train_real', ...]
        self.dataset_id2cls = cfg.DATASET_CONFIG[dataset_name]['id2cls']
        
        self.Tz_bins_num = cfg.Tz_BINS_NUM
        self.Tz_far = cfg.DATASET_CONFIG[dataset_name]['Tz_far']
        self.Tz_near = cfg.DATASET_CONFIG[dataset_name]['Tz_near']
        self.Tz_bins_index = np.arange(self.Tz_bins_num)

        self.Tz_bins_lb = self.Tz_near + (self.Tz_far - self.Tz_near) *  self.Tz_bins_index / self.Tz_bins_num      # bin lower bound
        self.Tz_bins_ub = self.Tz_near + (self.Tz_far - self.Tz_near) * (self.Tz_bins_index + 1) / self.Tz_bins_num # bin upper bound
        self.Tz_bins_val = (self.Tz_bins_lb + self.Tz_bins_ub) / 2.0  # meddle value of bin

        self.Tz_bins_lb = torch.as_tensor(self.Tz_bins_lb)
        self.Tz_bins_ub = torch.as_tensor(self.Tz_bins_ub)
        self.Tz_bins_val = torch.as_tensor(self.Tz_bins_val)

        self.num_classes = len(self.dataset_id2cls)

        self.img_format = 'BGR'
        self.mask_morph = True
        self.filter_invalid = True
        self.mask_morph_kernel_size = 3
        self.color_augmentor = eval(COLOR_AUG_CODE)
        
        self.DZI_PAD_SCALE = cfg.ZOOM_PAD_SCALE
        self.DZI_SCALE_RATIO = cfg.ZOOM_SCALE_RATIO  # wh scale
        self.DZI_SHIFT_RATIO = cfg.ZOOM_SHIFT_RATIO  # center shift
        self.Rz_rotation_aug = cfg.RZ_ROTATION_AUG
        self.CHANGE_BG_PROB = cfg.CHANGE_BG_PROB
        self.COLOR_AUG_PROB = cfg.COLOR_AUG_PROB
        self.PEmap_normalize = cfg.PEMAP_NORMALIZE

        self.TRUNCATE_FG = False
        self.BG_KEEP_ASPECT_RATIO = True
        self.NUM_BG_IMGS = 10000
        self.BG_TYPE = "VOC_table"      # VOC_table | coco | VOC | SUN2012
        self.BG_ROOT = cfg.VOC_BG_ROOT  # "datasets/coco/train2017/"

        self.use_cache = cfg.USE_CACHE
        self.cache_dir = os.path.join(CUR_FILE_DIR, ".cache")  # .cache

        hashed_file_name = hashlib.md5(("_".join(self.train_set)
            + "dataset_dicts_{}_{}_{}".format(self.dataset_name, self.data_dir, __name__)
        ).encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, 
            "dataset_dicts_{}_{}_{}.pkl".format(self.dataset_name, "_".join(self.train_set), hashed_file_name))

        self.dataset_dicts = list() # the whole dataset information
        if self.use_cache and os.path.exists(cache_path):
            logger.info("load cached dataset dicts from {}".format(cache_path))
            self.dataset_dicts = mmcv.load(cache_path)
        else:
            for img_type in self.train_set:
                image_counter = 0
                instance_counter = 0
                train_dir = os.path.join(self.data_dir, img_type)
                logger.info("preparing data from {}".format(img_type))
                for scene in sorted(os.listdir(train_dir)):
                    if not scene.startswith('00'):  #  BOP images start with '0000xx' 
                        continue
                    scene_id = int(scene)
                    scene_dir = os.path.join(train_dir, scene)
                    scene_cam_dict = mmcv.load(os.path.join(scene_dir, "scene_camera.json"))      # gt_intrinsic
                    scene_gt_pose_dict = mmcv.load(os.path.join(scene_dir, "scene_gt.json"))      # gt_poses
                    scene_gt_bbox_dict = mmcv.load(os.path.join(scene_dir, "scene_gt_info.json")) # gt_bboxes
                    for img_id_str in tqdm(scene_gt_pose_dict, postfix=f"{scene_id}"):
                        img_id_int = int(img_id_str)
                        rgb_path = os.path.join(scene_dir, "rgb/{:06d}.jpg").format(img_id_int)
                        if not os.path.exists(rgb_path):
                            rgb_path = os.path.join(scene_dir, "rgb/{:06d}.png").format(img_id_int)
                        assert os.path.exists(rgb_path), rgb_path
                        cam_K = np.array(scene_cam_dict[img_id_str]["cam_K"], dtype=np.float32).reshape(3, 3)

                        record = {
                            "dataset_name": self.dataset_name,
                            "file_name": rgb_path,
                            "image_id": img_id_int,
                            "img_type": img_type, 
                            "height": self.height,
                            "width": self.width,
                            "cam_K": cam_K,
                        }
                        view_insts = []
                        view_inst_count = dict() # count the object number per instance in a single image
                        for anno_idx, anno_dict in enumerate(scene_gt_pose_dict[img_id_str]):
                            obj_id = anno_dict["obj_id"]
                            if obj_id not in self.dataset_id2cls: # ignore the non-target objects 
                                continue
                            R = np.array(anno_dict["cam_R_m2c"], dtype="float32").reshape(3, 3)
                            t = np.array(anno_dict["cam_t_m2c"], dtype="float32") / 1000.0
                            
                            bbox_visib = scene_gt_bbox_dict[img_id_str][anno_idx]["bbox_visib"]
                            x1, y1, w, h = bbox_visib
                            if self.filter_invalid:
                                if h <= 1 or w <= 1:
                                    continue
                                
                            mask_visib_file = os.path.join(scene_dir, "mask_visib/{:06d}_{:06d}.png".format(img_id_int, anno_idx))
                            assert os.path.exists(mask_visib_file), mask_visib_file
                            mask_single = mmcv.imread(mask_visib_file, "unchanged").astype(bool).astype(np.uint8)
                            area = mask_single.sum()
                            if area < 50:  # filter out too small or nearly invisible instances
                                continue
                            if self.mask_morph:
                                kernel = np.ones((self.mask_morph_kernel_size, self.mask_morph_kernel_size))
                                mask_single = cv2.morphologyEx(mask_single.astype(np.uint8), cv2.MORPH_CLOSE, kernel) # remove holes
                                mask_single = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel)  # remove outliers 

                            if obj_id not in view_inst_count:
                                view_inst_count[obj_id] = 0
                            view_inst_count[obj_id] += 1  # accumulate the object number per instance in a single image
                            inst = {
                                "R": R,
                                "t": t,
                                "obj_id": obj_id,    # 1-based label
                                "bbox": bbox_visib,  
                                "bbox_mode": BoxMode.XYWH_ABS,
                                "segmentation": misc.binary_mask_to_rle(mask_single, compressed=True),
                            }
                            view_insts.append(inst)
                        if len(view_insts) == 0:  # filter im without anno
                            continue
                        record["annotations"] = view_insts
                        record['obj_inst_count'] = view_inst_count
                        self.dataset_dicts.append(record)

                        image_counter += 1
                        instance_counter += len(view_insts)
                
                    print(img_type, ', images: ', image_counter, ', instances: ', instance_counter)
            
            if self.use_cache:
                mmcv.mkdir_or_exist(os.path.dirname(cache_path))
                mmcv.dump(self.dataset_dicts, cache_path, protocol=4)
                logger.info("Dumped dataset_dicts to {}".format(cache_path))

        self.dataset_dicts = misc.flat_dataset_dicts(self.dataset_dicts) # flatten the image-level dict to instance-level dict

    def __len__(self):
        return len(self.dataset_dicts)
    
    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)
    
    def __getitem__(self, idx):
        while True:
            data_dict = self.dataset_dicts[idx]
            batch = self.read_data(data_dict)
            if batch is None:  # skip the invaild samples
                idx = self._rand_another(idx) 
                continue
            return batch
    
    def collate_fn(self, batch):
        """
        batch: list of [data_dict]
        """
        new_batch = dict()
        for each_bat in batch:
            for key, val in each_bat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)
        for key, lst in new_batch.items():
            try:
                new_batch[key] = torch.stack(lst, dim=0)
            except:
                pass
        return new_batch
   
    def read_data(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        inst_infos = dataset_dict.pop("inst_infos")
        obj_id = inst_infos['obj_id']
        
        image_file = dataset_dict["file_name"]
        if not os.path.exists(image_file):
            return None
        
        image = mmcv.imread(image_file, 'color', self.img_format)
        ### RGB augmentation ###
        if np.random.rand() < self.COLOR_AUG_PROB:
            image = self.color_augmentor.augment_image(image)
        im_H, im_W = image.shape[:2]
        mask = misc.cocosegm2mask(inst_infos["segmentation"], im_H, im_W)

        obj_R = inst_infos['R'].astype("float32").reshape(3, 3)
        obj_t = inst_infos['t'].astype("float32").reshape(3,)
        cam_K = dataset_dict['cam_K'].astype("float32")

        ego_pose = np.eye(4)
        ego_pose[:3, 3] = obj_t
        ego_pose[:3, :3] = obj_R
        allo_pose = gdr.egocentric_to_allocentric(ego_pose) # convert (GT) egocentric pose to aollocentric pose
        allo_R = allo_pose[:3, :3]
        allo_R = allo_R.astype("float32")

        bx, by, bw, bh = inst_infos["bbox"]
        bbox_xyxy = np.array([bx, by, bx+bw, by+bh])
        bbox_center, bbox_scale = misc.aug_bbox_DZI(bbox_xyxy, im_H, im_W, 
                                                    scale_ratio=self.DZI_SCALE_RATIO, 
                                                    shift_ratio=self.DZI_SHIFT_RATIO, 
                                                    pad_scale=self.DZI_PAD_SCALE,
                                                    )  # Dynamic zoom-in see the paper GDR-Net
        #### randomly replace the background if an image contains multiple instances of the same object ####
        obj_inst_count = dataset_dict.pop('obj_inst_count')
        if (obj_inst_count[obj_id] > 2) and (np.random.rand() < self.CHANGE_BG_PROB):
            image = self.replace_bg(image.copy(), mask) # multiple instances in a ROI

        roi_mask = misc.crop_resize_by_warp_affine(
            mask, bbox_center, bbox_scale, self.mask_size, interpolation=cv2.INTER_NEAREST
        ) # HxW
        if roi_mask.sum() / self.mask_size**2  < 0.05: # visible pixell ratio not smaller than 5% of the bbox area
            return None

        roi_img = misc.crop_resize_by_warp_affine(
            image, bbox_center, bbox_scale, self.rgb_size, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1) / 255.0 # HxWx3 -> 3xHxW

        T_img2roi = misc.transform_to_local_ROIcrop(bbox_center=bbox_center, bbox_scale=bbox_scale, zoom_scale=self.rgb_size)
        roi_camK = T_img2roi.numpy() @ cam_K
        roi_PEmap = misc.generate_PEmap(im_hei=self.rgb_size, im_wid=self.rgb_size, cam_K=roi_camK) # 2xHxW

        if self.PEmap_normalize:  # normalize the PEmap by moving the principal point from the left upper corner to the crop center
            PE_normK = roi_camK.copy()
            PE_normK[:2, 2] += self.rgb_size / 2 # shift the principal points the region center
            roi_PEmap = misc.generate_PEmap(im_hei=self.rgb_size, im_wid=self.rgb_size, cam_K=PE_normK) # 2xHxW
        
        dataset_dict["roi_camK"] = torch.as_tensor(roi_camK, dtype=torch.float32).squeeze()      # 3x3
        dataset_dict["T_img2roi"] = torch.as_tensor(T_img2roi, dtype=torch.float32).squeeze()    # 3x3
        dataset_dict["roi_image"] = torch.as_tensor(roi_img, dtype=torch.float32).contiguous()   # 3xHxW
        dataset_dict["roi_mask"] = torch.as_tensor(roi_mask, dtype=torch.float32).contiguous()   # H/4xW/4 
        dataset_dict["roi_PEmap"] = torch.as_tensor(roi_PEmap, dtype=torch.float32).contiguous() # 2xHxW
        dataset_dict["obj_cls"] = torch.as_tensor(self.dataset_id2cls[obj_id], dtype=torch.int64)

        dataset_dict["roi_obj_t"] = torch.as_tensor(obj_t, dtype=torch.float32)          # object GT 3D location
        dataset_dict["roi_obj_R"] = torch.as_tensor(obj_R, dtype=torch.float32)          # object GT egocentric 3D orientation
        dataset_dict["roi_allo_R"] = torch.as_tensor(allo_R, dtype=torch.float32)        # object GT allocentric 3D orientation
        dataset_dict["bbox_scale"] = torch.as_tensor(bbox_scale, dtype=torch.float32)    # object (padded) bbox scale
        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)  # object bbox center

        roi_delta_pxpy, roi_delta_tz = misc.convert_TxTyTz_to_delta_PxPyTz(T3=obj_t, camK=cam_K, bbox_center=bbox_center, 
                                                                           bbox_scale=bbox_scale, zoom_scale=self.rgb_size)
        
        dataset_dict["roi_delta_tz"] = roi_delta_tz  # scale-invariant z-axis translation
        dataset_dict["roi_delta_pxpy"] =torch.as_tensor(roi_delta_pxpy, dtype=torch.float32)    # object GT scale-invariant projection shift delta_pxpy
        dataset_dict["roi_delta_tz_cls"] = max(0, ((roi_delta_tz - self.Tz_bins_lb) > 0).sum() - 1) # 0 <= cls <= K-1, gt Tz class label
        
        if self.Rz_rotation_aug: # rotation augmentation
            Rz_index = torch.randperm(4)[0] # 0:0˚, 1:90˚, 2:180˚, 3:270˚
            Rz_rad = torch.tensor([0.0, 0.0, math.pi * Rz_index * 0.5]) # 0˚, 90˚, 180˚, 270˚
            Rz_mat = euler_angles_to_matrix(Rz_rad, 'XYZ').type(torch.float32)

            roi_img = dataset_dict["roi_image"].clone()
            roi_mask = dataset_dict["roi_mask"].clone()

            ##### rotate the corresponding RGB, Mask, rotation, object projection
            if Rz_index == 1:
                roi_img = torch.flip(roi_img, [-2]).transpose(-1, -2)   # 90 deg
                roi_mask = torch.flip(roi_mask, [-2]).transpose(-1, -2) # 90 deg
            elif Rz_index == 2:
                roi_img = torch.flip(roi_img, [-1, -2])                 # 180 deg
                roi_mask = torch.flip(roi_mask, [-1, -2])               # 180 deg
            elif Rz_index == 3:
                roi_img = torch.flip(roi_img, [-1]).transpose(-1, -2)   # 270 deg
                roi_mask = torch.flip(roi_mask, [-1]).transpose(-1, -2) # 270 deg
            
            dataset_dict["roi_image"] = roi_img
            dataset_dict["roi_mask"] = roi_mask

            # calculate the object pose after in-plane rotation
            dataset_dict["roi_obj_R"] = Rz_mat @ dataset_dict["roi_obj_R"]
            dataset_dict["roi_allo_R"] = Rz_mat @ dataset_dict["roi_allo_R"]
            dataset_dict["roi_delta_pxpy"] =  Rz_mat[:2, :2] @ dataset_dict["roi_delta_pxpy"]

            # calculate the object location after in-plane rotation
            roi_obj_camK = dataset_dict["roi_camK"] 
            roi_homo_proj = F.pad(dataset_dict["roi_delta_pxpy"] * self.rgb_size, pad=[0, 1], value=1.0) # [s_zoom * delta_x, s_zoom * delta_y, 1.0]
            dataset_dict["roi_obj_t"] = self.rgb_size / bbox_scale * roi_delta_tz * torch.inverse(roi_obj_camK) @ roi_homo_proj # r * delta_z * inv(K_B) @ P_B

        return dataset_dict

    @misc.lazy_property
    def _bg_img_paths(self):
        bg_type = self.BG_TYPE 
        bg_root = self.BG_ROOT
        bg_num = self.NUM_BG_IMGS
       
        logger.info("get bg image paths")
        hashed_file_name = hashlib.md5(
            ("{}_{}_{}_get_bg_imgs".format(bg_root, bg_num, bg_type)).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmcv.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        logger.info("building bg imgs cache {}...".format(bg_type))
        assert osp.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        if bg_type == "coco":
            img_paths = [
                osp.join(bg_root, fn.name) for fn in os.scandir(bg_root) if ".png" in fn.name or "jpg" in fn.name
            ]
        elif bg_type == "VOC_table":  # used in original deepim
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        elif bg_type == "VOC":
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN2012":
            img_paths = [
                osp.join(bg_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        else:
            raise ValueError(f"BG_TYPE: {bg_type} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), bg_num)
        bg_img_paths = np.random.choice(img_paths, num_bg_imgs)

        mmcv.dump(bg_img_paths, cache_path)
        logger.info("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def trunc_mask(self, mask):
        # return the bool truncated mask
        mask = mask.copy().astype(np.bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        return mask

    def replace_bg(self, im, im_mask, return_mask=False, truncate_fg=False):
        # cfg = self.cfg
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]
        
        if self.BG_KEEP_ASPECT_RATIO:
            bg_img = self.get_bg_image(filename, H, W)
        else:
            bg_img = self.get_bg_image_v2(filename, H, W)

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(bool)
        if truncate_fg:
            mask = self.trunc_mask(im_mask)
        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)
        if return_mask:
            return im, mask  # bool fg mask
        else:
            return im

    def get_bg_image(self, filename, imH, imW, channel=3):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = utils.read_image(filename, format=self.img_format)
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                else:
                    bg_image_crop = bg_image
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                else:
                    bg_image_crop = bg_image
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                # logger.info(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
        bg_image_resize_0 = misc.resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
        return bg_image_resize

    def get_bg_image_v2(self, filename, imH, imW, channel=3):
        _bg_img = utils.read_image(filename, format=self.img_format)
        try:
            # randomly crop a region as background
            bw = _bg_img.shape[1]
            bh = _bg_img.shape[0]
            x1 = np.random.randint(0, int(bw / 3))
            y1 = np.random.randint(0, int(bh / 3))
            x2 = np.random.randint(int(2 * bw / 3), bw)
            y2 = np.random.randint(int(2 * bh / 3), bh)
            bg_img = cv2.resize(_bg_img[y1:y2, x1:x2], (imW, imH), interpolation=cv2.INTER_LINEAR)
        except:
            bg_img = np.zeros((imH, imW, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))
        return bg_img


if __name__ == "__main__":    
    import config as bop_cfg

    dataset_name = 'tless'
    # dataset_name = 'itodd'
    # dataset_name = 'ycbv'

    train_dataset = BOP_Dataset(dataset_name=dataset_name, cfg=bop_cfg)
    print('image_number: ', len(train_dataset))

    data_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=8, 
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            )

    for batch_data in data_loader:
        batch_rgb = batch_data['roi_image']

    for i in range(10):
        data_batch = train_dataset[i]
        t = data_batch['roi_obj_t']
        ego_R = data_batch['roi_obj_R']
        allo_R = data_batch['roi_allo_R']
        delta_tz = data_batch['roi_delta_tz']  
        delta_pxpy = data_batch['roi_delta_pxpy']
        print(t)
        print(ego_R)
        print(allo_R)
        print(delta_tz)
        print(delta_pxpy)
        print('#############################')


