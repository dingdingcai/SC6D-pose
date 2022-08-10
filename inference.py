import os
import cv2
import sys
import time
import math
import json
import mmcv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from bop_toolkit_lib import inout
from pytorch3d.transforms import euler_angles_to_matrix

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import config as cfg
from lib import network
from lib import data_utils as misc

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--eval_finetune',  action='store_true',
                    help='evaluation with the fine-tuned model')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
DEVICE = torch.device('cuda:{}'.format(args.gpu_id))

dataset_name = args.dataset_name
assert dataset_name in cfg.DATASET_CONFIG, '{} not a valid dataset'.format(dataset_name)

prefix_str = time.strftime('%m%d-%H%M%S', time.localtime())
dataset_path = os.path.join(cfg.DATASET_ROOT, dataset_name)  # dataset_path
eval_result_dir = os.path.join(root_dir, 'evaluation/{}_BOP19_eval'.format(dataset_name))

dataset_cfg = cfg.DATASET_CONFIG[dataset_name]
DELTA_TZ_BINS_LB = dataset_cfg['Tz_near'] + (dataset_cfg['Tz_far'] - dataset_cfg['Tz_near']) * np.arange(cfg.Tz_BINS_NUM) / cfg.Tz_BINS_NUM        # bin lower bound
DELTA_TZ_BINS_UB = dataset_cfg['Tz_near'] + (dataset_cfg['Tz_far'] - dataset_cfg['Tz_near']) * (1 + np.arange(cfg.Tz_BINS_NUM)) / cfg.Tz_BINS_NUM  # bin upper bound
DELTA_TZ_BINS_VAL = torch.as_tensor(DELTA_TZ_BINS_LB + DELTA_TZ_BINS_UB) / 2.0  # meddle value of bin
OBJS_LABEL_CLASS = dataset_cfg['id2cls']

prefix_str += '-Seed{}'.format(cfg.RANDOM_SEED)
if cfg.TEST_TIME_AUGMENT:
    TEST_TIME_AUG = euler_angles_to_matrix(
        torch.tensor(
            [[0.0, 0.0, math.pi * 0.0], 
            [0.0, 0.0, math.pi * 0.5],
            [0.0, 0.0, math.pi * 1.0], 
            [0.0, 0.0, math.pi * 1.5]]
            ), 'XYZ') # 4x3x3
    prefix_str += '-TestAug'
else:
    prefix_str += '-NoAug'

######  construct SO(3) representations ######
num_Rz_samples = cfg.SO3_TESTING_Rz_SAMPLINGS   # 120:3˚, the number of in-plane roations for each viewpoint
num_Rxy_samples = cfg.SO3_TESTING_VP_SAMPLINGS  # 4000:3˚, the number of viewpoint samplings on the sphere
uniform_Rz = misc.uniform_z_rotation(num_Rz_samples)
uniform_Rxy_samples = misc.evenly_distributed_rotation(num_Rxy_samples, random_seed=cfg.RANDOM_SEED) # Sx3x3
uniform_so3mat_samples = (uniform_Rz.unsqueeze(1) @ uniform_Rxy_samples.unsqueeze(0)).view(-1, 3, 3) # Rx3x3
input_so3_samplings = uniform_so3mat_samples.flatten(1).transpose(-2, -1)  # Rx9 
print('so3 samplings: ', input_so3_samplings.shape)

model_net = network.SC6D_Network(num_classes=dataset_cfg['num_class'], 
                                so3_emb_dim=cfg.SO3_EMB_DIM, 
                                Tz_bins_num=cfg.Tz_BINS_NUM,
                                rgb_emb_dim=cfg.RGB_EMB_DIM,
                                so3_input_dim=cfg.SO3_INPUT_DIM,
                                ).to(DEVICE)

if args.eval_finetune:  # evaluation with finetuned model
    ckpt_file = os.path.join(root_dir, 'checkpoints', dataset_name, 'finetuned_synt+real_30epoch.pth')
    est_pose_file = "{}-{}x{}-synt+real_{}-test.csv".format(prefix_str, num_Rz_samples, num_Rxy_samples, dataset_name)
    det_file = os.path.join(root_dir, 'bop22_default_detections_and_segmentations', 
            'cosypose_maskrcnn_synt+real', 'challenge2022-642947_{}-test.json'.format(dataset_name))
else:
    ckpt_file = os.path.join(root_dir, 'checkpoints', dataset_name, 'pretrained_pbr_75epoch.pth')
    est_pose_file = "{}-{}x{}-pbr_{}-test.csv".format(prefix_str, num_Rz_samples, num_Rxy_samples, dataset_name)
    det_file = os.path.join(root_dir, 'bop22_default_detections_and_segmentations', 
            'cosypose_maskrcnn_pbr', 'challenge2022-524061_{}-test.json'.format(dataset_name))

print('Loading cosypose detection: ', det_file)
with open(det_file, 'r') as f:
    object_pred_dets = json.load(f)

print('loading pre-trained model from {}'.format(ckpt_file))
model_net.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
model_net.eval()

SO3_normalized_embs = dict()
with torch.no_grad():
    for obj_id, obj_cls in OBJS_LABEL_CLASS.items():
        inst_so3_emb = model_net.so3_encoder.inference(input_so3_samplings[None, ...].to(DEVICE), obj_cls) # 1x9xR => 1xCxR 
        SO3_normalized_embs[obj_cls] = F.normalize(inst_so3_emb, dim=1).detach().cpu().squeeze(0)          # 1xCxR => CxR 

image_detect_dict = dict()
for det_entry in object_pred_dets:
    view_id = det_entry['image_id']
    scene_id = det_entry['scene_id']    
    obj_id = det_entry['category_id']
    obj_conf = det_entry['score']
    x1, y1, w, h = det_entry['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
    scene_view_str = '{:06d}/{:06d}'.format(scene_id, view_id)
    if scene_view_str not in image_detect_dict:
        image_detect_dict[scene_view_str] = {
            'ids': list(),
            'bboxes': list(),
            'scores': list(),
            'labels': list()
        }
    image_detect_dict[scene_view_str]['ids'].append(obj_id)
    image_detect_dict[scene_view_str]['scores'].append(obj_conf)
    image_detect_dict[scene_view_str]['bboxes'].append(np.array([x1, y1, x2, y2]))
    image_detect_dict[scene_view_str]['labels'].append(OBJS_LABEL_CLASS[obj_id])
num_test_images = len(image_detect_dict)
num_test_instances = len(object_pred_dets)

print('Evaluation on BOP19 challenge subset: {} instances in {} images'.format(num_test_images, num_test_instances))  # 10079    
print(est_pose_file)

eval_steps = 0
obj_runtime = list()
view_runtime = list()
bop19_pose_est_results = list()
for scene_view_str, det_data in sorted(image_detect_dict.items()):
    eval_steps += 1
    scene_id_str, view_id_str = scene_view_str.split('/')
    view_id = int(view_id_str)
    scene_id = int(scene_id_str)

    scene_dir = os.path.join(dataset_path, dataset_cfg['test_set'], '{:06d}'.format(scene_id))
    scene_camK = mmcv.load(os.path.join(scene_dir, 'scene_camera.json'))
    view_rgb_file = os.path.join(scene_dir, 'rgb', '{:06d}.png'.format(view_id))
    if not os.path.exists(view_rgb_file):
        view_rgb_file = os.path.join(scene_dir, 'gray', '{:06d}.tif'.format(view_id)) # gray images in ITODD 
    
    view_cam_K = torch.as_tensor(scene_camK[str(view_id)]['cam_K'], dtype=torch.float32).view(3, 3) # 3x3
    view_image = torch.as_tensor(mmcv.imread(view_rgb_file, 'color', 'BGR'), dtype=torch.float32)   # HxWx3
    img_H, img_W = view_image.shape[:2] # HxW
    
    det_objIDs = det_data['ids']        # list: N
    det_bboxes = det_data['bboxes']     # array: Nx4
    det_scores = det_data['scores']     # array: N
    det_labels = det_data['labels']     # list: N

    inst_time = list()
    view_objs_ts = list()
    view_objs_Rs = list()
    view_objs_IDs = list()
    view_objs_scores = list()

    for inst_ix, inst_id in enumerate(det_objIDs):
        inst_timer = time.time()
        inst_score = det_scores[inst_ix]
        inst_cls = OBJS_LABEL_CLASS[inst_id]
        x1, y1, x2, y2 = det_bboxes[inst_ix]
        cx = min((x1 + x2) / 2.0, img_W)
        cy = min((y1 + y2) / 2.0, img_H)
        bw = int(max(0, min(x2 - x1, img_W)))
        bh = int(max(0, min(y2 - y1, img_H)))
        bx = int(max(0, cx - bw // 2))
        by = int(max(0, cy - bh // 2))

        box_scale = max(bw, bh) * cfg.ZOOM_PAD_SCALE         # box square size max(w, h) * pad
        box_center = torch.as_tensor([cx, cy], dtype=torch.float32)
        zooming_factor = cfg.INPUT_IMG_SIZE / box_scale

        #### transformation from RGB image X to object-centric crop B ####
        T_img2roi = misc.transform_to_local_ROIcrop(bbox_center=box_center, bbox_scale=box_scale, zoom_scale=cfg.INPUT_IMG_SIZE) # Equation (4)
        roi_camK = T_img2roi @ view_cam_K   #  Equation (5), calibrated object-centric intrinsic
        
        if cfg.PEMAP_NORMALIZE: # normalize the PEmap by moving the principal point from the left upper corner to the crop center
            PE_mapK = roi_camK.clone()
            PE_mapK[:2, 2] += cfg.INPUT_IMG_SIZE / 2 # 
            roi_PEmap = misc.generate_PEmap(im_hei=cfg.INPUT_IMG_SIZE, im_wid=cfg.INPUT_IMG_SIZE, cam_K=PE_mapK).unsqueeze(0) # 2xHxW 
        else:
            roi_PEmap = misc.generate_PEmap(im_hei=cfg.INPUT_IMG_SIZE, im_wid=cfg.INPUT_IMG_SIZE, cam_K=roi_camK).unsqueeze(0) # 2xHxW 
        
        roi_rgb = misc.crop_resize_by_warp_affine(
            view_image.numpy(), box_center.numpy(), box_scale, cfg.INPUT_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        roi_rgb = torch.as_tensor(roi_rgb.transpose(2, 0, 1)).unsqueeze(0) / 255.0 # 1x3xHxW
        if cfg.TEST_TIME_AUGMENT:
            roi_rgb = misc.rotate_batch(roi_rgb.squeeze(0)) # test-time rotations, 4x3xHxW
        
        batchsize = len(roi_rgb)
        que_rgb_feat, que_out_mask = model_net.rgb_encoder(
            torch.cat([roi_rgb, roi_PEmap.repeat(batchsize, 1, 1, 1)], dim=1).to(DEVICE),
            torch.as_tensor(inst_cls)[None, ...].repeat(batchsize, 1).to(DEVICE)) # 4xCx64x64, cx1x64x64
        
        que_rgb_emb, est_delta_Pxy, que_delta_Tz = model_net.pose_decoder(rgb=que_rgb_feat, mask=que_out_mask.sigmoid())

        que_rgb_emb = F.normalize(que_rgb_emb, dim=1) # 4xC        
        inst_so3_emb = SO3_normalized_embs[inst_cls]  # CxR
        so3_log_probs = torch.einsum('bc,ck->bk', que_rgb_emb, inst_so3_emb.to(que_rgb_emb.device))
        max_log_probs, max_inds = so3_log_probs.topk(k=1, dim=1) # 4xR => 4x1
        max_log_probs = max_log_probs.squeeze(1) # 4x1 => 4
        max_inds = max_inds.squeeze(1)           # 4x1 => 4
        est_allo_R = uniform_so3mat_samples[max_inds] # 4x3x3

        if cfg.TEST_TIME_AUGMENT:
            est_allo_R = TEST_TIME_AUG.transpose(-1, -2) @ est_allo_R # 4x3x3, rotate back
            est_delta_Pxy = (TEST_TIME_AUG[:, :2, :2].transpose(-1, -2).to(est_delta_Pxy.device) @ est_delta_Pxy[..., None]).squeeze(2)

        ######################### obtain the object projection offsets prediction #########################
        que_delta_Tz_probs = torch.softmax(que_delta_Tz, dim=1)            # 4xK => 4xK
        delta_Tz_top_conf, delta_Tz_top_idx = que_delta_Tz_probs.topk(k=1) # 4x1
        delta_Tz_top_conf = delta_Tz_top_conf.squeeze(1)  # 4x1 => 4
        delta_Tz_top_idx = delta_Tz_top_idx.squeeze(1)    # 4x1 => 4
        est_delta_Tz = torch.sum(que_delta_Tz_probs * DELTA_TZ_BINS_VAL[None, ...].to(que_delta_Tz_probs.device), dim=1) # 4xK => 4
        
        est_delta_Tz = est_delta_Tz.mean(dim=0).detach().cpu()    # averaging the predictions
        est_delta_Pxy = est_delta_Pxy.mean(dim=0).detach().cpu()  # averaging the predictions
        est_allo_R = est_allo_R[max_log_probs.argmax()].detach().cpu()
        
        #### recover the object actual location ####
        homo_delta_Pxy = F.pad(cfg.INPUT_IMG_SIZE * est_delta_Pxy, (0, 1), value=1.0) # Equation (3), offset in pixels
        est_obj_ray = torch.inverse(roi_camK) @ homo_delta_Pxy    # the estimated ray through object origin
        est_t = zooming_factor * est_delta_Tz * est_obj_ray       # recover the actual 3D location
        est_Rc = misc.rotation_from_Allo2Ego(obj_ray=est_obj_ray) # Equation (6)
        est_R = est_Rc @ est_allo_R    # recover the actual (egocentric) 3D orientation

        view_objs_ts.append(est_t)
        view_objs_Rs.append(est_R)
        view_objs_IDs.append(inst_id)
        view_objs_scores.append(inst_score)
        inst_time.append(time.time() - inst_timer)
    
    view_cost = np.sum(inst_time)
    inst_cost = np.mean(inst_time)
    view_runtime.append(view_cost)
    obj_runtime.append(inst_cost)

    for eix, obj_id in enumerate(view_objs_IDs):
        est_t = view_objs_ts[eix]
        est_R = view_objs_Rs[eix]
        det_conf = view_objs_scores[eix]
        bop19_pose_est_results.append({'time': view_cost,
                                    'scene_id': int(scene_id),
                                    'im_id': int(view_id),
                                    'obj_id': int(obj_id),
                                    'score': det_conf,
                                    'R': est_R.squeeze().numpy(),
                                    't': est_t.squeeze().numpy() * 1000.0}) # convert estimated pose to BOP format

    #####  print progress of the entire evaluation #####
    if eval_steps % cfg.LOGGING_STEPS == 0:
        time_stamp = time.strftime('%m-%d_%H:%M:%S', time.localtime())
        print('[{}/{}], img_cost: {:.1f} ms, inst_cost:{:.1f} ms, {}'.format(
            eval_steps, num_test_images,
            np.mean(view_runtime) * 1000,
            np.mean(obj_runtime) * 1000,
            time_stamp))

if not os.path.exists(eval_result_dir):
    os.makedirs(eval_result_dir)

inout.save_bop_results(os.path.join(eval_result_dir, est_pose_file), bop19_pose_est_results)
print(est_pose_file)
