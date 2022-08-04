import os
import sys
import time
import argparse
import torch
import shutil
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

file_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(proj_root)

from lib import focal_loss as FL
from lib import warmup_lr
import lib.geometry3D as geo
from lib.data_utils import *
from lib.visualization import *
from lib import bop_network as network

from torch.utils.tensorboard import SummaryWriter

# from lib import bop_config as bop_cfg
# from lib.bop_config import SC6D_DATASET_CONFIG #, DATA_CACHES

from dataset import bop_dataset as dataset
from dataset import bop_config as bop_cfg
from dataset.bop_config import SC6D_DATASET_CONFIG

# from lib.reference_coordinate import BOP_REF_POSE, POSE_TO_BOP

import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.data.distributed import DistributedSampler

"""
This task is for:
    trained with PBR, primesense, render_reconst
    
"""

def tb_figure(img, normalize=False):
    plt_fig = plt.figure()        
    if normalize:
        plt.imshow(normalize_visulization(img))
    else:
        plt.imshow(img)
    plt.close()
    return plt_fig

def train(gpu, args):
    args.gpu = gpu
    local_rank = args.local_rank * args.ngpus + gpu

    # local_rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus + gpu
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=local_rank
    )
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    batch_size = args.batchsize
    max_epochs = args.epochs

    dataset_name = args.dataset_name    
    assert (dataset_name in SC6D_DATASET_CONFIG)
    dataset_cfg = SC6D_DATASET_CONFIG[dataset_name]

    # cache_file = dataset_cfg.get('cache_file', None)      # fetch the corresponding cached file
    # cache_prefix = dataset_cfg.get('cache_prefix', None)  # fetch the corresponding cached file

    train_dataset = dataset.BOP_Dataset(data_dir=args.rgb_dir,
                                        dataset_name=dataset_name,
                                        rgb_size=bop_cfg.in_rgb_size,
                                        mask_size=bop_cfg.out_mask_size,
                                        use_cache=bop_cfg.use_cache,
                                        change_bg_prob=bop_cfg.CHANGE_BG_PROB,
                                        Rz_rotation_aug=bop_cfg.uniform_depth_sampling,
                                        uniform_depth_sampling=bop_cfg.uniform_depth_sampling,
                                        depth_bins_num=bop_cfg.depth_bins_num,
                                        img_width=dataset_cfg['width'],
                                        img_height=dataset_cfg['height'],
                                        depth_far=dataset_cfg['dep_far'],
                                        depth_near=dataset_cfg['dep_near'],
                                        dataset_type=dataset_cfg['type'],
                                        dataset_id2cls=dataset_cfg['id2cls'],
                                        # cache_file=cache_file,
                                        # cache_prefix=cache_prefix,
    )
    depth_bin_lb = train_dataset.depth_bins_lb            # K
    depth_bin_distribution = train_dataset.depth_bins_val # K

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=local_rank)
    data_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=(train_sampler is None),
            num_workers=8, 
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=train_dataset.collate_fn,
            )

    total_batches = len(data_loader) * max_epochs  # total batches for all epochs
    
    model_net = network.SC6D_Network(num_classes=train_dataset.num_classes, 
                                    so3_emb_dim=bop_cfg.so3_emb_dim, 
                                    depth_bins_num=bop_cfg.depth_bins_num,
                                    rgb_emb_dim=bop_cfg.rgb_emb_dim,
                                    so3_input_dim=bop_cfg.so3_input_dim,
                                    )
    
    model_net = model_net.cuda(gpu)
    model_net = torch.nn.parallel.DistributedDataParallel(model_net, device_ids=[args.gpu], find_unused_parameters=True)

    if local_rank == 0:
        checkpoints = os.path.join('{}/checkpoints'.format(dataset_name))
        if not os.path.exists(checkpoints):
            os.makedirs(checkpoints)
        
        tb_dir = os.path.join(checkpoints, 'tb')
        tb_old = tb_dir.replace('tb', 'tb_old')
        if os.path.exists(tb_old):
            shutil.rmtree(tb_old)
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        shutil.move(tb_dir, tb_old)
        tb_writer = SummaryWriter(tb_dir)

    optimizer = optim.AdamW(model_net.parameters(), lr=bop_cfg.start_lr, weight_decay=bop_cfg.weight_decay)

    lr_scheduler = warmup_lr.CosineAnnealingWarmupRestarts(optimizer, 
                                                            total_batches,
                                                            max_lr=bop_cfg.start_lr, 
                                                            min_lr=bop_cfg.ending_lr, 
                                                            warmup_steps=bop_cfg.warmup_steps)
    mask_BCE = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    focal_loss = FL.FocalLoss(alpha=bop_cfg.focal_alpha, gamma=bop_cfg.focal_gamma).cuda(gpu)

    iter_step = 0
    TB_SKIP_STEPS = 0
    step_interval = 100
    ckpt_epoch_interval = 15
    min_delta_Tz = 1.0
    max_delta_Tz = 0.0

    model_net.train()
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()

    rot_losses = list()
    pxy_losses = list()
    mask_losses = list()
    depth_losses = list()
    total_losses = list()

    if local_rank == 0:
        print('gpu:', gpu)
        print('local_rank:', local_rank)
        print('loading data from ' + args.rgb_dir)
        print('images:{}, batches:{}'.format(len(train_dataset), len(data_loader)))
        print('start training ...')

    for epoch in range(max_epochs):
        for i, batch_data in enumerate(data_loader):
            iter_step += 1
            batch_start = time.time()
            if iter_step > 2000:
                step_interval = 200

            #### images ###
            roi_uv = batch_data["roi_uv"].cuda(non_blocking=True)     # Bx2x64x64
            roi_rgb = batch_data["roi_image"].cuda(non_blocking=True) # Bx3x256x256
            roi_mask = batch_data["roi_mask"].cuda(non_blocking=True).unsqueeze(1) # Bx1x64x64

            #### labels ###
            roi_allo_R = batch_data['roi_allo_R'].cuda(non_blocking=True)          # Bx3x3
            roi_delta_pxy = batch_data["roi_delta_pxpy"].cuda(non_blocking=True)   # Bx2
            roi_depth_cls = batch_data['roi_delta_tz_cls'].cuda(non_blocking=True) # B

            roi_obj_cls = batch_data['obj_cls'].cuda(non_blocking=True)   # B

            min_delta_Tz = min(min_delta_Tz, batch_data["roi_delta_tz"].min())
            max_delta_Tz = max(max_delta_Tz, batch_data["roi_delta_tz"].max())

            # ### constructing the negative training rotation samples
            knn_Rmat = geo.random_xy_rotation(bop_cfg.knn_sampling_num, 
                                              eps_degree=bop_cfg.knn_Rz_range, 
                                              rang_degree=bop_cfg.knn_sampling_range).cuda(non_blocking=True)    # Kx3x3
            uni_Rmat = geo.evenly_distributed_rotation(bop_cfg.uni_sampling_num).cuda(non_blocking=True)  # Ux3x3
            
            full_Rmat = uni_Rmat.unsqueeze(0) @ roi_allo_R.unsqueeze(1) # 1xUx3x3, Bx1x3x3 => BxUx3x3
            part_Rmat = knn_Rmat.unsqueeze(0) @ roi_allo_R.unsqueeze(1) # 1xKx3x3, Bx1x3x3 => BxKx3x3
            neg_Rmat_samples = torch.cat([part_Rmat, full_Rmat], dim=1) # Bx(K+U)x3x3

            roi_R6d = matrix_to_rotation_6d(roi_allo_R)     # Bx6
            roi_R6d += (torch.randn_like(roi_R6d) * bop_cfg.R6d_std) # adding noise
            roi_Rmat = rotation_6d_to_matrix(roi_R6d)      # Bx3x3

            so3_samples = torch.cat([roi_Rmat.unsqueeze(1), neg_Rmat_samples], dim=1) # BxQx3x3
            so3_samples = so3_samples.flatten(2).permute(0, 2, 1) # Bx9xQ
                        
            # with torch.cuda.amp.autocast():            
            (pred_rgb_mask,   # Bx1x64x64
                out_rgb_emb,     # BxC
                pred_delta_pxy,  # Bx2
                pred_depth_cls,  # BxK
                out_so3_emb      # BxCxQ
            ) = model_net(que_rgb=roi_rgb, que_uv=roi_uv, rotation_so3=so3_samples, obj_idx=roi_obj_cls)

            pos_rot_egy = torch.sum(out_rgb_emb * out_so3_emb[:, :, 0], dim=1) / bop_cfg.taus  # BxC @ BxC -> B
            sum_rot_egy = torch.logsumexp(
                torch.sum(out_rgb_emb.unsqueeze(2) * out_so3_emb, # BxCx1 @ BxCxQ => BxQ
            dim=1) / bop_cfg.taus, dim=1)                        # BxQ => B
                            
            msk_loss = mask_BCE(pred_rgb_mask, roi_mask)
            rot_loss = (sum_rot_egy - pos_rot_egy).mean()
            pxy_loss = (pred_delta_pxy - roi_delta_pxy).abs().mean() # Bx2, Bx2 
            depth_loss = focal_loss(pred_depth_cls, roi_depth_cls)

            loss = (rot_loss * bop_cfg.rot_w 
                    + msk_loss * bop_cfg.msk_w 
                    + pxy_loss * bop_cfg.pxy_w 
                    + depth_loss * bop_cfg.dep_w
            )
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # lr_scheduler.step() 
            # optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            lr_scheduler.step() 
            optimizer.zero_grad()

            if local_rank == 0:
                rot_losses.append(rot_loss.item())
                pxy_losses.append(pxy_loss.item())
                mask_losses.append(msk_loss.item())
                depth_losses.append(depth_loss.item())
                if iter_step > TB_SKIP_STEPS:
                    tb_writer.add_scalar("0_Loss/1_rot_loss", rot_losses[-1], iter_step)
                    tb_writer.add_scalar("0_Loss/2_pxy_loss", pxy_losses[-1], iter_step)
                    tb_writer.add_scalar("0_Loss/3_depth_loss", depth_losses[-1], iter_step)
                    tb_writer.add_scalar("0_Loss/4_mask_loss", mask_losses[-1], iter_step)
                    tb_writer.add_scalar("Other/lr", optimizer.param_groups[0]['lr'], iter_step)

                if (iter_step > 5 and iter_step < 50 and iter_step % 5 == 0) or (iter_step % step_interval) == 0:
                    tb_vis_idx = 0
                    curr_lr = optimizer.param_groups[0]['lr']                    
                    time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
                    logging_str = "[{}/{:.1f}k/{}K], msk:{:.4f}, pxy:{:.3f}, rot:{:.3f}, dep:{:.3f}".format(
                                    epoch+1, iter_step/1000, total_batches//1000, 
                                    np.mean(mask_losses[-2000:]) * bop_cfg.msk_w, 
                                    np.mean(pxy_losses[-2000:]) * bop_cfg.pxy_w, 
                                    np.mean(rot_losses[-2000:]) * bop_cfg.rot_w, 
                                    np.mean(depth_losses[-2000:]) * bop_cfg.dep_w, 
                    )
                    logging_str +=  ', lr:{:.6f}, {}'.format(curr_lr, time_stamp)
                    logging_str += ', min_tz:{:.3f}, max_tz:{:.3f}'.format(min_delta_Tz.item(), max_delta_Tz.item())
                    print(logging_str)
                                        
        if local_rank == 0:
            if ((epoch + 1) % ckpt_epoch_interval == 0) or (epoch + 1 == max_epochs):
                time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())
                ckpt_name = 'epoch{}_R{:.3f}_P{:.4f}_M{:.4f}_D{:.4f}'.format(
                            epoch + 1, 
                            np.mean(rot_losses[-2000:]), 
                            np.mean(pxy_losses[-2000:]), 
                            np.mean(mask_losses[-2000:]),
                            np.mean(depth_losses[-2000:])
                )
                ckpt_name += '_{}.pth'.format(time_stamp)  
                ckpt_file = os.path.join(checkpoints, ckpt_name)                
                torch.save(model_net.module.state_dict(), ckpt_file)
                print('saving ', ckpt_file)


if __name__ == "__main__":
    
    print('start running...')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    
    parser.add_argument('-g', '--ngpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('--local_rank', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    
    parser.add_argument('--ip_adress', default=9999, type=str, required=True,
                        help='ip address of the host node')
    
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint to restore")

    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--rgb_dir', type=str, required=True,
                        help='decomprssed training data path')
    parser.add_argument('--port', type=int, required=True,
                        help='master port')

    parser.add_argument('--dataset_name',default='tless', type=str, required=True,
                        help='BOP detaset name')

    parser.add_argument('--batchsize', default=64, type=int, 
                        help='the batchsize in each gpu')
    
    args = parser.parse_args()

    print('loading data from ', args.rgb_dir)
    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    
    print("ip_adress is ", args.ip_adress)
    print('total GPUs: ', args.world_size)
    print('total epochs: ', args.epochs)
    print('batchsize per GPU: ', args.batchsize)

    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    mp.spawn(train, nprocs=args.ngpus, args=(args,))

    
