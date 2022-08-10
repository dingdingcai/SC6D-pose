import os
import sys
import time
import torch
import shutil
import random
import numpy as np
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp

file_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(proj_root)

import config as bop_cfg
import lib.geometry3D as geo
from lib import focal_loss as FL
from lib import network, warmup_lr
from dataset import bop_dataset as dataset

from torch.utils.tensorboard import SummaryWriter

def train(gpu, args):
    args.gpu = gpu
    torch.cuda.set_device(gpu)
    random.seed(bop_cfg.RANDOM_SEED)
    np.random.seed(bop_cfg.RANDOM_SEED)
    torch.manual_seed(bop_cfg.RANDOM_SEED)

    local_rank = args.local_rank * args.ngpus + gpu  #### comment this if running on local machines
    # local_rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus + gpu ## uncomment if running on remote servers

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=local_rank)

    max_epochs = args.epochs
    batch_size = args.batchsize
    dataset_name = args.dataset_name
    assert (dataset_name in bop_cfg.DATASET_CONFIG), '{} is not valid dataset name.'.format(dataset_name)
    train_dataset = dataset.BOP_Dataset(dataset_name=dataset_name, cfg=bop_cfg)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                    num_replicas=args.world_size, 
                                                                    rank=local_rank)
    data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=(train_sampler is None),
                                                num_workers=8, 
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                collate_fn=train_dataset.collate_fn)

    total_batches = len(data_loader) * max_epochs  # total batches for all epochs
    
    model_net = network.SC6D_Network(num_classes=train_dataset.num_classes, 
                                    so3_input_dim=bop_cfg.SO3_INPUT_DIM,
                                    so3_emb_dim=bop_cfg.SO3_EMB_DIM, 
                                    Tz_bins_num=bop_cfg.Tz_BINS_NUM,
                                    rgb_emb_dim=bop_cfg.RGB_EMB_DIM)
    
    model_net = model_net.cuda(gpu)
    model_net = torch.nn.parallel.DistributedDataParallel(model_net, device_ids=[args.gpu], find_unused_parameters=True)

    if local_rank == 0:
        checkpoints = os.path.join('checkpoints2/{}'.format(dataset_name))
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

    optimizer = optim.AdamW(model_net.parameters(), lr=bop_cfg.START_LR, weight_decay=bop_cfg.DECAY_WEIGHT)
    lr_scheduler = warmup_lr.CosineAnnealingWarmupRestarts(optimizer, 
                                                            total_batches,
                                                            max_lr=bop_cfg.START_LR, 
                                                            min_lr=bop_cfg.END_LR)
    mask_BCE = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    focal_loss = FL.FocalLoss(alpha=bop_cfg.FOCAL_ALPHA, gamma=bop_cfg.FOCAL_GAMMA).cuda(gpu)

    iter_step = 0
    TB_SKIP_STEPS = 0
    step_interval = 100
    min_delta_Tz = 1.0
    max_delta_Tz = 0.0
    ckpt_epoch_interval = 15

    Tz_losses = list()
    Rot_losses = list()
    Pxy_losses = list()
    Msk_losses = list()
    total_losses = list()

    if local_rank == 0:
        print('gpu:', gpu)
        print('local_rank:', local_rank)
        print('images:{}, batches:{}'.format(len(train_dataset), len(data_loader)))
        print('start training ...')
    
    model_net.train()
    optimizer.zero_grad()
    for epoch in range(max_epochs):
        for i, batch_data in enumerate(data_loader):
            iter_step += 1
            if iter_step > 2000:
                step_interval = 200
            min_delta_Tz = min(min_delta_Tz, batch_data["roi_delta_tz"].min())
            max_delta_Tz = max(max_delta_Tz, batch_data["roi_delta_tz"].max())
            #### images ###
            roi_PEmap = batch_data["roi_PEmap"].cuda(non_blocking=True)     # Bx2x64x64
            roi_rgb = batch_data["roi_image"].cuda(non_blocking=True) # Bx3x256x256
            roi_mask = batch_data["roi_mask"].cuda(non_blocking=True).unsqueeze(1) # Bx1x64x64

            #### labels ###
            roi_obj_cls = batch_data['obj_cls'].cuda(non_blocking=True)   # B
            roi_allo_R = batch_data['roi_allo_R'].cuda(non_blocking=True)          # Bx3x3
            roi_delta_pxy = batch_data["roi_delta_pxpy"].cuda(non_blocking=True)   # Bx2
            roi_delta_tz = batch_data['roi_delta_tz_cls'].cuda(non_blocking=True) # B

            uni_Rmat = geo.evenly_distributed_rotation(bop_cfg.SO3_TRAINING_SAMPLINGS).cuda(non_blocking=True)  # Ux3x3
            neg_Rmat_samples = uni_Rmat.unsqueeze(0) @ roi_allo_R.unsqueeze(1) # 1xUx3x3, Bx1x3x3 => BxUx3x3

            so3_samples = torch.cat([roi_allo_R.unsqueeze(1), neg_Rmat_samples], dim=1) # BxQx3x3
            so3_samples = so3_samples.flatten(2).permute(0, 2, 1) # Bx9xQ
                        
            (pred_rgb_mask,   # Bx1x64x64
            out_rgb_emb,     # BxC
            pred_delta_pxy,  # Bx2
            pred_delta_tz,  # BxK
            out_so3_emb      # BxCxQ
            ) = model_net(que_rgb=roi_rgb, que_PEmap=roi_PEmap, rotation_so3=so3_samples, obj_idx=roi_obj_cls)

            pos_rot_egy = torch.sum(out_rgb_emb * out_so3_emb[:, :, 0], dim=1) / bop_cfg.LOSS_NCE_TAUS  # BxC @ BxC -> B
            sum_rot_egy = torch.logsumexp(
                torch.sum(out_rgb_emb.unsqueeze(2) * out_so3_emb, # BxCx1 @ BxCxQ => BxQ
            dim=1) / bop_cfg.LOSS_NCE_TAUS, dim=1)                        # BxQ => B
                            
            msk_loss = mask_BCE(pred_rgb_mask, roi_mask)
            rot_loss = (sum_rot_egy - pos_rot_egy).mean()
            pxy_loss = (pred_delta_pxy - roi_delta_pxy).abs().mean() # Bx2, Bx2 
            tz_loss = focal_loss(pred_delta_tz, roi_delta_tz)

            loss = (rot_loss * bop_cfg.LOSS_ROT_W 
                    + msk_loss * bop_cfg.LOSS_MSK_W 
                    + pxy_loss * bop_cfg.LOSS_PXY_W 
                    + tz_loss * bop_cfg.LOSS_TZ_W)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step() 
            optimizer.zero_grad()

            if local_rank == 0:
                Rot_losses.append(rot_loss.item())
                Pxy_losses.append(pxy_loss.item())
                Msk_losses.append(msk_loss.item())
                Tz_losses.append(tz_loss.item())
                total_losses.append(loss.item())
                if iter_step > TB_SKIP_STEPS:
                    tb_writer.add_scalar("Loss/0_total_loss", total_losses[-1], iter_step)
                    tb_writer.add_scalar("Loss/1_rot_loss", Rot_losses[-1], iter_step)
                    tb_writer.add_scalar("Loss/2_delta_pxy_loss", Pxy_losses[-1], iter_step)
                    tb_writer.add_scalar("Loss/3_delta_Tz_loss", Tz_losses[-1], iter_step)
                    tb_writer.add_scalar("Loss/4_mask_loss", Msk_losses[-1], iter_step)
                    tb_writer.add_scalar("Other/lr", optimizer.param_groups[0]['lr'], iter_step)

                if (iter_step > 5 and iter_step < 50 and iter_step % 5 == 0) or (iter_step % step_interval) == 0:
                    curr_lr = optimizer.param_groups[0]['lr']                    
                    time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
                    logging_str = "[{}/{:.1f}k/{}K], loss:{:.3f}, msk:{:.3f}, pxy:{:.3f}, rot:{:.3f}, tz:{:.3f}".format(
                                    epoch+1, iter_step/1000, total_batches//1000, 
                                    np.mean(total_losses[-2000:]),
                                    np.mean(Msk_losses[-2000:]) * bop_cfg.LOSS_MSK_W, 
                                    np.mean(Pxy_losses[-2000:]) * bop_cfg.LOSS_PXY_W, 
                                    np.mean(Rot_losses[-2000:]) * bop_cfg.LOSS_ROT_W, 
                                    np.mean(Tz_losses[-2000:]) * bop_cfg.LOSS_TZ_W, 
                    )
                    logging_str += ', lr:{:.6f}, {}'.format(curr_lr, time_stamp)
                    logging_str += ', LB_Delta_Tz: {:.3f} mm, UB_Delta_Tz: {:.3f} mm'.format(min_delta_Tz.item(), max_delta_Tz.item())
                    print(logging_str)
                                        
        if local_rank == 0:
            if ((epoch + 1) % ckpt_epoch_interval == 0) or (epoch + 1 == max_epochs):
                time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())
                ckpt_name = 'epoch{}_All{:.3f}_Rot{:.3f}_Pxy{:.4f}_Msk{:.4f}_Tz{:.4f}'.format(
                            epoch + 1, 
                            np.mean(total_losses[-2000:]),
                            np.mean(Rot_losses[-2000:]), 
                            np.mean(Pxy_losses[-2000:]), 
                            np.mean(Msk_losses[-2000:]),
                            np.mean(Tz_losses[-2000:])
                )
                ckpt_name += '_{}.pth'.format(time_stamp)  
                ckpt_file = os.path.join(checkpoints, ckpt_name)                
                torch.save(model_net.module.state_dict(), ckpt_file)
                print('saving ', ckpt_file)


if __name__ == "__main__":
    import argparse
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
    
    # parser.add_argument('--rgb_dir', type=str, required=True,
    #                     help='decomprssed training data path')
    parser.add_argument('--port', type=int, required=True,
                        help='master port')

    parser.add_argument('--dataset_name',default='tless', type=str, required=True,
                        help='BOP detaset name')

    parser.add_argument('--batchsize', default=64, type=int, 
                        help='the batchsize in each gpu')
    
    args = parser.parse_args()

    # print('loading data from ', args.rgb_dir)
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

    
