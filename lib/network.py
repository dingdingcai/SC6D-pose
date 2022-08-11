import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
from torchvision import models as model_zoo
from lib import geometry3D as geo


def convrelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )


class SO3_Encoder(nn.Module):
    def __init__(self, input_dim=9, so3_emb_dim=128, num_classes=1):
        super().__init__()
        self.so3_emb_dim = so3_emb_dim
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv1d(self.input_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.encoders = [
            dict(
                output_layer=nn.Conv1d(256, self.so3_emb_dim, 1, 1),
            ) for _ in range(self.num_classes)
            ]
        # register decoder modules
        for i, decoder in enumerate(self.encoders):
            for key, val in decoder.items(): # layer0_1x1: conv, ... 
                setattr(self, f'encoder{i}_{key}', val)
        
    def forward(self, input, obj_idx):
        """
        input:
            samples of SO(3) Bx9xV
        return:
            BxV
        """
        assert len(obj_idx) == len(input)
        assert(input.dim() == 3 and input.shape[1] == self.input_dim), 'input must be BxCxN'
        
        out = []
        input = self.stem(input)
        for x, dec_idx in zip(input, obj_idx): # Bx9xS, 
            encoder = self.encoders[dec_idx]
            x = encoder['output_layer'](x[None, ...]) # 1x9xS   # obtain the corresponding batch feature and squeeze the feature dimension
            out.append(x)
        return torch.cat(out, dim=0)

    def inference(self, x, obj_idx):
        assert (obj_idx >= 0 and obj_idx < self.num_classes)
        x = self.stem(x)
        encoder = self.encoders[obj_idx]
        x = encoder['output_layer'](x) # Bx256x8x8 => Bx256x4x4     
        return x


class ResNet34_AsymUNet(nn.Module):
    def __init__(self, out_feat_dim=64, rgb_input_dim=3, n_decoders=1):
        super().__init__()
        self.out_feat_dim = out_feat_dim
        self.base_model = model_zoo.resnet34(pretrained=True)
        if rgb_input_dim != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(rgb_input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) # 3xHxW      -> 64xH/2xW/2
        else:
            self.layer0 = nn.Sequential(*list(self.base_model.children())[0:3]) # 3xHxW      -> 64xH/2xW/2     
                
        self.layer1 = nn.Sequential(*list(self.base_model.children())[4:5]) # 64xH/2xW/2   -> 64xH/2xW/2
        self.layer2 = nn.Sequential(*list(self.base_model.children())[5:6]) # 64xH/2xW/2   -> 128xH/4xW/4
        self.layer3 = nn.Sequential(*list(self.base_model.children())[6:7]) # 128xH/4xW/4  -> 256xH/8xW/8
        self.layer4 = nn.Sequential(*list(self.base_model.children())[7:8]) # 256xH/8xW/8  -> 512xH/16xW/16

        #  n_decoders
        self.decoders = [dict(
            layer2_1x1=convrelu(128, 128, kernel_size=1, stride=1),
            layer3_1x1=convrelu(256, 256, kernel_size=1, stride=1),
            layer4_1x1=convrelu(512, 512, kernel_size=1, stride=1),
            conv_up3=convrelu(256 + 512, 512, kernel_size=3, stride=1, padding=1),
            conv_up2=convrelu(128 + 512, 256, kernel_size=3, stride=1, padding=1),
            conv_last=nn.Conv2d(256, self.out_feat_dim+1, kernel_size=1, stride=1),
        ) for _ in range(n_decoders)]

        # register decoder modules
        for i, decoder in enumerate(self.decoders):
            for key, val in decoder.items():
                setattr(self, f'decoder{i}_{key}', val)

    def forward(self, input, decoder_idx):
        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # decoders
        x_out = []
        for i, dec_idx in enumerate(decoder_idx):
            decoder = self.decoders[dec_idx]
            batch_slice = slice(None) if len(decoder_idx) == 1 else slice(i, i + 1)
            stem_x = layer4[batch_slice]
            stem_x = decoder['layer4_1x1'](stem_x)   # Bx512x16x16 => Bx512x16x16
            stem_x = self.upsample(stem_x)           # Bx512x16x16 => Bx512x32x32
            layer_slice = layer3[batch_slice] # Bx256x32x32 
            layer_projection = decoder[f'layer3_1x1'](layer_slice) # Bx256x32x32 =>  # Bx256x32x32 
            stem_x = torch.cat([stem_x, layer_projection], dim=1) # Bx(512+256)x32x32 
            stem_x = decoder[f'conv_up3'](stem_x)                 # Bx(512+256)x32x32 => Bx512x32x32
            stem_x = self.upsample(stem_x)                        # Bx512x32x32 => Bx512x64x64

            layer_slice = layer2[batch_slice]                      # Bx128x64x64
            layer_projection = decoder[f'layer2_1x1'](layer_slice) # Bx128x64x64 => Bx128x64x64
            stem_x = torch.cat([stem_x, layer_projection], dim=1)  # Bx(512+128)x64x64
            stem_x = decoder[f'conv_up2'](stem_x)                  # Bx(512+128)x64x64 => Bx256x64x64
            x_out.append(decoder['conv_last'](stem_x))             # Bx256x64x64 => BxCx64x64

        x_out = torch.cat(x_out, dim=0)  # BxCx64x64

        rgb_emb = x_out[:, :-1]         # BxCx64x64
        visib_msk = x_out[:, -1:]       # Bx1x64x64
        return rgb_emb, visib_msk            


class PoseDecoder(nn.Module):
    def __init__(self, so3_emb_dim=64, rgb_emb_dim=64, Tz_bins_num=1000):
        super().__init__()
        
        self.so3_emb_dim = so3_emb_dim
        self.rgb_emb_dim = rgb_emb_dim
        self.Tz_bins_num = Tz_bins_num

        self.conv_down_x8 = nn.Sequential(
            nn.Conv2d(self.rgb_emb_dim + 1, 128, 
                        kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 3x3/1
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16_d3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3), # 7x7/3
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16_d5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=5, dilation=5),  # 11x11/5
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16_d7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=7, dilation=7), # 15x15/7
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )

        self.conv_down_x16_d2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2), # 5x5/2
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16_d4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),  # 9x9/4
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        self.conv_down_x16_d6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=6, dilation=6), # 13x13/6
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.enc_R6d = nn.Sequential(
            nn.Conv2d(128 * 8, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Flatten(1), # BxCHW
            nn.Linear(128*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.so3_emb_dim)
        )
        self.reg_Pxy = nn.Sequential(
            nn.Conv2d(128 * 8, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Flatten(1), # BxCHW
            nn.Linear(128*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.cls_Tz = nn.Sequential(
            nn.Conv2d(128 * 8, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Flatten(1), # BxCHW
            nn.Linear(128*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.Tz_bins_num)
        )
        
    def forward(self, rgb, mask):
        x = torch.cat([rgb, mask], dim=1)         # BxCx64x64, Bx2x64x64
        x = self.conv_down_x8(x)                # BxCxH/4xW/4 => Bx128xH/8xW/8
        x_x16 = self.conv_down_x16(x)           # Bx128xH/8xW/8 => Bx128xH/16xW/16
        x_pool = self.global_pool(x_x16)        # BxCxH/32xW/32 => BxCx1x1
        x_pool = x_pool.repeat(1, 1, *x_x16.shape[-2:]) # BxCx1x1 => BxCxH/16xW/16

        x_x16_d3 = self.conv_down_x16_d3(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16
        x_x16_d5 = self.conv_down_x16_d5(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16
        x_x16_d7 = self.conv_down_x16_d7(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16

        x_x16_d2 = self.conv_down_x16_d2(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16
        x_x16_d4 = self.conv_down_x16_d4(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16
        x_x16_d6 = self.conv_down_x16_d6(x_x16) # Bx128xH/8xW/8 => Bx128xH/16xW/16

        x = torch.cat([x_x16, x_x16_d2, x_x16_d3, x_x16_d4, 
            x_x16_d5, x_x16_d6, x_x16_d7, x_pool], dim=1) # Bx(128*8)xH/16/W/16

        R6d_emb = self.enc_R6d(x)                      # Bx5CxH/16xW/16 => BxCxH/16xW/16
        out_pxy = self.reg_Pxy(x)                      # Bx5CxH/16xW/16 => Bx2xH/16xW/16

        cls_tz = self.cls_Tz(x)    # Bx5CxH/16xW/16 => BxCxH/16xW/16
        
        return R6d_emb, out_pxy, cls_tz 


class SC6D_Network(nn.Module):
    """
    joint RGB and XYZ for training
    dense pixel-wise latent PnP;
    Zooming invariant Distance Discretization Light MultiHead SO3Encoder with MultiHead ResNet18_RGB_UV

    """
    def __init__(self, num_classes, 
                       so3_input_dim=9, 
                       so3_emb_dim=32, 
                       rgb_emb_dim=32,
                       Tz_bins_num=1000
                       ):
        super(SC6D_Network, self).__init__()
        self.rgb_emb_dim = rgb_emb_dim
        self.so3_emb_dim = so3_emb_dim       # the embedding dimension of 3D rotation samples
        self.num_classes = num_classes
        self.so3_input_dim = so3_input_dim   # dimention of rotation parameters (R6d:6 or Rmat:9)
        self.Tz_bins_num = Tz_bins_num
        
        self.so3_encoder = SO3_Encoder(
            so3_emb_dim=self.so3_emb_dim,
            input_dim=self.so3_input_dim,
            num_classes=self.num_classes,
        )     # Bx so3_input_dim xN => Bx so3_emb_dim xN

        self.rgb_encoder = ResNet34_AsymUNet(
            rgb_input_dim=3+2,
            n_decoders=self.num_classes, 
            out_feat_dim=self.rgb_emb_dim,  # feat + visib_mask
        )
        
        self.pose_decoder = PoseDecoder(
            rgb_emb_dim=self.rgb_emb_dim,
            so3_emb_dim=self.so3_emb_dim,
            Tz_bins_num=self.Tz_bins_num,
        )

    def forward(self, que_rgb, que_PEmap, rotation_so3, obj_idx):
        """
        inputs:
            que_rgb: object RGB image, Bx3xHxW
            que_PEmap: pixel coordinate of RGB image (positional encoding map), Bx2xHxW
            rotation_so3: 3D rotation samplings, Bx9xS
        """
        
        so3_sample_emb = self.so3_encoder(input=rotation_so3, obj_idx=obj_idx)            # Bx6xR => BxCxR 
        
        rgb_PEmap = torch.cat([que_rgb, que_PEmap], dim=1)
        rgb_emb, visib_msk = self.rgb_encoder(input=rgb_PEmap, decoder_idx=obj_idx) # BxCx64x64, Bx1x64x64
        
        img_rot_emb, delta_Pxy, cls_Tz = self.pose_decoder(rgb=rgb_emb, mask=visib_msk.sigmoid()) # BxC, Bx2, Bx1, BxK

        img_rot_emb = F.normalize(img_rot_emb, dim=1)
        so3_sample_emb = F.normalize(so3_sample_emb, dim=1)

        return visib_msk, img_rot_emb, delta_Pxy, cls_Tz, so3_sample_emb



class IPDF_Encoder(nn.Module):
    def __init__(self, input_dim=6, num_classes=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.encoders = [
            dict(
                input_layer=nn.Conv1d(self.input_dim, 256, kernel_size=1),
                hidden_layer=nn.Conv1d(256, 256, kernel_size=1),
                output_layer=nn.Conv1d(256, 1, kernel_size=1),
            ) for _ in range(self.num_classes)
            ]
        # register decoder modules
        for i, decoder in enumerate(self.encoders):
            for key, val in decoder.items(): # layer0_1x1: conv, ... 
                setattr(self, f'encoder{i}_{key}', val)
        
    def forward(self, input, obj_idx):
        """
        input:
            samples of SO(3) Bx6xV
        return:
            BxV
        """
        assert len(obj_idx) == len(input)
        assert(input.dim() == 3 and input.shape[1] == self.input_dim), 'input must be BxCxN'
        
        out = []
        for x, dec_idx in zip(input, obj_idx): # Bx6xS, 
            encoder = self.encoders[dec_idx]
            x = F.relu(encoder['input_layer'](x[None, ...])) # 1x6xS   # obtain the corresponding batch feature and squeeze the feature dimension
            x = F.relu(encoder['hidden_layer'](x)) # 1x6xS   # obtain the corresponding batch feature and squeeze the feature dimension
            x = encoder['output_layer'](x) # 1x6xS   # obtain the corresponding batch feature and squeeze the feature dimension
            out.append(x)
        return torch.cat(out, dim=0)



class SC6D_Network_IPDF(nn.Module):
    """
    Implicit-PDF based SO(3) orientation estimation

    joint RGB and XYZ for training
    dense pixel-wise latent PnP;
    Zooming invariant Distance Discretization Light MultiHead SO3Encoder with MultiHead ResNet18_RGB_UV

    """
    def __init__(self, num_classes=30, 
                       so3_input_dim=9, 
                       so3_emb_dim=32, 
                       rgb_emb_dim=32,
                       Tz_bins_num=200
                       ):
        super(SC6D_Network_IPDF, self).__init__()
        self.rgb_emb_dim = rgb_emb_dim
        self.so3_emb_dim = so3_emb_dim       # the embedding dimension of 3D rotation samples
        self.num_classes = num_classes
        self.so3_input_dim = so3_input_dim   # dimention of rotation parameters (R6d:6 or Rmat:9)
        self.Tz_bins_num = Tz_bins_num
        
        self.so3_encoder = IPDF_Encoder(
            input_dim=self.so3_input_dim + self.so3_emb_dim,
            num_classes=self.num_classes,
        )     # Bx so3_input_dim xN => Bx so3_emb_dim xN

        self.rgb_encoder = ResNet34_AsymUNet(
            rgb_input_dim=3+2,
            n_decoders=self.num_classes, 
            out_feat_dim=self.rgb_emb_dim,  # feat + visib_mask
        )
        
        self.pose_decoder = PoseDecoder(
            rgb_emb_dim=self.rgb_emb_dim,
            R6d_emb_dim=self.so3_emb_dim,
            depth_bins_num=self.Tz_bins_num,
        )


    def forward(self, que_rgb, que_PEmap, rotation_so3, obj_idx):
        """
        inputs:
            que_rgb: object RGB image, Bx3xHxW
            que_PEmap: pixel coordinate of RGB image (positional encoding map), Bx2xHxW
            rotation_so3: 3D rotation samplings, Bx9xS
        """
        
        rgb_PEmap = torch.cat([que_rgb, que_PEmap], dim=1)
        rgb_emb, visib_msk = self.rgb_encoder(input=rgb_PEmap, decoder_idx=obj_idx) # BxCx64x64, Bx1x64x64
        
        img_rot_emb, delta_Pxy, cls_Tz = self.pose_decoder(rgb=rgb_emb, mask=visib_msk.sigmoid()) # BxC, Bx2, Bx1, BxK

        img_rot_emb = F.normalize(img_rot_emb, dim=1)
        rgb_emb_rot = torch.cat([img_rot_emb[..., None].repeat(1, 1, rotation_so3.shape[-1]), rotation_so3], dim=1) # BxCxR, Bx9xR => Bx(C+9)xR

        log_prob = self.so3_encoder(input=rgb_emb_rot, obj_idx=obj_idx).squeeze(1)            # Bx6xR => BxCxR 

        return visib_msk, delta_Pxy, cls_Tz, log_prob

