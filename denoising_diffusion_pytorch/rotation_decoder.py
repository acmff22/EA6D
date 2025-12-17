import os
import sys
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# insert the project root of this project to the python path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)

import numpy as np
from denoising_diffusion_pytorch.generalized_mean_pooling import GeM2D as GeMeanPool2d
# 模型定义部分
class RAEncoder(nn.Module):
    def __init__(self, cfg=None):
        super(RAEncoder, self).__init__()
        if cfg is None:
            cfg = dict()
        self.img_size = cfg.get('img_size', 224)   # image size
        self.batch_size = cfg.get('batch_size', 1) # batch size        
        self.temperature = cfg.get('temperature', 0.1) # softmax temperature
        self.pose_feat_dim = cfg.get('pose_feat_dim', 256)
        
        self.pos_embed = cfg.get('pos_embed', 'RoPE100') # ['cosine', 'RoPE100']
        self.dino_patch_size = cfg.get('dino_patch_size', 14)
        
        self.backbone_feat_dim = 768 #768 for 
        self.dino_block_indices = [2, 5, 8, 11] # the 3nd, 6th, 9th, 12th blocks
        #self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        self.dino_backbone = torch.hub.load('/root/autodl-tmp/models/step2/dinov2-main/', 'dinov2_vitb14', source='local',verbose=False)
        for param in self.dino_backbone.parameters():
            param.requires_grad = False
        
        self.pixelshuffle_x2 = nn.PixelShuffle(2)
        self.pixelshuffle_x7 = nn.PixelShuffle(7)
        self.dino_backbone_transition = nn.ModuleDict()
        for blk_idx in self.dino_block_indices:
            self.dino_backbone_transition[str(blk_idx)] = nn.Sequential(
                nn.Linear(self.backbone_feat_dim, self.backbone_feat_dim),
                nn.LayerNorm(self.backbone_feat_dim),
                nn.GELU(),
            )
        
        self.pose_aware_projection = nn.Sequential(
            nn.Conv2d(1200, self.pose_feat_dim, 1, 1, 0), #1200 x_rec_dinov2
            nn.GroupNorm(1, self.pose_feat_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.rotation_embedding_head = nn.Sequential(
            #nn.Conv2d(self.pose_feat_dim + 1, 128, 3, 1, 1), # Cx32x32 -> 128x32x32
            nn.Conv2d(self.pose_feat_dim , 128, 3, 1, 1), # Cx32x32 -> 128x32x32
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # 128x32x32 -> 256x16x16
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 128, 3, 1, 1), # 256x16x16 -> 128x16x16
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # 128x16x16 -> 256x8x8
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(inplace=True),

            GeMeanPool2d(1), # 128x8x8 -> 128x1x1
            nn.Flatten(1),

            nn.Linear(256, 64),
        )
        # Add the quaternion prediction layer
        self.quaternion_prediction = nn.Linear(64, 4)
        print('self.pose_feat_dim',self.pose_feat_dim)



    def extract_DINOv2_feature(self, x, return_last_dino_feat=False):
        """
        input: x: [BV, C, H, W]
        output: feat_maps: [BV, C, H/8, W/8]
        """
        dim_B, _, dim_H, dim_W = x.shape
        dim_h = dim_H // self.dino_patch_size
        dim_w = dim_W // self.dino_patch_size

        xs = list()
        #print('x',x.shape)
        x=torch.nn.functional.interpolate(x, size=(70, 70))
        #print('x',x.shape)
        
        x = self.dino_backbone.prepare_tokens_with_masks(x) #  Bx3xHxW -> Bx(1+HW)xC
        for blk_idx in range(len(self.dino_backbone.blocks)):
            x = self.dino_backbone.blocks[blk_idx](x) # Bx(1+HW)xC -> Bx(1+L)xC
            if blk_idx in self.dino_block_indices:
                new_x = x[:, 1:, :]
                new_x = self.dino_backbone_transition[str(blk_idx)](new_x) # BxLxC -> BxLxC
                xs.append(new_x)
        xs = torch.cat(xs, dim=-1) # list of [BxLxC]  -> BxLx4C
        xs = xs.view(dim_B, dim_h, dim_w, -1).permute(0, 3, 1, 2) # BxLx4C -> Bx4Cx16x16
        xs = self.pixelshuffle_x2(xs) # Bx4Cx16x16 -> BxCx32x32
        if return_last_dino_feat:
            x_norm = self.dino_backbone.norm(x)
            x_norm_clstoken = x_norm[:, 0, :] # BxC
            x_norm_patchtokens = x_norm[:, 1:, :] # BxLxC
            return xs, x_norm_clstoken, x_norm_patchtokens
        return xs

    def generate_rotation_aware_embedding_back(self, x, mask):
        """
        x: BxCxSxS
        """
        assert(x.dim() == 4), 'x: {}'.format(x.shape)
        assert(mask.dim() == 4), 'mask: {}'.format(mask.shape)
        if x.shape[-1] != mask.shape[-1]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=True)

        assert(x.shape[2:] == mask.shape[2:]), 'x: {}, mask: {}'.format(x.shape, mask.shape)

        x = self.pose_aware_projection(x)   # BxCxSxS -> BxCxSxS
        x = torch.cat([x, mask], dim=1)     # Bx(C+1)xSxS
        x = self.rotation_embedding_head(x) # BxCxSxS -> BxC
        x = F.normalize(x.float(), dim=-1, eps=1e-8) # large eps to avoid nan when using mixed precision
        
        # Predict quaternion
        quaternion = self.quaternion_prediction(x) # Bx64 -> Bx4
        return x, quaternion
    
    def generate_rotation_aware_embedding(self, x):
        """
        x: BxCxSxS
        """
        #print('x',x.shape)
        assert(x.dim() == 4), 'x: {}'.format(x.shape)
        #print('x',x.shape)
        x = self.pose_aware_projection(x)   # BxCxSxS -> BxCxSxS
        x = self.rotation_embedding_head(x) # BxCxSxS -> BxC
        x = F.normalize(x.float(), dim=-1, eps=1e-8) # large eps to avoid nan when using mixed precision
        # Predict quaternion
        quaternion = self.quaternion_prediction(x) # Bx64 -> Bx4
        return quaternion
    
    def rotation_matrix_to_quaternion(self, matrix):
        # 提取旋转矩阵部分 (2, 3, 3)
        # matrix = matrix[:, :, :, :3]
        # print("m", matrix)
        # 假设输入矩阵是 Bx3x3 的形状
        m = matrix.view(-1, 3, 3)
        t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        r = torch.sqrt(1.0 + t)
        qw = 0.5 * r
        qx = (m[:, 2, 1] - m[:, 1, 2]) / (4.0 * qw)
        qy = (m[:, 0, 2] - m[:, 2, 0]) / (4.0 * qw)
        qz = (m[:, 1, 0] - m[:, 0, 1]) / (4.0 * qw)
        return torch.stack([qw, qx, qy, qz], dim=-1)
    
    def rot2quaternion(self,rot):  
        
        rot = rot.view(-1, 3, 3)
        quat = R.from_matrix(rot.detach().clone().cpu().numpy()).as_quat()
        return torch.from_numpy(quat).to(dtype=torch.float16).cuda()   
    
    def quaternion_loss(self, pred_quat, target_quat):
        # 转换目标旋转矩阵为四元数
        #target_quat = self.rotation_matrix_to_quaternion(target_rot_matrix)
        #target_quat = self.rot2quaternion(target_rot_matrix)
        #print("pred_quat", pred_quat)
        #print("target_quat", target_quat)
        
        # 计算四元数之间的误差
        loss = F.mse_loss(pred_quat, target_quat)
        return loss
    
    def forward(self, data):
        dim_B = data.shape[0]
        img_feat=data
        dim_BVq = img_feat.shape[0]
        #print(dim_BVq)
        dim_Vq = img_feat.shape[0] // dim_B
        #print(dim_Vq)
        img_feat = img_feat.view(dim_B, -1, *img_feat.shape[1:])    # ==> BxVqxCx32x32
        #print(img_feat.shape)   
        x_feat=img_feat.flatten(0, 1)
        #print(x_feat.shape)
        quaternion = self.generate_rotation_aware_embedding(x_feat) # B(q+VqVn+Vd+Vr)xCxSxS -> B(q+VqVn+Vd+Vr)xC
    
        return quaternion
