# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

def append_img_feat(img_feat_list, end_points):
    batch_size = xyz.shape[0]
    num_seed = xyz.shape[1]
    feat_list = []
    xyz_list = []
    seed_inds_list = []

    seed_inds = end_points['fp2_inds']
    xyz = end_points['fp2_xyz']
    fp2_features = end_points['fp2_features']
    semantic_cues = end_points['cls_score_feats']
    texture_cues = end_points['full_img_1d']
    calib_Rtilt = end_points['calib_Rtilt']

    for img_feat in img_feat_list:
        # Rotate with Rtilt
        img_feat_xyz_camera = torch.cat((img_feat[:,:,0:2], torch.zeros((batch_size, num_seed, 1)).cuda()), -1)
        img_feat_xyz_depth = img_feat_xyz_camera[:,:,[0,2,1]]
        img_feat_xyz_depth[:,:,2] *= -1
        img_feat_xyz_upright_depth = torch.matmul(calib_Rtilt, img_feat_xyz_depth.transpose(2,1))
        ray_angle = xyz + img_feat_xyz_upright_depth.transpose(2,1)
        ray_angle /= torch.sqrt(torch.sum(ray_angle**2, -1)+1e-6).unsqueeze(-1) # normalize the angle
        img_mask = img_feat[:,:,-1].unsqueeze(-1) # zero out the angle for those out of any 2d boxes
        ray_angle *= img_mask

        # Compute C''
        new_img_feat_xz_upright = torch.zeros((batch_size, num_seed, 2)).cuda()
        new_img_feat_xz_upright[:,:,0] = ray_angle[:,:,0]/(ray_angle[:,:,1]+1e-6) * xyz[:,:,1] - xyz[:,:,0]
        new_img_feat_xz_upright[:,:,1] = ray_angle[:,:,2]/(ray_angle[:,:,1]+1e-6) * xyz[:,:,1] - xyz[:,:,2]
        new_img_feat_xz_upright *= img_mask

        # Point features, geometric cues
        sub_feat_list = [fp2_features, new_img_feat_xz_upright.transpose(2,1), ray_angle.transpose(2,1)]

        # Semantic cues
        img_feat_sem = img_feat[:,:,4].long()
        img_feat_sem = torch.gather(semantic_cues, 1, img_feat_sem.unsqueeze(-1).repeat(1,1,semantic_cues.shape[-1])).transpose(2, 1)
        sub_feat_list.append(img_feat_sem)

        # Texture cues
        img_feat_tex = img_feat[:,:,2:4].long()
        img_feat_tex = (img_feat_tex[:,:,0] + img_feat_tex[:,:,1] * end_points['full_img_width'].unsqueeze(-1))*3
        ch0 = torch.gather(texture_cues, 1, img_feat_tex).unsqueeze(1)
        ch1 = torch.gather(texture_cues, 1, img_feat_tex+1).unsqueeze(1)
        ch2 = torch.gather(texture_cues, 1, img_feat_tex+2).unsqueeze(1)
        img_feat_tex = torch.cat((ch0,ch1,ch2), 1)
        sub_feat_list.append(img_feat_tex)

        feat = torch.cat(sub_feat_list, 1)
        feat_list.append(feat)
        xyz_list.append(xyz)
        seed_inds_list.append(seed_inds)

    features = torch.cat(feat_list, -1)
    xyz = torch.cat(xyz_list, 1)
    seed_inds = torch.cat(seed_inds_list, 1)
    return xyz, features, seed_inds


class ImageFeatureModule(nn.Module):
    def __init__(self, max_imvote_per_pixel=3):
        super().__init__()
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.vote_dims = 1+self.max_imvote_per_pixel*4

    def forward(self, end_points):
        xyz2 = torch.matmul(end_points['calib_Rtilt'].transpose(2,1), (1/(end_points['scale']**2)).unsqueeze(-1).unsqueeze(-1)*end_points['fp2_xyz'].transpose(2,1))
        xyz2 = xyz2.transpose(2,1)
        xyz2[:,:,[0,1,2]] = xyz2[:,:,[0,2,1]]
        xyz2[:,:,1] *= -1
        end_points['xyz_camera_coord'] = xyz2
        uv = torch.matmul(xyz2, end_points['calib_K'].transpose(2,1))
        uv[:,:,0] /= uv[:,:,2]
        uv[:,:,1] /= uv[:,:,2]

        u = (uv[:,:,0]-1).round()
        v = (uv[:,:,1]-1).round()
        full_img_votes_1d = end_points['full_img_votes_1d']
        idx_beg = (u.float() + v.float() * end_points['full_img_width'].unsqueeze(-1).float())*self.vote_dims
        idx_beg = idx_beg.long()
        seed_gt_votes_cnt = torch.gather(full_img_votes_1d, 1, idx_beg)
        img_feat_list = []
        batch_size = xyz2.shape[0]
        num_seed = xyz2.shape[1]
        for i in range(self.max_imvote_per_pixel):
            vote_i_0 = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4)
            vote_i_1 = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4+1)
            seed_gt_votes_i = torch.cat((vote_i_0.unsqueeze(-1), vote_i_1.unsqueeze(-1)), -1)
            seed_gt_votes_mask_i = (seed_gt_votes_cnt > i).float()

            seed_gt_votes_i *= xyz2[:,:,2].unsqueeze(-1) # NOTE: this depth may be very off since points in the 2D box may belong to the background.
            seed_gt_votes_i /= end_points['calib_K'][:,0,0].unsqueeze(-1).unsqueeze(-1)

            ins_id = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4+3).unsqueeze(-1)
            img_feat_list_i = torch.cat((seed_gt_votes_i, u.unsqueeze(-1), v.unsqueeze(-1), ins_id, seed_gt_votes_mask_i.unsqueeze(-1)), -1)
            img_feat_list.append(img_feat_list_i)

        return img_feat_list


class ImageMLPModule(nn.Module):
    def __init__(self, input_dim, image_hidden_dim=256):
        super().__init__()
        self.img_feat_conv1 = torch.nn.Conv1d(input_dim, image_hidden_dim, 1)
        self.img_feat_conv2 = torch.nn.Conv1d(image_hidden_dim, image_hidden_dim, 1)
        self.img_feat_bn1 = torch.nn.BatchNorm1d(image_hidden_dim)
        self.img_feat_bn2 = torch.nn.BatchNorm1d(image_hidden_dim)

    def forward(self, img_features):
        img_features = F.relu(self.img_feat_bn1(self.img_feat_conv1(img_features)))
        img_features = F.relu(self.img_feat_bn2(self.img_feat_conv2(img_features)))

        return img_features
