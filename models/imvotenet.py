# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" ImVoteNet for 3D object detection with RGB-D.

Author: Charles R. Qi, Xinlei Chen and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from image_feature_module import ImageFeatureModule, ImageMLPModule, append_img_feat
from dump_helper import dump_results
from loss_helper import get_loss

def sample_valid_seeds(mask, num_sampled_seed=1024):
    """
    (TODO) write doc for this function
    """
    mask = mask.cpu().detach().numpy() # B,N
    all_inds = np.arange(mask.shape[1]) # 0,1,,,,N-1
    batch_size = mask.shape[0]
    sample_inds = np.zeros((batch_size, num_sampled_seed))
    for bidx in range(batch_size):
        valid_inds = np.nonzero(mask[bidx,:])[0] # return index of non zero elements
        if len(valid_inds) < num_sampled_seed:
            assert(num_sampled_seed <= 1024)
            rand_inds = np.random.choice(list(set(np.arange(1024))-set(np.mod(valid_inds, 1024))),
                                        num_sampled_seed-len(valid_inds),
                                        replace=False)
            cur_sample_inds = np.concatenate((valid_inds, rand_inds))
        else:
            cur_sample_inds = np.random.choice(valid_inds, num_sampled_seed, replace=False)
        sample_inds[bidx,:] = cur_sample_inds
    sample_inds = torch.from_numpy(sample_inds).long()
    return sample_inds


class ImVoteNet(nn.Module):
    r"""
        ImVoteNet module.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
        max_imvote_per_pixel: (default: 3)
            Maximum number of image votes per pixel.
        image_feature_dim: (default: 18)
            Total number of dimensions for image features, geometric + semantic + texture
        image_hidden_dim: (default: 256)
            Hidden dimensions for the image based VoteNet, default same as point based VoteNet
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', 
        max_imvote_per_pixel=3, image_feature_dim=18, image_hidden_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.image_feature_dim = image_feature_dim

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Image feature extractor
        self.image_feature_extractor = ImageFeatureModule(max_imvote_per_pixel=self.max_imvote_per_pixel)
        # MLP on image features before fusing with point features
        self.image_mlp = ImageMLPModule(image_feature_dim, image_hidden_dim=image_hidden_dim)

        # Hough voting modules
        self.img_only_vgen = VotingModule(self.vote_factor, image_hidden_dim)
        self.pc_only_vgen = VotingModule(self.vote_factor, 256)
        self.pc_img_vgen = VotingModule(self.vote_factor, image_hidden_dim+256)

        # Vote aggregation and detection
        self.img_only_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, seed_feat_dim=image_hidden_dim, key_prefix='img_only_')
        self.pc_only_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, seed_feat_dim=256, key_prefix='pc_only_')
        self.pc_img_pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, seed_feat_dim=image_hidden_dim+256, key_prefix='pc_img_')

    def forward(self, inputs, joint_only=False):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)

                (TODO) write doc for this function
        Returns:
            end_points: dict
        """
        end_points = {}
        end_points.update(inputs)
        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        xyz = end_points['fp2_xyz']
        fp2_features = end_points['fp2_features']

        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = fp2_features

        img_feat_list = self.image_feature_extractor(end_points)
        assert len(img_feat_list) == self.max_imvote_per_pixel
        xyz, features, seed_inds = append_img_feat(xyz, fp2_features, img_feat_list, end_points)
        seed_sample_inds = sample_valid_seeds(features[:,-1,:], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1,features.shape[1],1))
        xyz = torch.gather(xyz, 1, seed_sample_inds.unsqueeze(-1).repeat(1,1,3))
        seed_inds = torch.gather(seed_inds, 1, seed_sample_inds)

        end_points['seed_inds'] = seed_inds
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        pc_features = features[:,:256,:]
        img_features = features[:,256:,:]
        img_features = self.image_mlp(img_features)
        joint_features = torch.cat((pc_features, img_features), 1)

        if not joint_only:
            # --------- IMAGE-ONLY TOWER ---------
            prefix = 'img_only_'
            xyz, features = self.img_only_vgen(end_points['seed_xyz'], img_features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            end_points[prefix+'vote_xyz'] = xyz
            end_points[prefix+'vote_features'] = features
            end_points = self.img_only_pnet(xyz, features, end_points)

            # --------- POINTS-ONLY TOWER ---------
            prefix = 'pc_only_'
            xyz, features = self.pc_only_vgen(end_points['seed_xyz'], pc_features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            end_points[prefix+'vote_xyz'] = xyz
            end_points[prefix+'vote_features'] = features
            end_points = self.pc_only_pnet(xyz, features, end_points)

        # --------- JOINT TOWER ---------
        prefix = 'pc_img_'
        xyz, features = self.pc_img_vgen(end_points['seed_xyz'], joint_features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points[prefix+'vote_xyz'] = xyz
        end_points[prefix+'vote_features'] = features
        end_points = self.pc_img_pnet(xyz, features, end_points)

        return end_points

