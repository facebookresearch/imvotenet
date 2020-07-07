# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection on SUN RGB-D with VoteNet/ImVoteNet.

Sample usage:
python train.py --use_imvote --log_dir log_imvotenet

To use Tensorboard (need to install TensorFlow):
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
# ImVoteNet related options
parser.add_argument('--use_imvotenet', action='store_true', help='Use ImVoteNet (instead of VoteNet) with RGB.')
parser.add_argument('--max_imvote_per_pixel', type=int, default=3, help='Maximum number of image votes per pixel [default: 3]')
parser.add_argument('--tower_weights', default='0.3,0.3,0.4', help='Tower weights for img_only, pc_only and pc_img [default: 0.3,0.3,0.4]')
# Shared options with VoteNet
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--num_workers', type=int, default=4, help='Number of works for loading training data [default: 4]')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR
# Setting tower weights
if FLAGS.use_imvotenet:
    KEY_PREFIX_LIST = ['img_only_', 'pc_only_', 'pc_img_']
    weights = [float(x) for x in FLAGS.tower_weights.split(',')]
    TOWER_WEIGHTS = {'img_only_weight': weights[0], 'pc_only_weight': weights[1], 'pc_img_weight': weights[2]}
    print('Tower weights', TOWER_WEIGHTS)
else:
    KEY_PREFIX_LIST = ['pc_only_']
    TOWER_WEIGHTS = {'pc_only_weight': 1.0}

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
from model_util_sunrgbd import SunrgbdDatasetConfig
DATASET_CONFIG = SunrgbdDatasetConfig()
TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
    augment=True,
    use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
    use_imvote=FLAGS.use_imvotenet,
    max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
    use_v1=(not FLAGS.use_sunrgbd_v2))
TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
    augment=False,
    use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
    use_imvote=FLAGS.use_imvotenet,
    max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
    use_v1=(not FLAGS.use_sunrgbd_v2))
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.use_imvotenet:
    MODEL = importlib.import_module('imvotenet')
    net = MODEL.ImVoteNet(num_class=DATASET_CONFIG.num_class,
                        num_heading_bin=DATASET_CONFIG.num_heading_bin,
                        num_size_cluster=DATASET_CONFIG.num_size_cluster,
                        mean_size_arr=DATASET_CONFIG.mean_size_arr,
                        num_proposal=FLAGS.num_target,
                        input_feature_dim=num_input_channel,
                        vote_factor=FLAGS.vote_factor,
                        sampling=FLAGS.cluster_sampling,
                        max_imvote_per_pixel=FLAGS.max_imvote_per_pixel,
                        image_feature_dim=TRAIN_DATASET.image_feature_dim)

else:
    MODEL = importlib.import_module('votenet')
    net = MODEL.VoteNet(num_class=DATASET_CONFIG.num_class,
                        num_heading_bin=DATASET_CONFIG.num_heading_bin,
                        num_size_cluster=DATASET_CONFIG.num_size_cluster,
                        mean_size_arr=DATASET_CONFIG.mean_size_arr,
                        num_proposal=FLAGS.num_target,
                        input_feature_dim=num_input_channel,
                        vote_factor=FLAGS.vote_factor,
                        sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
  log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        if FLAGS.use_imvotenet:
            inputs.update({'scale': batch_data_label['scale'],
                           'calib_K': batch_data_label['calib_K'],
                           'calib_Rtilt': batch_data_label['calib_Rtilt'],
                           'cls_score_feats': batch_data_label['cls_score_feats'],
                           'full_img_votes_1d': batch_data_label['full_img_votes_1d'],
                           'full_img_1d': batch_data_label['full_img_1d'],
                           'full_img_width': batch_data_label['full_img_width'],
                           })
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            if key not in end_points:
                end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG, KEY_PREFIX_LIST, TOWER_WEIGHTS)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
                (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            output_str = "batch id: %d " % batch_idx
            for key_prefix in KEY_PREFIX_LIST:
                output_str += '%s: %f '%(key_prefix+'loss',
                                         stat_dict[key_prefix+'loss']/batch_interval)
            log_string(output_str)
            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator_dict = {}
    for key_prefix in KEY_PREFIX_LIST:
        ap_calculator_dict[key_prefix+'ap_calculator'] = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                                                                      class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        if FLAGS.use_imvotenet:
            inputs.update({'scale': batch_data_label['scale'],
                           'calib_K': batch_data_label['calib_K'],
                           'calib_Rtilt': batch_data_label['calib_Rtilt'],
                           'cls_score_feats': batch_data_label['cls_score_feats'],
                           'full_img_votes_1d': batch_data_label['full_img_votes_1d'],
                           'full_img_1d': batch_data_label['full_img_1d'],
                           'full_img_width': batch_data_label['full_img_width'],
                           })
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            if key not in end_points:
                end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG, KEY_PREFIX_LIST, TOWER_WEIGHTS)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        for key_prefix in KEY_PREFIX_LIST:
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, key_prefix)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
            ap_calculator_dict[key_prefix+'ap_calculator'].step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT %10 == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

    # Log statistics
    TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for key_prefix in KEY_PREFIX_LIST:
        metrics_dict = ap_calculator_dict[key_prefix+'ap_calculator'].compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
