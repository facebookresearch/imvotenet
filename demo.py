import os
import sys
import numpy as np
import argparse
import importlib
import time
from plyfile import PlyData
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import parse_predictions

NUM_CLS = 10             # sunrgbd number of classes
MAX_NUM_2D_DET = 100     # maximum number of 2d boxes per image
MAX_NUM_PIXEL = 530*730  # maximum number of pixels per image
KEY_PREFIX_LIST = ['pc_img_']

max_imvote_per_pixel = 3
vote_dims = 1 + max_imvote_per_pixel*4

type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6, 'night_stand': 7,
              'bookshelf': 8, 'bathtub': 9}
class2type = {type2class[t]: t for t in type2class}
num_point = 20000

# choose any of the samples provided
bbox_2d_path = os.path.join(BASE_DIR, 'demo/FasterRCNN_labels/000002.txt')
calib_filepath = os.path.join(BASE_DIR, 'demo/calib/000002.txt')
image_path = os.path.join(BASE_DIR, 'demo/images/000002.jpg')
depth_filepath = os.path.join(BASE_DIR, 'demo/depth/000002.mat')


def processData():
    """
    A pipeline to extract point cloud, geometric, semantic and texture cues for ImVoteNet training from Depth file,
    calibration file, 2D RGB Image and corresponding 2D Faster RCNN object detection algorithm output.

    :return:
    """
    def pre_load_2d_bboxes(bbox_2d_path):
        print("pre-loading 2d boxes from: " + bbox_2d_path)

        # Read 2D object detection boxes and scores
        cls_id_list = []
        cls_score_list = []
        bbox_2d_list = []

        for line in open(os.path.join(bbox_2d_path), 'r'):
            det_info = line.rstrip().split(" ")
            prob = float(det_info[-1])
            # Filter out low-confidence 2D detections
            if prob < 0.1:
                continue
            cls_id_list.append(type2class[det_info[0]])
            cls_score_list.append(prob)
            bbox_2d_list.append(np.array([float(det_info[i]) for i in range(4, 8)]).astype(np.int32))

        return cls_id_list, cls_score_list, bbox_2d_list


    def getCameraParameters(calib_filepath):
        """
        Calib .txt file looks as follows
        0.97959 0.012593 0.20061 0.012593 0.99223 -0.12377 -0.20061 0.12377 0.97182  -----> line 1 = rotation matrix
        529.5 0 0 0 529.5 0 365 265 1             ------> line 2 = camera intrinsic matrix

        :return: 3x3 rotation matrix and 3x3 camera intrinsic matrix
        """
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.reshape(np.array([float(x) for x in lines[0].rstrip().split(' ')]), (3, 3), 'F')
        Rtilt = np.expand_dims(Rtilt.astype(np.float32), 0)
        K = np.reshape(np.array([float(x) for x in lines[1].rstrip().split(' ')]), (3, 3), 'F')
        K = np.expand_dims(K.astype(np.float32), 0)

        return Rtilt, K

    def read_ply(ply_filepath):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(ply_filepath)
        pc = plydata['vertex'].data
        pc_array = np.array([[x, y, z] for x,y,z in pc])
        return pc_array

    def loadDepthMat(depth_filepath):
        """
        depth is represented as a .mat file.
        :param depth_filepath: path to directory
        :return: point cloud (x, y, z) values of each pixel
        """
        depth = sio.loadmat(depth_filepath)['instance']
        # print(np.array(depth).shape)   # -----> (N, 6)
        pc = np.array(depth)[:, :3]

        return pc

    def random_sampling(pc, num_sample, replace=None, return_choices=False):
        """ Input is NxC, output is num_samplexC
        """
        if replace is None: replace = (pc.shape[0]<num_sample)
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
        if return_choices:
            return pc[choices], choices
        else:
            return pc[choices]

    def preprocess_point_cloud(point_cloud):
        ''' Prepare the numpy point cloud (N,3) for forward pass '''
        point_cloud = point_cloud[:, 0:3]  # do not use color for now
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
        point_cloud = random_sampling(point_cloud, num_point)
        pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
        return pc

    def getClsScoreFeats(cls_id_list, cls_score_list):
        # Semantic cues: one-hot vector for class scores
        cls_score_feats = np.zeros((1 + MAX_NUM_2D_DET, NUM_CLS), dtype=np.float32)
        # First row is dumpy feature
        len_obj = len(cls_id_list)
        if len_obj:
            ind_obj = np.arange(1, len_obj + 1)
            ind_cls = np.array(cls_id_list)
            cls_score_feats[ind_obj, ind_cls] = np.array(cls_score_list)

        cls_score_feats = np.expand_dims(cls_score_feats.astype(np.float32), 0)

        return cls_score_feats

    def load_image(img_path):
        img = cv2.imread(img_path)
        return img

    def getTextureCues(full_img):
        # Texture cues: normalized RGB values
        full_img_height = full_img.shape[0]
        full_img_width = full_img.shape[1]
        full_img = (full_img - 128.) / 255.
        # Serialize data to 1D and save image size so that we can recover the original location in the image
        full_img_1d = np.zeros((MAX_NUM_PIXEL * 3), dtype=np.float32)
        full_img_1d[:full_img_height * full_img_width * 3] = full_img.flatten()
        full_img_1d = np.expand_dims(full_img_1d.astype(np.float32), 0)

        return full_img_1d, np.array(full_img_width)

    def get_full_img_votes_1d(full_img, cls_id_list, bbox_2d_list):
        obj_img_list = []
        for i2d, (cls2d, box2d) in enumerate(zip(cls_id_list, bbox_2d_list)):
            xmin, ymin, xmax, ymax = box2d

            obj_img = full_img[ymin:ymax, xmin:xmax, :]
            obj_h = obj_img.shape[0]
            obj_w = obj_img.shape[1]

            # Bounding box coordinates (4 values), class id, index to the semantic cues
            meta_data = (xmin, ymin, obj_h, obj_w, cls2d, i2d)
            if obj_h == 0 or obj_w == 0:
                continue

            # Use 2D box center as approximation
            uv_centroid = np.array([int(obj_w / 2), int(obj_h / 2)])
            uv_centroid = np.expand_dims(uv_centroid, 0)

            v_coords, u_coords = np.meshgrid(range(obj_h), range(obj_w), indexing='ij')
            img_vote = np.transpose(np.array([u_coords, v_coords]), (1, 2, 0))
            img_vote = np.expand_dims(uv_centroid, 0) - img_vote

            obj_img_list.append((meta_data, img_vote))

        full_img_height = full_img.shape[0]
        full_img_width = full_img.shape[1]
        full_img_votes = np.zeros((full_img_height, full_img_width, vote_dims), dtype=np.float32)

        # Empty votes: 2d box index is set to -1
        full_img_votes[:, :, 3::4] = -1.

        for obj_img_data in obj_img_list:
            meta_data, img_vote = obj_img_data
            u0, v0, h, w, cls2d, i2d = meta_data
            for u in range(u0, u0 + w):
                for v in range(v0, v0 + h):
                    iidx = int(full_img_votes[v, u, 0])
                    if iidx >= max_imvote_per_pixel:
                        continue
                    full_img_votes[v, u, (1 + iidx * 4):(1 + iidx * 4 + 2)] = img_vote[v - v0, u - u0, :]
                    full_img_votes[v, u, (1 + iidx * 4 + 2)] = cls2d
                    full_img_votes[v, u, (1 + iidx * 4 + 3)] = i2d + 1
                    # add +1 here as we need a dummy feature for pixels outside all boxes
            full_img_votes[v0:(v0 + h), u0:(u0 + w), 0] += 1

        full_img_votes_1d = np.zeros((MAX_NUM_PIXEL * vote_dims), dtype=np.float32)
        full_img_votes_1d[0:full_img_height * full_img_width * vote_dims] = full_img_votes.flatten()

        full_img_votes_1d = np.expand_dims(full_img_votes_1d.astype(np.float32), 0)

        return full_img_votes_1d

    scale_ratio = np.array(1.0).astype(np.float32)
    point_cloud = loadDepthMat(depth_filepath)
    pc = preprocess_point_cloud(point_cloud)
    Rtilt, K = getCameraParameters(calib_filepath)
    cls_id_list, cls_score_list, bbox_2d_list = pre_load_2d_bboxes(bbox_2d_path)
    cls_score_feats = getClsScoreFeats(cls_id_list, cls_score_list)
    full_img = load_image(image_path)
    full_img_1d, full_img_width = getTextureCues(full_img)
    full_img_votes_1d = get_full_img_votes_1d(full_img, cls_id_list, bbox_2d_list)

    return scale_ratio, pc, Rtilt, K, cls_score_feats, full_img_1d, full_img_width, full_img_votes_1d

if __name__ == '__main__':

    from sunrgbd_detection_dataset import DC  # dataset config

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo')
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    checkpoint_path = os.path.join(demo_dir, 'checkpoint.tar')

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('imvotenet')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = MODEL.ImVoteNet(input_feature_dim=1, num_proposal=256, vote_factor=1, sampling='vote_fps',
                          max_imvote_per_pixel=3, image_feature_dim=18,
                          num_class=DC.num_class, num_heading_bin=DC.num_heading_bin,
                          num_size_cluster=DC.num_size_cluster,
                          mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    # Load and preprocess inputs
    net.eval()  # set model to eval mode (for bn and dp)
    scale_ratio, pc, Rtilt, K, cls_score_feats, full_img_1d, full_img_width, full_img_votes_1d = processData()

    print('Input Data Loaded')

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    inputs.update({'scale': torch.from_numpy(scale_ratio).to(device),
                   'calib_K': torch.from_numpy(K).to(device),
                   'calib_Rtilt': torch.from_numpy(Rtilt).to(device),
                   'cls_score_feats': torch.from_numpy(cls_score_feats).to(device),
                   'full_img_votes_1d': torch.from_numpy(full_img_votes_1d).to(device),
                   'full_img_1d': torch.from_numpy(full_img_1d).to(device),
                   'full_img_width': torch.from_numpy(full_img_width).to(device),
                   })
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs, joint_only=True)
    toc = time.time()
    print('Inference time: %f' % (toc - tic))

    end_points['point_clouds'] = inputs['point_clouds']
    end_points['scale'] = inputs['scale']
    end_points['calib_K'] = inputs['calib_K']
    end_points['calib_Rtilt'] = inputs['calib_Rtilt']
    end_points['cls_score_feats'] = inputs['cls_score_feats']
    end_points['full_img_votes_1d'] = inputs['full_img_votes_1d']
    end_points['full_img_1d'] = inputs['full_img_1d']
    end_points['full_img_width'] = inputs['full_img_width']
    pred_map_cls = parse_predictions(end_points, eval_config_dict, KEY_PREFIX_LIST[0])
    print('Finished detection. %d object detected.' % (len(pred_map_cls[0])))

    dump_dir = os.path.join(demo_dir, 'results')
    if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    MODEL.dump_results(end_points, dump_dir, DC, inference_switch=False, key_prefix='pc_img_')
    print('Dumped detection results to folder %s' % (dump_dir))