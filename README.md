# ImVoteNet
**Boosting 3D Object Detection in Point Clouds with Image Votes**

<p align="center">
  <img src="http://xinleic.xyz/images/imvote.png" width="500" />
</p>

This repository contains the code release of the [paper](https://arxiv.org/abs/2001.10692):
```
@inproceedings{qi2020imvotenet,
  title={Imvotenet: Boosting 3d object detection in point clouds with image votes},
  author={Qi, Charles R and Chen, Xinlei and Litany, Or and Guibas, Leonidas J},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Installation
Overall, the installation is similar to [VoteNet](https://github.com/facebookresearch/votenet). GPU is required. The code is tested with Ubuntu 18.04, Python 3.7.7, PyTorch 1.4.0, CUDA 10.0 and cuDNN v7.4.

First install [PyTorch](https://pytorch.org/get-started/locally/), for example through [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
Next, install Python dependencies via `pip` ([tensorboardX](https://github.com/lanpa/tensorboardX) is used for for visualizations):
```bash
pip install matplotlib opencv-python plyfile tensorboardX tqdm networkx==2.2 trimesh==2.35.39
```
Now we are ready to clone this repository:
```bash
git clone git@github.com:facebookresearch/imvotenet.git
cd imvotenet
```
The code depends on [PointNet++](http://arxiv.org/abs/1706.02413) as a backbone, which needs compilation:
```bash
cd pointnet2
python setup.py install
cd ..
```

## Data
Please follow the steps listed [here](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) to set up the SUN RGB-D dataset in the `sunrgbd` folder. The expected dataset structure under `sunrgbd` is:
```
sunrgbd/
  sunrgbd_pc_bbox_votes_50k_{v1,v2}_{train,val}/
  sunrgbd_trainval/
    # raw image data and camera used by ImVoteNet
    calib/*.txt
    image/*.jpg
```
For ImVoteNet, we provide 2D detection results from a pre-trained Faster R-CNN detector [here](https://dl.fbaipublicfiles.com/imvotenet/2d_bbox/sunrgbd_2d_bbox_50k_v1.tgz). Please download the file, uncompress it, and place the resulting folders (`sunrgbd_2d_bbox_50k_v1_{train,val}`) under `sunrgbd` as well.

## Training and Evaluation

Once the code and data are set up, one can train ImVoteNet by the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_imvotenet --log_dir log_imvotenet
```

To reproduce the experimental results in the paper and in general have faster development cycles, one can use a shorter learning schedule: 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_imvotenet --log_dir log_140ep --max_epoch 140 --lr_decay_steps 80,120 --lr_decay_rates 0.1,0.1
```

As a baseline, this code also supports training of the original VoteNet, which is launched by:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --log_dir log_votenet
```
In fact, the code is based on the VoteNet repository at commit [2f6d6d3](https://github.com/facebookresearch/votenet/tree/2f6d6d3), as a reference, it gives around 58 mAP@0.25.

For other training options, one can use `python train.py -h` for assistance.

## TODO
1. Check the intermediate dimension of `VotingModule` (whether fixed 256, or change as input dimension changes)
2. Learning rate schedule, check the 140/180/200 epoch schedules
3. Add docs for some functions
