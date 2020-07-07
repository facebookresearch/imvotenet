# ImVoteNet
**Boosting 3D Object Detection in Point Clouds with Image Votes**

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
pip install matplotlib opencv-python plyfile networkx==2.2 trimesh==2.35.39 tensorboardX
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

## Training and Evaluation

