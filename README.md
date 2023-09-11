# WCNN3D

This repo demonstrates how to reproduce the results from
[WCNN3D: Wavelet Convolutional Neural Network-Based 3D Object Detection for Autonomous Driving](https://www.mdpi.com/1424-8220/22/18/7010) (Published in MDPI Sensors journal, 2022) on the
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/). The codebase is built on top of [PointtPillars](https://github.com/nutonomy/second.pytorch) and [SECOND](https://github.com/traveller59/second.pytorch). If you want to train on nuScenes dataset, check [Second](https://github.com/traveller59/second.pytorch). 
![wcnn3d](images/wcnn3d_arch.png)

If you find this work useful, please cite:


```bash
@article{alaba2022wcnn3d,
  title={Wcnn3d: Wavelet convolutional neural network-based 3d object detection for autonomous driving},
  author={Alaba, Simegnew Yihunie and Ball, John E},
  journal={Sensors},
  volume={22},
  number={18},
  pages={7010},
  year={2022},
  publisher={MDPI}
}
```

## Getting Started


Download the KITTI dataset from [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). The LiDAR dataset only is used for this experiment. Download the velodyne point clouds and prepare it based on the guidelines. 


### Code Support

The code was tested on Python 3.7, PyTorch 1.10.0, and CUDA 11.3 on Ubuntu 20.04. It should work for recent versions of Python and PyTorch.

### Package Install

#### 1. Clone code

```bash
git clone https://github.com/Simeon340703/WCNN3D.git
```

#### 2. Install Python packages

It is recommend to use the Anaconda package manager.

First, use Anaconda to configure as many packages as possible.
```bash
conda create -n wcnn3d python=3.7 anaconda
conda activate wcnn3d
conda install shapely pybind11 protobuf scikit-image numba pillow
#It has been tested on the following pyTorch version
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge 
conda install google-sparsehash -c bioconda
```
Then use pip for the packages missing from Anaconda.
```bash
pip install --upgrade pip
pip install fire tensorboardX
```

Install SparseConvNet and [spconv](https://github.com/traveller59/spconv). These are not required for this work, but the general 
SECOND and Pointpillars code base expects these for correct configuration. 
spconv can be installed from the terminal, but it is recommeneded to install SparseConvNet from source.
```bash
pip install spconv-cu113	

git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
# NOTE: if bash build.sh fails, try bash develop.sh instead
```
	

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```


#### 3. Setup cuda for numba

This is deperecated. How ever, recommended to add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

#### 4. PYTHONPATH

Add wcnn3d/ to your PYTHONPATH. You may add like the following if you are using 
ubuntu and the code is in your home directory.
```bash
export PYTHONPATH=$PYTHONPATH:/home/xxxx/wcnn3d
```

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and prepare similar to the following file structure:

```plain
└── KITTI_DATASET_ROOT
       ├──                |  training    <-- 7481 train data
   data|                  |  ├── image_2 <-- for visualization
       |sets|Kitt_second- |  ├── calib
       |                  |  ├── label_2
       |                  |  ├── velodyne
       |                  |  └── velodyne_reduced <-- empty directory
       |                  |  testing     <-- 7580 test data
       |                  |  ├── image_2 <-- for visualization
       |                  |  ├── calib
                          |  ├── velodyne
       |                  |  └── velodyne_reduced <-- empty directory
       |ImageSets|test.txt
                  |train.txt
                  |val.txt
                  |trainval.txt
```


#### 2. Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
python create_data.py create_kitti_info_file --data_path=data/sets/kitti_second/
```

#### 3. Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
python create_data.py create_reduced_point_cloud --data_path=data/sets/kitti_second/
```
#### 4. Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
python create_data.py create_groundtruth_database --data_path=data/sets/kitti_second/
```

#### 5. Modify config file

The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```


### Train
The default training is for the car model. If you want to train pedestrians and cyclists' network, WcnnRPNPedCycle() class is inside the wcnn3d.py (second/pytorch/models/wcnn3d.py) file. Create the class instance in the RPN class (inside the voxelnet.py file). The default wavelet is the Haar wavelet, with four levels of decomposition for cars and three-level of decomposition for pedestrians and cyclists. Download the checkpoint and place it in the model directory [checkpoint](https://drive.google.com/drive/folders/1Y2MBxfxHY2PQ_x4nVI4tx1xNgQF_p85y?usp=share_link).

```bash
cd ~/wcnn3d/second
python ./pytorch/train.py train --config_path=./configs/wcnn3d/car/xyres_16.proto --model_dir=/path/to/model_dir
If you want train ped_cycle class
python ./pytorch/train.py train --config_path=./configs/wcnn3d/ped_cycle/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.



### Evaluate


```bash
cd ~/wcnn3d/second/
#python pytorch/train.py evaluate --config_path= configs/wcnn3d/car/xyres_16.proto --model_dir=/path/to/model_dir
for ped_cycle
python ./pytorch/train.py evaluate --config_path=./configs/wcnn3d/ped_cycle/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.

## Visualization -Try Kitti Viewer Web
### Major step
run python ./kittiviewer/backend/main.py main --port=xxxx in your server/local.

run cd ./kittiviewer/frontend && python -m http.server to launch a local web server.

open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

input backend url (e.g. http://127.0.0.1:16666)

input root path, info path, and det path (optional)

click load, loadDet (optional), and input image index.

## Inference step
Firstly the load button must be clicked and load successfully.

input checkpointPath and configPath.

click buildNet.

click inference.
