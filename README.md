# TSN-Pytorch

Pytorch版TSN，模型比较丰富，代码改动比较灵活。

### Install

Choice 1：如果使用V100的卡进行训练，则需要使用cuda9.0的镜像，从`reg-xs.qiniu.io/atlab/base/mxnet/gpu.1.0.0.cu9:example`创建一个容器，进入到容器中。安装：

```
wget http://p22k53ufs.bkt.clouddn.com/cuda9.0/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
```

Choice 2：使用其他cuda8.0的卡进行训练，则从基础镜像（`reg-xs.qiniu.io/atlab/atnet-mxnet-trainer-gpu:20170714v1`）创建一个容器，进入到容器中。安装：

```
wget http://p1wqzrl8v.bkt.clouddn.com/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
pip install torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
```

### Data Preparation

1. 此过程和caffe版的数据预处理过程一致，请参看`Alg-VideoAlgorithm/video-classification/temporal-segment-networks/README.md`

### Training

以UCF101在BN-Inception上训练Spatial Network为例。

1. 生成训练集和测试集的file lists：

   由于训练时的输入数据依赖于caffe的`VideoDataLayer`层，这个层需要指定一个file list作为其数据来源。file list的每一行包含每个视频的帧存储位置，视频帧数，视频的groudtruth类别。例如，一个file list长这样：

   ```
   /workspace/data/UCF-frames/v_HorseRace_g11_c02 279 40
   /workspace/data/UCF-frames/v_Rowing_g10_c01 481 75
   /workspace/data/UCF-frames/v_PlayingTabla_g12_c03 256 65
   /workspace/data/UCF-frames/v_BandMarching_g21_c01 311 5
   ...
   ```

   要构建file list，运行以下脚本：

   ```
   bash scripts/build_file_list.sh ucf101 FRAME_PATH
   ```

   生成的file list存储在`data/`目录下，命名规则如`ucf101_rgb_train_split_1.txt`。

2. 开始训练：

   对于RGB数据的训练，执行：

   ```python
   python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
      --arch BNInception --num_segments 3 \
      --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
      -b 128 -j 8 --dropout 0.8 \
      --snapshot_pref ucf101_bninception_ 
   ```

   对于Flow数据的训练，执行：

   ```python
   python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
      --arch BNInception --num_segments 3 \
      --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
      -b 128 -j 8 --dropout 0.7 \
      --snapshot_pref ucf101_bninception_ --flow_pref flow_  
   ```

   对于RGB-diff数据的训练，执行：

   ```python
   python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
      --arch BNInception --num_segments 7 \
      --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
      -b 128 -j 8 --dropout 0.8 \
      --snapshot_pref ucf101_bninception_ 
   ```

   ​

### Testing

to be continued ...



### Reference

[1] https://github.com/yjxiong/tsn-pytorch