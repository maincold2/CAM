# Coordinate-Aware Modulation for Neural Fields (ICLR 2024 Spotlight)
### Joo Chan Lee, Daniel Rho, Seungtae Nam, Jong Hwan Ko, and Eunbyung Park

### [[Project Page](https://maincold2.github.io/cam/)] [[Paper(arxiv)](https://arxiv.org/abs/2311.14993)]

## Method Overview
<img src="https://github.com/maincold2/maincold2.github.io/blob/master/cam/images/fig_overview.jpg?raw=true" />

In this work, we propose a novel way for exploiting both MLPs and grid representations in neural fields. Unlike the prevalent methods that combine them sequentially (extract features from the grids first and feed them to the MLP), we inject spectral bias-free grid representations into the intermediate features in the MLP. More specifically, we suggest a Coordinate-Aware Modulation (CAM), which modulates the intermediate features using scale and shift parameters extracted from the grid representations. This can maintain the strengths of MLPs while mitigating any remaining potential biases, facilitating the rapid learning of high-frequency components. 

## Setup

CAM is applied to various methods in a plug-and-play manner. We recommend following the original configurations of each method ([FFNeRV](https://github.com/maincold2/FFNeRV), [Mip-NeRF](https://github.com/google/mipnerf), [Mip-NeRF 360](https://github.com/google-research/multinerf), and [NerfAcc@ebeb5dd](https://github.com/nerfstudio-project/nerfacc/tree/ebeb5ddf733c04b425d5070efae9c3d23f64b078)).

## Image representation
### FFN
We implement CAM on the FFN pytorch implementation in [FINN](https://github.com/yixin26/FINN).

```shell
python CAM_image.py -g [GPU ID] --data [image or image_dir] --model [CAM or FFN] --exp [work_dir]
```

#### --downsample
Downsample factor of training images, where 1 means image regression, 2 by default

## Video representation
### FFNeRV

```bash
cd CAM_ffnerv

# For representation
python main.py -e 300 --lower-width 96 --num-blocks 1 --dataset [data_dir] --outf [work_dir] --fc-hw-dim 9_16_156 --expansion 1 --loss Fusion6 --strides 5 2 2 2 2  --conv-type conv -b 1  --lr 0.0005 --agg-ind -2 -1 1 2 --lw 0.1 --t-dim 64 128 256 512 --mod-dim 60

# For compression
python main.py -e 600 --lower-width 24 --num-blocks 1 --dataset [data_dir] --outf [work_dir] --fc-hw-dim 9_16_48 --expansion 8 --loss Fusion6 --strides 5 3 2 2 2  --conv-type compact -b 1  --lr 0.0005 --agg-ind -2 -1 1 2 --lw 0.1 --wbit 6 --t-dim 300 600 --resol 1920 1080 --mod-dim 30
```

#### --mod-dim
Temporal resolution of modulation grids, where 0 means the original FFNeRV, 60 by default

## NeRF
### Mip-NeRF
```shell
cd CAM_mipnerf

python -m train --data_dir=[nerf_synthetic_dir] --train_dir=[work_dir] --gin_file=configs/blender.gin --logtostderr

python -m eval --data_dir=[nerf_synthetic_dir] --train_dir=[work_dir] --gin_file=configs/blender.gin --logtostderr 
```
### Mip-NeRF 360
```shell
cd CAM_multinerf-mipnerf360

python -m train --gin_configs=configs/360.gin --gin_bindings="Config.data_dir = '[360 data_dir]'" --gin_bindings="Config.checkpoint_dir = '[work_dir]'" --logtostderr

python -m eval --gin_configs=configs/360.gin --gin_bindings="Config.data_dir = '[360 data_dir]'" --gin_bindings="Config.checkpoint_dir = '[work_dir]'" --logtostderr
```
### NerfAcc

```shell
cd CAM_nerfacc

PYTHONPATH=./ python examples/train_mlp_nerf.py --scene [scene_name] --data_root [nerfsyn_dataset_dir] --exp [work_dir]
```

## Dynamic NeRF

### NerfAcc

```shell
cd CAM_nerfacc

PYTHONPATH=./ python examples/train_mlp_tnerf.py --scene [scene_name] --data_root [dnerf_dataset_dir] --exp [work_dir]
```

## BibTeX
```
@inproceedings{
lee2024cam,
title={Coordinate-Aware Modulation for Neural Fields},
author={Joo Chan Lee and Daniel Rho and Seungtae Nam and Jong Hwan Ko and Eunbyung Park},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=4UiLqimGm5}
}
```
