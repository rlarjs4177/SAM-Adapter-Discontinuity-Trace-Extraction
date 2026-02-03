# 터널 막장면 경계선 추출 및 천공 예측

## 전체 개요
1. SAM-Adapter 를 이용하여 경계선 마스크 예측
2. 후처리 과정을 통해 경계선을 추출하고 천공 위치를 예측 


## 프로젝트 진행 목표

터널 막장면에서 할암봉 파쇄 작업을 자동화하는 시스템을 구축하기 위해 
불연속면 경계선을 추출하고 천공위치를 예측하여 이를 구현 하는 것

## SAM-Adapter
기존 SAM-Adapter에서 디코더에 Adapter와 Learnable Domain Prompt Token 추가


## Installation
모든 종속성 코드는 SAM-adapter (https://github.com/tianrun-chen/SAM-Adapter-PyTorch).

```
pip install -r requirements.txt
```

## Train
Training the SAM-adapter requires a large memory usage even with vit-base model (smallest).
Thus, we train it by original SAM-adapter script implemented by pytorch DistributedDataParallel with 4xA6000 GPUs.

If there is a machine with 4 GPUs in a signle node, please run below script to train the SAM-adapter.
```
bash _script/train.sh
```

Datasets should be located in below directory (or need to change "root_path_1" and "root_path_2" in "configs/custom-b.yaml").
```
(Train image, gts): "../dataset/img", "../dataset/gt"
(Test image, gts): "../dataset/test_raw", "../dataset/test_gt"
```

## Inference
Please check "inference.ipynb"

Main block in inference.ipynb has 4 configuration arguments as belows
- config: configuration file for SAM-adapter (currently, default path of train.sh)
- model: checkpoint file for SAM-adapter (currently, default path of train.sh)
- images_path: directory path that contains test images
- save_path: directory path to save the outputs (containing edge masks and hole csv)

In postprocessing there are 3 parameters as belows
- hole_min_distance: minimum distance between holes
- hole_gridsearch_distance: distance between hole candidates
- fragile_clearance: distance from edges to refine clearance areas