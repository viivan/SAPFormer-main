#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls.py --config config/scannet/scannet_sapformer.yaml \
                                 model_path checkpoints/scannet_sapformer.pth
