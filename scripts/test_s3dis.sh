#!/bin/bash
set -ex

# eval pretrained model
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore test_seg.py --config config/s3dis/s3dis_sapformer.yaml \
                                  model_path checkpoints/s3dis_sapformer.pth \
                                  save_folder 'temp/s3dis/results'


