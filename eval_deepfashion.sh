#!/bin/bash

python tools/metrics_deepfashion.py \
--gt_path datasets/deepfashion/img/ground_truth \
--distorated_path checkpoints/PoseTransfer_deepfashion/output \
--fid_real_path datasets/deepfashion/img/train