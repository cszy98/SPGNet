#!/bin/bash

python tools/metrics_market.py \
--gt_path datasets/market1501/img/ground_truth/ \
--distorated_path checkpoints/PoseTransfer_market/output/ \
--fid_real_path datasets/market1501/img/train/