#!/bin/bash

python scripts/train_pose_transfer_model.py \
    --id deepfashion \
    --gpu_ids 0,1 \
    --dataset_name deepfashion \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg_deepfashion \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing \
    --check_grad_freq 3000 \
    --batch_size 4 \
    --n_epoch 45