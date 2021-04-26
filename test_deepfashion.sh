#!/bin/bash

python scripts/test_pose_transfer_model.py \
    --id deepfashion \
    --gpu_ids 0 \
    --dataset_name deepfashion \
    --which_model_G dual_unet \
    --G_feat_warp 1 \
    --G_vis_mode residual \
    --pretrained_flow_id FlowReg_deepfashion \
    --pretrained_flow_epoch best \
    --dataset_type pose_transfer_parsing \
    --which_epoch latest \
    --batch_size 4 \
    --save_output \
    --output_dir output
