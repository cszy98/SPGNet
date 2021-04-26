#!/bin/bash

python scripts/test_pose_transfer_model.py \
 --id market \
 --gpu_ids 0 \
 --dataset_name market \
 --which_model_G dual_unet \
 --G_feat_warp 1 \
 --G_vis_mode residual \
 --pretrained_flow_id FlowReg_market \
 --pretrained_flow_epoch best \
 --dataset_type pose_transfer_parsing_market \
 --which_epoch latest \
 --batch_size 1 \
 --save_output \
 --output_dir output