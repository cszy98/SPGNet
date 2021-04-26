#!/bin/bash

python scripts/train_pose_transfer_model.py \
--id market_test \
--gpu_ids 0,1,2,3 \
--dataset_name market \
--which_model_G dual_unet \
--G_feat_warp 1 \
--G_vis_mode residual \
--pretrained_flow_id FlowReg_market \
--pretrained_flow_epoch best \
--dataset_type pose_transfer_parsing_market \
--check_grad_freq 3000 \
--batch_size 32 \
--n_epoch 10