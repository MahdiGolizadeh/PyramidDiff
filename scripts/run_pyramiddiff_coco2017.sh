#!/usr/bin/env bash
set -euo pipefail

# PyramidDiff COCO2017 setup from the paper protocol:
# - MS COCO 2017 train split for training, val split for quantitative evaluation.
# - 8x A100, total batch size 64 (per-device batch size 8).
# - 30 epochs, AdamW.
# - MSOP enabled in the HiCo/GroundNet conditioning branch.
# - DA-ACL enabled with frozen YOLOv11-X (Ultralytics yolo11x.pt).

accelerate launch --config_file utils/accelerate_config.yaml train_pyramiddiff.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL:-runwayml/stable-diffusion-v1-5}" \
  --train_data_dir "${COCO_ROOT:-./coco}" \
  --train_data_yaml utils/dataset/latent_LayoutDiffusion_large_coco.yaml \
  --dataset_backend coco \
  --output_dir "${OUTPUT_DIR:-outputs/pyramiddiff_coco2017}" \
  --resolution 512 \
  --train_batch_size 8 \
  --num_train_epochs 30 \
  --learning_rate 5e-6 \
  --adam_weight_decay 1e-2 \
  --enable_msop \
  --enable_da_acl \
  --verification_detector yolo11x.pt \
  --lambda_loc 1.0 \
  --lambda_cls 1.0 \
  --lambda_miss 2.0 \
  --da_acl_weight 0.1 \
  --synthetic_eval_images 10000 \
  --downstream_detector yolo11m.pt
