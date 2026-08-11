# PyramidDiff Training and Inference Guide

PyramidDiff is a layout-conditioned image generation method built on the HiCo-style ControlNet branch in this repository. It uses Stable Diffusion 1.5 as the frozen image generation backbone and trains only the PyramidDiff localization refinement components.

## Architecture Summary

### Frozen base model

The pretrained Stable Diffusion/HiCo backbone remains frozen during PyramidDiff training:

- **VAE encoder/decoder:** maps 512 × 512 images to and from 64 × 64 latent tensors.
- **UNet:** denoising encoder/decoder with convolutional and transformer blocks.
- **Text encoder:** CLIP ViT-L/14 text conditioning.
- **HiCo/ControlNet layout branch:** initialized from a pretrained HiCo checkpoint when provided.

### Frozen localization teacher

PyramidDiff uses a frozen COCO-pretrained YOLOv11n detector as a localization teacher. Detector neck features are extracted from three scales:

| Level | Resolution | Channels | Role |
| --- | --- | --- | --- |
| P3 | 64 × 64 | 256 | fine spatial detail |
| P4 | 32 × 32 | 512 | balanced detail and semantics |
| P5 | 16 × 16 | 1024 | high-level context |

The detector is used for localization supervision only; classification loss is not part of the PyramidDiff objective because object semantics are supplied by layout captions.

## Trainable PyramidDiff Components

### Localization Refinement Decoder (LRD)

The LRD mirrors four frozen UNet decoder stages and produces ControlNet-style residual corrections:

```text
Delta_i = Z_i(C_i(U_i))
U_i_hat = U_i + Delta_i
```

Each trainable convolution block `C_i` is:

1. 3 × 3 convolution, stride 1, padding 1
2. GroupNorm with up to 32 groups
3. SiLU activation

Each zero-conv `Z_i` is a 1 × 1 convolution with all weights and biases initialized to zero, preserving pretrained behavior at the start of training.

| Stage | Resolution | Channels | Source |
| --- | --- | --- | --- |
| d1 | 4 × 4 | 1280 | deepest decoder stage |
| d2 | 8 × 8 | 1280 | second deepest decoder stage |
| d3 | 16 × 16 | 640 | mid-resolution decoder stage |
| d4 | 32 × 32 | 720 | highest-resolution decoder stage |

Transformer/attention blocks inherited from HiCo remain frozen; only convolutional PyramidDiff refinement parameters are optimized.

### Adaptive Cross-scale Distillation Module (ACDM)

ACDM is used only during training and discarded for inference. It transfers YOLOv11n localization knowledge to the LRD using a shared feature space of **128 channels at 32 × 32 resolution**.

ACDM contains:

1. **Localization Alignment Block (LAB):** GroupNorm → SiLU → 1 × 1 projection → resize to 32 × 32.
2. **Adaptive Scale Aggregation (ASA):** global average pooling over concatenated detector scales followed by `384 -> 128 -> 3` MLP and softmax scale weights.
3. **Adaptive Pyramid Fusion (APF):** weighted detector features are concatenated and fused by 3 × 3 and 1 × 1 convolutions.
4. **Spatial Localization Transfer (SLT):** 4-head cross-attention where aligned LRD features are queries and fused detector features are keys/values. Each stage has a learnable scale initialized to zero.

## Data Preparation

Use COCO 2017 layout/image pairs:

- train: 118,287 images
- validation: 5,000 images
- input resolution: 512 × 512
- boxes normalized to `[0, 1]` relative to 512 × 512
- local caption per box from the COCO category name

Training augmentation:

- random crop probability: 5%
- horizontal flip probability: 50%
- retain small objects, including area below 32 × 32 pixels

## Environment

Recommended reproducibility environment:

| Package | Version |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| Diffusers | 0.24.0/local fork |
| Transformers | 4.35.2 |
| Ultralytics | 8.0.196 |
| OpenCV | 4.8.1 |
| Albumentations | 1.3.1 |

Hardware used in the appendix: single Google Colab A100 40 GB, 25 GB RAM. Mixed precision and gradient checkpointing require about 22 GB VRAM. Full 50-epoch COCO training takes about seven days on one A100.

## Training Pipeline

Install dependencies and expose the local diffusers fork:

```bash
pip install -r requirements.txt
pip install -e diffusers
```

Launch training:

```bash
accelerate launch train_pyramiddiff.py \
  --pretrained_model_name_or_path /path/to/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path /path/to/hico-controlnet \
  --train_data_yaml utils/dataset/latent_LayoutDiffusion_large_grit.yaml \
  --output_dir outputs/pyramiddiff \
  --resolution 512 \
  --train_batch_size 8 \
  --num_train_epochs 50 \
  --learning_rate 1e-4 \
  --adam_weight_decay 1e-2 \
  --lr_scheduler cosine \
  --lr_warmup_steps 5000 \
  --max_grad_norm 1.0 \
  --detector_neck_channels 256 512 1024 \
  --diffusion_loss_weight 1.0 \
  --detector_localization_loss_weight 0.1 \
  --gradient_checkpointing \
  --mixed_precision fp16
```

Notes:

- Omit `--controlnet_model_name_or_path` to initialize the ControlNet/HiCo branch from the frozen UNet.
- Use `--acdm_decoder_model_name_or_path outputs/pyramiddiff/acdm_decoder.pt` to resume PyramidDiff LRD/ACDM weights.
- The intended total objective is `L = L_diff + lambda * gamma * L_loc`, with `lambda = 0.1`.
- `gamma` is the mean IoU between detector predictions and ground-truth layout boxes.
- The localization term uses Ultralytics CIoU box loss over raw P3/P4/P5 predictions before NMS.
- The current training script preserves standard diffusion fine-tuning and exposes the detector localization loss weight for detector-aware datasets or pipeline integrations that provide YOLOv11n outputs.

## Inference Pipeline

At inference time, ACDM is removed. Generation uses the frozen backbone plus trained LRD residual convolutions. The expected denoising loop is:

1. Read layout condition: boxes plus local captions.
2. Sample latent Gaussian noise.
3. For each denoising timestep, predict noise with the frozen UNet and add LRD residuals at decoder stages.
4. Decode the final latent through the frozen VAE.

Run batch inference from a JSON layout file:

```bash
python infer-avg.py \
  --json results/examples/json_1.json \
  --save-dir results/pyramiddiff
```

The inference JSON entries must contain at least these six fields:

```text
[base_info, caption, obj_nums, img_size, path_img, list_bbox_info]
```

Each `list_bbox_info` item should include the local category label and its box. Input image paths are optional; when omitted, the script creates a blank layout canvas from the JSON image size.

For the Gradio demo, edit `base_model_path` and `controlnet_path` in `gradio_pyramiddiff.py`, then run:

```bash
python gradio_pyramiddiff.py
```

## Main Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-2 |
| batch size | 8 |
| epochs | 50 |
| gradient clipping | 1.0 |
| scheduler | cosine with warmup |
| warmup steps | 5,000 |
| localization balance λ | 0.1 |
| diffusion timesteps | 1,000 |
| ACDM channels | 128 |
| ACDM resolution | 32 × 32 |
| SLT heads | 4 |

## References

- ControlNet: Zhang, Rao, and Agrawala, ICCV 2023.
- HiCo: Cheng et al., NeurIPS 2024.
- Latent Diffusion / Stable Diffusion: Rombach et al., CVPR 2022.
- DDPM: Ho, Jain, and Abbeel, NeurIPS 2020.
- Ultralytics YOLO11: https://github.com/ultralytics/ultralytics
- CIoU/DIoU loss: Zheng et al., AAAI 2020.
