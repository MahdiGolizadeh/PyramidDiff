# PyramidDiff training and inference

PyramidDiff is built on the original HiCo ControlNet/layout branch in this repository. The Stable Diffusion VAE, text encoder, and UNet remain frozen. Training adds two extra components:

1. **Copied UNet decoder branch** (`CopiedUNetDecoderBranch`): a copy of the frozen UNet decoder. Its ResNet parameters are trainable, while copied transformer/attention parameters remain frozen.
2. **ACDM adapter** (`ACDMAdapter`): a separate adapter from the decoder. It receives copied-decoder ResNet features and YOLOv11n neck features, applies LAB to each level, APF over the three detector neck scales, and SLT cross-attention with decoder features as queries and detector features as keys/values. Its zero-conv outputs are intended to be injected into the frozen UNet decoder as residual corrections, ControlNet-style.

## Training

Use the existing HiCo/PyramidDiff training script and point it at the Stable Diffusion base checkpoint plus an optional HiCo ControlNet checkpoint:

```bash
accelerate launch train_pyramiddiff.py \
  --pretrained_model_name_or_path /path/to/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path /path/to/hico-controlnet \
  --train_data_yaml utils/dataset/latent_LayoutDiffusion_large_grit.yaml \
  --output_dir outputs/pyramiddiff-acdm \
  --resolution 512 \
  --train_batch_size 4 \
  --learning_rate 5e-6 \
  --detector_neck_channels 256 512 1024 \
  --diffusion_loss_weight 1.0 \
  --detector_localization_loss_weight 1.0
```

Notes:

- Omit `--controlnet_model_name_or_path` to initialize the HiCo/ControlNet branch from the frozen UNet.
- Use `--acdm_decoder_model_name_or_path outputs/pyramiddiff-acdm/acdm_decoder.pt` to resume or fine-tune the copied decoder and ACDM adapter weights.
- Final outputs include the HiCo/ControlNet-compatible weights in `output_dir` and the additional PyramidDiff branch weights in `acdm_decoder.pt`.
- The detector localization loss flag is exposed for detector-aware KD datasets/pipelines that provide frozen YOLOv11n outputs; the script keeps the standard diffusion reconstruction loss active for normal fine-tuning.

## Inference

Use the PyramidDiff-named pipeline from the local diffusers fork. The inference scripts now load the HiCo base branch under PyramidDiff naming:

```bash
python infer-avg.py \
  --json results/examples/json_1.json \
  --save-dir results/pyramiddiff
```

For the Gradio demo, set `base_model_path` and `controlnet_path` in `gradio_pyramiddiff.py`, then run:

```bash
python gradio_pyramiddiff.py
```

At inference time, load the existing HiCo/ControlNet-compatible checkpoint as the base conditioning branch and load `acdm_decoder.pt` when your pipeline wiring injects copied-decoder/ACDM residual corrections into the frozen UNet decoder.
