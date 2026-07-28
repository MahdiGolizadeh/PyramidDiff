# Copyright 2026 PyramidDiff authors.
# Licensed under the Apache License, Version 2.0.
"""ACDM decoder modules for PyramidDiff.

The module implements the detector-aware decoder branch requested by PyramidDiff:

* a trainable copy of the Stable Diffusion UNet decoder ResNet path,
* frozen copied transformer/attention blocks,
* LAB adapters for every decoder and detector-neck feature,
* APF detector-neck weighting and fusion, and
* SLT cross-attention from decoder features to detector features followed by
  zero-conv residual corrections that can be injected into the frozen UNet decoder.
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _num_groups(channels: int, preferred: int = 32) -> int:
    for groups in (preferred, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class LevelAdapterBlock(nn.Module):
    """LAB: GroupNorm -> SiLU -> 1x1 Conv projection to a common channel size."""

    def __init__(self, in_channels: int, out_channels: int = 128, norm_groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(in_channels, norm_groups), in_channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature: Tensor, size: Optional[Tuple[int, int]] = None) -> Tensor:
        if size is not None and feature.shape[-2:] != size:
            feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        return self.proj(self.act(self.norm(feature)))


class AdaptivePyramidFusion(nn.Module):
    """APF: softmax weight detector-neck scales and fuse them into 128 channels."""

    def __init__(self, num_levels: int = 3, channels: int = 128, hidden_dim: int = 256, norm_groups: int = 32) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.router = nn.Sequential(
            nn.Linear(num_levels * channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_levels),
            nn.Softmax(dim=-1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(num_levels * channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(channels, norm_groups), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, detector_features: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        if len(detector_features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} detector neck levels, got {len(detector_features)}")
        pooled = torch.cat([feature.mean(dim=(-2, -1)) for feature in detector_features], dim=1)
        weights = self.router(pooled)
        weighted = [feature * weights[:, level].view(-1, 1, 1, 1) for level, feature in enumerate(detector_features)]
        return self.fuse(torch.cat(weighted, dim=1)), weights


class ScaleLinkTransformer(nn.Module):
    """SLT cross-attention with decoder features as queries and detector features as keys/values."""

    def __init__(self, channels: int = 128, num_heads: int = 8, norm_groups: int = 32) -> None:
        super().__init__()
        self.q_norm = nn.GroupNorm(_num_groups(channels, norm_groups), channels)
        self.kv_norm = nn.GroupNorm(_num_groups(channels, norm_groups), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, decoder_feature: Tensor, detector_feature: Tensor) -> Tensor:
        bsz, channels, height, width = decoder_feature.shape
        if detector_feature.shape[-2:] != (height, width):
            detector_feature = F.interpolate(detector_feature, size=(height, width), mode="bilinear", align_corners=False)
        query = self.q_norm(decoder_feature).flatten(2).transpose(1, 2)
        key_value = self.kv_norm(detector_feature).flatten(2).transpose(1, 2)
        attended, _ = self.attn(query, key_value, key_value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(bsz, channels, height, width)
        return decoder_feature + self.out(attended)


class ACDM(nn.Module):
    """Attention-guided cross-scale detector modulation for decoder features."""

    def __init__(
        self,
        decoder_channels: Sequence[int],
        detector_neck_channels: Sequence[int] = (256, 512, 1024),
        hidden_channels: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if len(detector_neck_channels) != 3:
            raise ValueError("ACDM expects three YOLOv11n detector neck levels by default.")
        self.decoder_labs = nn.ModuleList(LevelAdapterBlock(channels, hidden_channels) for channels in decoder_channels)
        self.detector_labs = nn.ModuleList(LevelAdapterBlock(channels, hidden_channels) for channels in detector_neck_channels)
        self.apf = AdaptivePyramidFusion(len(detector_neck_channels), hidden_channels)
        self.slt = nn.ModuleList(ScaleLinkTransformer(hidden_channels, num_heads) for _ in decoder_channels)
        self.out = nn.ModuleList(zero_module(nn.Conv2d(hidden_channels, channels, kernel_size=1)) for channels in decoder_channels)

    def forward(self, decoder_features: Sequence[Tensor], detector_neck_features: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
        corrections: List[Tensor] = []
        weights: Optional[Tensor] = None
        for level, decoder_feature in enumerate(decoder_features):
            size = decoder_feature.shape[-2:]
            adapted_decoder = self.decoder_labs[level](decoder_feature)
            adapted_detector = [lab(feature, size=size) for lab, feature in zip(self.detector_labs, detector_neck_features)]
            fused_detector, weights = self.apf(adapted_detector)
            linked = self.slt[level](adapted_decoder, fused_detector)
            corrections.append(self.out[level](linked))
        return corrections, weights if weights is not None else decoder_features[0].new_zeros((decoder_features[0].shape[0], 3))


class ACDMDecoderBranch(nn.Module):
    """Trainable copied UNet decoder branch that emits zero-conv decoder residual corrections."""

    def __init__(self, unet: nn.Module, detector_neck_channels: Sequence[int] = (256, 512, 1024), hidden_channels: int = 128) -> None:
        super().__init__()
        self.up_blocks = copy.deepcopy(unet.up_blocks)
        self.acdm = ACDM(tuple(unet.config.block_out_channels[::-1]), detector_neck_channels, hidden_channels)
        self.freeze_transformer_blocks()

    def freeze_transformer_blocks(self) -> None:
        for name, parameter in self.up_blocks.named_parameters():
            parameter.requires_grad = not any(token in name.lower() for token in ("attn", "attention", "transformer"))

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, decoder_features: Sequence[Tensor], detector_neck_features: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
        # The copied up_blocks are owned here for checkpoint parity with the frozen UNet decoder. Feature capture/injection
        # is performed by the pipeline around the frozen UNet, then ACDM converts branch features to zero-conv residuals.
        return self.acdm(decoder_features, detector_neck_features)
