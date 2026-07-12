# Copyright 2026 PyramidDiff authors.
# Licensed under the Apache License, Version 2.0.
"""GroundNet, MSOP, and DA-ACL building blocks for PyramidDiff.

This module contains lightweight, composable PyTorch modules that implement the
paper method used by PyramidDiff:

* GLIGEN-style object grounding tokens from CLIP object embeddings and boxes.
* Multi-Scale Object Pyramid (MSOP) routing for scale-adaptive GroundNet stages.
* Detection-Aware Annotation Consistency Learning (DA-ACL) losses.

The classes are intentionally independent of a specific detector or diffusion
pipeline so they can be used by training scripts, ControlNet-style side branches,
or unit tests without downloading pretrained weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FourierBoxEmbedder(nn.Module):
    """Parameter-free Fourier encoding for normalized boxes.

    Input boxes are expected in ``(x1, y1, x2, y2)`` or normalized equivalent
    coordinates. With 8 frequencies this returns 4 * 8 * 2 = 64 dimensions,
    matching the GroundNet/GLIGEN token construction described by PyramidDiff.
    """

    def __init__(self, num_frequencies: int = 8, temperature: float = 100.0) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.temperature = temperature
        frequencies = temperature ** (torch.arange(num_frequencies, dtype=torch.float32) / num_frequencies)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return 4 * self.num_frequencies * 2

    def forward(self, boxes: Tensor) -> Tensor:
        if boxes.shape[-1] != 4:
            raise ValueError(f"Expected boxes with last dimension 4, got {tuple(boxes.shape)}")
        angles = boxes.to(self.frequencies.dtype).unsqueeze(-1) * self.frequencies
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return encoded.flatten(start_dim=-2).to(dtype=boxes.dtype)


class GroundingTokenizer(nn.Module):
    """Construct GroundNet object tokens from object text embeddings and boxes."""

    def __init__(
        self,
        text_dim: int = 768,
        hidden_dim: int = 512,
        token_dim: int = 768,
        num_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.box_embedder = FourierBoxEmbedder(num_frequencies=num_frequencies)
        self.proj = nn.Sequential(
            nn.Linear(text_dim + self.box_embedder.output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, object_text_embeds: Tensor, boxes: Tensor) -> Tensor:
        if object_text_embeds.shape[:-1] != boxes.shape[:-1]:
            raise ValueError(
                "Object text embeddings and boxes must share batch/object dimensions: "
                f"{tuple(object_text_embeds.shape)} vs {tuple(boxes.shape)}"
            )
        box_embeds = self.box_embedder(boxes)
        return self.proj(torch.cat((object_text_embeds, box_embeds), dim=-1))


class GatedSelfAttention(nn.Module):
    """GLIGEN-style gated self-attention over visual and grounding tokens."""

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, visual_tokens: Tensor, grounding_tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tokens = torch.cat((visual_tokens, grounding_tokens), dim=1)
        attended, _ = self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens), key_padding_mask=key_padding_mask)
        attended_visual = attended[:, : visual_tokens.shape[1]]
        return visual_tokens + torch.tanh(self.gamma) * attended_visual


class MultiScaleObjectPyramid(nn.Module):
    """Learned scale-adaptive routing of object tokens across GroundNet levels."""

    def __init__(self, token_dim: int, level_channels: Sequence[int], router_hidden_dim: int = 256) -> None:
        super().__init__()
        self.level_channels = tuple(level_channels)
        self.router = nn.Sequential(
            nn.Linear(token_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, len(self.level_channels)),
        )
        self.level_projs = nn.ModuleList(nn.Linear(token_dim, channels) for channels in self.level_channels)
        self.up_projs = nn.ModuleList(
            nn.Conv2d(self.level_channels[i + 1], self.level_channels[i], kernel_size=1)
            for i in range(len(self.level_channels) - 1)
        )
        self.down_projs = nn.ModuleList(
            nn.Conv2d(self.level_channels[i - 1], self.level_channels[i], kernel_size=1)
            for i in range(1, len(self.level_channels))
        )

    def route_tokens(self, grounding_tokens: Tensor, object_mask: Optional[Tensor] = None) -> Tensor:
        logits = self.router(grounding_tokens)
        if object_mask is not None:
            logits = logits.masked_fill(~object_mask.bool().unsqueeze(-1), torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def forward(
        self, features: Sequence[Tensor], grounding_tokens: Tensor, object_mask: Optional[Tensor] = None
    ) -> Tuple[List[Tensor], Tensor]:
        if len(features) != len(self.level_channels):
            raise ValueError(f"Expected {len(self.level_channels)} feature levels, got {len(features)}")
        routes = self.route_tokens(grounding_tokens, object_mask)
        conditioned: List[Tensor] = []
        for level, feature in enumerate(features):
            projected = self.level_projs[level](grounding_tokens)
            weights = routes[..., level : level + 1]
            if object_mask is not None:
                weights = weights * object_mask.bool().unsqueeze(-1).to(weights.dtype)
            token_bias = (weights * projected).sum(dim=1).unsqueeze(-1).unsqueeze(-1)
            conditioned.append(feature + token_bias.to(dtype=feature.dtype))

        fused: List[Tensor] = []
        for level, feature in enumerate(conditioned):
            update = feature
            if level + 1 < len(conditioned):
                up = F.interpolate(conditioned[level + 1], size=feature.shape[-2:], mode="nearest")
                update = update + self.up_projs[level](up).to(dtype=feature.dtype)
            if level > 0:
                down = F.interpolate(conditioned[level - 1], size=feature.shape[-2:], mode="area")
                update = update + self.down_projs[level - 1](down).to(dtype=feature.dtype)
            fused.append(update)
        return fused, routes


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None]
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


@dataclass
class DAACLOutput:
    loss: Tensor
    loc_loss: Tensor
    cls_loss: Tensor
    miss_loss: Tensor
    num_matched: int




@dataclass
class DetectionPredictions:
    """Dense detector predictions consumed by DA-ACL.

    Attributes:
        boxes: Predicted boxes in ``xyxy`` image coordinates, shaped ``(M, 4)``.
        scores: Objectness or category confidence for each prediction, shaped ``(M,)``.
        class_probs: Per-class probabilities, shaped ``(M, C)``.
        raw_outputs: Optional detector-native raw tensors for debugging or custom matching.
    """

    boxes: Tensor
    scores: Tensor
    class_probs: Tensor
    raw_outputs: Optional[object] = None


class YOLOv11xVerificationDetector(nn.Module):
    """Frozen YOLO11x verification network for PyramidDiff DA-ACL.

    The current Ultralytics naming convention for the requested "YOLOv11x" model
    is ``yolo11x.pt``. This wrapper lazy-loads Ultralytics so importing diffusers
    does not require the optional detector dependency. The detector is kept in
    evaluation mode with all parameters frozen and returns prediction tensors in
    the format expected by :class:`DetectionAwareAnnotationConsistencyLoss`.

    Set ``pre_nms=True`` to request raw model-head outputs for DA-ACL research
    workflows that want dense candidates before NMS. Because Ultralytics can
    change raw head formats across releases, raw outputs are also returned in
    ``DetectionPredictions.raw_outputs`` for project-specific decoding.
    """

    def __init__(
        self,
        model_name: str = "yolo11x.pt",
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.001,
        pre_nms: bool = False,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLOv11xVerificationDetector requires the optional 'ultralytics' package. "
                "Install it with `pip install ultralytics` and ensure the yolo11x.pt weights are available."
            ) from exc

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.pre_nms = pre_nms
        self.yolo = YOLO(model_name)
        self.detector = self.yolo.model
        if device is not None:
            self.detector.to(device)
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad_(False)

    @property
    def names(self):
        return getattr(self.yolo, "names", getattr(self.detector, "names", None))

    def train(self, mode: bool = True):  # noqa: D401 - intentionally pins eval mode
        """Keep the verification detector frozen/eval even when parent modules train."""
        super().train(False)
        self.detector.eval()
        return self

    def forward(self, images: Tensor) -> List[DetectionPredictions]:
        with torch.no_grad():
            if self.pre_nms:
                raw_outputs = self.detector(images)
                return self._decode_raw_outputs(raw_outputs, images)

            results = self.yolo.predict(images, conf=self.confidence_threshold, verbose=False)
            predictions: List[DetectionPredictions] = []
            num_classes = len(self.names) if self.names is not None else 0
            for result in results:
                boxes_obj = result.boxes
                boxes = boxes_obj.xyxy.to(images.device)
                scores = boxes_obj.conf.to(images.device)
                cls = boxes_obj.cls.long().to(images.device)
                class_probs = boxes.new_zeros((boxes.shape[0], num_classes))
                if boxes.shape[0] > 0 and num_classes > 0:
                    class_probs[torch.arange(boxes.shape[0], device=images.device), cls] = scores
                predictions.append(DetectionPredictions(boxes=boxes, scores=scores, class_probs=class_probs, raw_outputs=result))
            return predictions

    def _decode_raw_outputs(self, raw_outputs: object, images: Tensor) -> List[DetectionPredictions]:
        """Best-effort raw-output adapter for pre-NMS DA-ACL experiments.

        Ultralytics raw outputs are version-dependent. When the common tensor
        layout ``(B, 4 + C, M)`` or ``(B, M, 4 + C)`` is observed, this method
        decodes boxes and probabilities directly. Otherwise it returns empty
        decoded tensors while preserving ``raw_outputs`` for custom decoding.
        """
        raw = raw_outputs[0] if isinstance(raw_outputs, (tuple, list)) else raw_outputs
        batch_size = images.shape[0]
        predictions: List[DetectionPredictions] = []
        if torch.is_tensor(raw) and raw.dim() == 3:
            if raw.shape[1] < raw.shape[2]:
                raw = raw.transpose(1, 2)
            boxes = raw[..., :4]
            class_probs = raw[..., 4:].sigmoid()
            scores = class_probs.max(dim=-1).values if class_probs.numel() else boxes.new_zeros(boxes.shape[:2])
            for batch_idx in range(raw.shape[0]):
                keep = scores[batch_idx] >= self.confidence_threshold
                predictions.append(
                    DetectionPredictions(
                        boxes=boxes[batch_idx, keep],
                        scores=scores[batch_idx, keep],
                        class_probs=class_probs[batch_idx, keep],
                        raw_outputs=raw_outputs,
                    )
                )
            return predictions

        for _ in range(batch_size):
            predictions.append(
                DetectionPredictions(
                    boxes=images.new_zeros((0, 4)),
                    scores=images.new_zeros((0,)),
                    class_probs=images.new_zeros((0, 0)),
                    raw_outputs=raw_outputs,
                )
            )
        return predictions


class DetectionAwareAnnotationConsistencyLoss(nn.Module):
    """DA-ACL task-aligned matching and annotation-consistency objective."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 6.0,
        lambda_loc: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_miss: float = 1.0,
        min_assignment_score: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_loc = lambda_loc
        self.lambda_cls = lambda_cls
        self.lambda_miss = lambda_miss
        self.min_assignment_score = min_assignment_score

    def forward(self, pred_boxes: Tensor, pred_probs: Tensor, target_boxes: Tensor, target_classes: Tensor) -> DAACLOutput:
        device = pred_boxes.device
        zero = pred_boxes.sum() * 0.0
        if target_boxes.numel() == 0:
            return DAACLOutput(zero, zero, zero, zero, 0)
        if pred_boxes.numel() == 0:
            miss = target_boxes.new_tensor(1.0)
            return DAACLOutput(self.lambda_miss * miss, zero, zero, miss, 0)

        ious = box_iou(target_boxes, pred_boxes)
        cls_conf = pred_probs[:, target_classes.long()].transpose(0, 1).clamp(min=1e-8)
        scores = cls_conf.pow(self.alpha) * ious.clamp(min=1e-8).pow(self.beta)

        matched_targets: List[int] = []
        matched_preds: List[int] = []
        for flat_idx in torch.argsort(scores.flatten(), descending=True):
            target_idx = torch.div(flat_idx, scores.shape[1], rounding_mode="floor").item()
            pred_idx = (flat_idx % scores.shape[1]).item()
            if scores[target_idx, pred_idx] < self.min_assignment_score:
                break
            if target_idx not in matched_targets and pred_idx not in matched_preds:
                matched_targets.append(target_idx)
                matched_preds.append(pred_idx)

        if matched_targets:
            target_index = torch.tensor(matched_targets, device=device, dtype=torch.long)
            pred_index = torch.tensor(matched_preds, device=device, dtype=torch.long)
            loc_loss = F.smooth_l1_loss(pred_boxes[pred_index], target_boxes[target_index], reduction="mean")
            cls_loss = F.nll_loss(pred_probs[pred_index].clamp(min=1e-8).log(), target_classes[target_index].long())
        else:
            loc_loss = zero
            cls_loss = zero
        miss_loss = target_boxes.new_tensor((target_boxes.shape[0] - len(matched_targets)) / max(target_boxes.shape[0], 1))
        loss = self.lambda_loc * loc_loss + self.lambda_cls * cls_loss + self.lambda_miss * miss_loss
        return DAACLOutput(loss, loc_loss, cls_loss, miss_loss, len(matched_targets))
