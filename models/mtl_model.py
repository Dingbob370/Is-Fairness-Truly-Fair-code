# Copyright 2026 Junbo Ding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .lipschitz_regularizer import LipschitzRegularizer
from .task_heads import build_task_heads

class DynamicLipschitzMTL(nn.Module):

    def __init__(self, dataset_name: str='mimic3', tasks_enabled: Optional[List[str]]=None, force_backbone_fp32: bool=False, backbone_config: Optional[Dict]=None, head_config: Optional[Dict]=None, lip_config: Optional[Dict]=None, nyuv2_proxy_stats: Optional[Dict[str, Dict[str, float]]]=None, nyuv2_proxy_stats_path: Optional[str]=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.tasks_enabled = tasks_enabled or ['mortality', 'decomp', 'los', 'phenotype']
        self.force_backbone_fp32 = bool(force_backbone_fp32)
        self.nyuv2_proxy_stats_path = nyuv2_proxy_stats_path
        backbone_config = backbone_config or {}
        head_config = head_config or {}
        lip_config = lip_config or {}
        self.backbone = build_backbone(dataset_name, **backbone_config)
        self.backbone_dim = getattr(self.backbone, 'output_dim', None)
        if self.backbone_dim is None:
            raise RuntimeError('Backbone must define output_dim')
        self.task_heads = build_task_heads(dataset_name, self.tasks_enabled, self.backbone_dim, **head_config)
        self.lip_regularizer = LipschitzRegularizer(**lip_config)
        self.nyuv2_proxy_stats = self._resolve_nyuv2_proxy_stats(nyuv2_proxy_stats=nyuv2_proxy_stats, nyuv2_proxy_stats_path=nyuv2_proxy_stats_path)

    @staticmethod
    def _resolve_nyuv2_proxy_stats(*, nyuv2_proxy_stats: Optional[Dict[str, Dict[str, float]]], nyuv2_proxy_stats_path: Optional[str]) -> Dict[str, Dict[str, float]]:
        stats_source = nyuv2_proxy_stats
        if stats_source is None and nyuv2_proxy_stats_path:
            stats_path = os.path.abspath(os.path.expanduser(str(nyuv2_proxy_stats_path)))
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f'NYUv2 proxy stats file not found: {stats_path}')
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats_source = json.load(f)
        if not isinstance(stats_source, dict):
            return {}
        task_stats = stats_source.get('tasks', stats_source)
        if not isinstance(task_stats, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for (task, stats) in task_stats.items():
            if not isinstance(stats, dict):
                continue
            q10 = stats.get('q10')
            q90 = stats.get('q90')
            if q10 is None or q90 is None:
                continue
            normalized[str(task)] = {'q10': float(q10), 'q90': float(q90)}
        return normalized

    @staticmethod
    def _resize_spatial_tensor(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _prepare_spatial_mask(mask: Optional[torch.Tensor], size: Tuple[int, int]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 4 and mask.size(1) > 1:
            mask = mask.any(dim=1)
        elif mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]
        mask = mask.bool()
        if mask.dim() != 3:
            raise ValueError(f'Expected spatial mask with ndim=3 after squeeze, got shape={tuple(mask.shape)}')
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).bool()
        return mask

    @staticmethod
    def _masked_spatial_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=(-2, -1))
        mask_f = mask.to(dtype=values.dtype)
        denom = mask_f.sum(dim=(-2, -1)).clamp(min=1.0)
        return (values * mask_f).sum(dim=(-2, -1)) / denom

    @classmethod
    def _feature_magnitude_score(cls, feature_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        spatial_response = torch.linalg.vector_norm(feature_map, ord=2, dim=1)
        return cls._masked_spatial_mean(spatial_response, mask)

    def _squash_nyuv2_proxy(self, task: str, raw_score: torch.Tensor) -> torch.Tensor:
        if task == 'seg':
            return raw_score.clamp(0.0, 1.0)
        stats = self.nyuv2_proxy_stats.get(task)
        if stats is None:
            return raw_score / (1.0 + raw_score)
        q10 = raw_score.new_tensor(float(stats['q10']))
        q90 = raw_score.new_tensor(float(stats['q90']))
        denom = (q90 - q10).clamp(min=1e-06)
        return ((raw_score - q10) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _align_mask_to_loss(mask: torch.Tensor, loss: torch.Tensor, task: str) -> torch.Tensor:
        mask_f = mask.to(dtype=loss.dtype)
        while mask_f.dim() > loss.dim():
            squeezed = False
            for dim in range(1, mask_f.dim()):
                if mask_f.size(dim) == 1:
                    mask_f = mask_f.squeeze(dim)
                    squeezed = True
                    break
            if not squeezed:
                break
        while mask_f.dim() < loss.dim():
            mask_f = mask_f.unsqueeze(1)
        if mask_f.shape != loss.shape:
            try:
                mask_f = mask_f.expand_as(loss)
            except RuntimeError as e:
                raise ValueError(f"Mask shape {tuple(mask_f.shape)} is not compatible with loss shape {tuple(loss.shape)} for task '{task}'") from e
        return mask_f

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor]=None, y_mask: Optional[Dict[str, torch.Tensor]]=None, return_embeddings: bool=False) -> Dict:
        logits: Dict[str, torch.Tensor] = {}
        probs: Dict[str, torch.Tensor] = {}
        confidences: Dict[str, torch.Tensor] = {}
        raw_confidences: Dict[str, torch.Tensor] = {}
        if self.dataset_name in ('mimic3', 'eicu'):
            if seq_mask is None:
                raise ValueError('The decomp task requires sequence masks and sequence representations.')
            if self.force_backbone_fp32:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    z = self.backbone(x.float(), seq_mask)
            else:
                z = self.backbone(x, seq_mask)
            z_seq = None
            if 'decomp' in self.tasks_enabled:
                if not hasattr(self.backbone, 'forward_sequence'):
                    raise RuntimeError('The decomp task requires sequence masks and sequence representations.')
                if self.force_backbone_fp32:
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        z_seq = self.backbone.forward_sequence(x.float(), seq_mask)
                else:
                    z_seq = self.backbone.forward_sequence(x, seq_mask)
            for task in self.tasks_enabled:
                if task == 'decomp':
                    if z_seq is None:
                        raise RuntimeError('The decomp task requires sequence masks and sequence representations.')
                    if y_mask is None or 'decomp' not in y_mask:
                        raise ValueError('The decomp task requires sequence masks and sequence representations.')
                    (task_logits, task_probs) = self.task_heads[task](z_seq)
                    valid_decomp = seq_mask & y_mask['decomp']
                    v = valid_decomp.to(dtype=task_probs.dtype)
                    conf = (task_probs * v).sum(dim=1) / v.sum(dim=1).clamp(min=1)
                    logits[task] = task_logits
                    probs[task] = task_probs
                    confidences[task] = conf
                    continue
                (task_logits, task_probs) = self.task_heads[task](z)
                logits[task] = task_logits
                probs[task] = task_probs
                if task == 'phenotype':
                    conf = task_probs.max(dim=1)[0]
                elif task == 'los':
                    conf = task_probs.max(dim=-1)[0] if task_probs.dim() > 1 else task_probs
                else:
                    conf = task_probs if task_probs.dim() == 1 else task_probs.squeeze(-1)
                if y_mask is not None and task in y_mask and (y_mask[task] is not None):
                    mask_t = y_mask[task]
                    if task == 'phenotype':
                        valid_sample = mask_t.any(dim=1)
                    elif mask_t.dim() > 1 and mask_t.size(-1) == 1:
                        valid_sample = mask_t.squeeze(-1).bool()
                    else:
                        valid_sample = mask_t.bool()
                    conf = conf * valid_sample.to(dtype=conf.dtype)
                confidences[task] = conf
            outputs = {'logits': logits, 'probs': probs, 'confidences': confidences}
            if return_embeddings:
                outputs['embeddings'] = z
            return outputs
        z_map = self.backbone(x)
        embeddings = F.adaptive_avg_pool2d(z_map, 1).squeeze(-1).squeeze(-1)
        image_size = (x.size(-2), x.size(-1))
        for task in self.tasks_enabled:
            head_outputs = self.task_heads[task](z_map, return_features=True)
            if len(head_outputs) == 3:
                (task_logits, task_probs, task_features) = head_outputs
            else:
                (task_logits, task_probs) = head_outputs
                task_features = task_probs
            task_features = self._resize_spatial_tensor(task_features, image_size)
            if task == 'seg':
                task_logits = self._resize_spatial_tensor(task_logits, image_size)
                task_probs = F.softmax(task_logits, dim=1)
            elif task == 'depth':
                task_logits = self._resize_spatial_tensor(task_logits, image_size)
                task_probs = task_logits
            elif task == 'normal':
                task_logits = self._resize_spatial_tensor(task_logits, image_size)
                task_logits = F.normalize(task_logits, p=2, dim=1)
                task_probs = task_logits
            logits[task] = task_logits
            probs[task] = task_probs
            if task == 'seg':
                valid_mask = self._prepare_spatial_mask(y_mask.get('seg') if y_mask is not None else None, image_size)
                max_probs = task_probs.max(dim=1)[0]
                raw_conf = self._masked_spatial_mean(max_probs, valid_mask)
            else:
                valid_mask = self._prepare_spatial_mask(y_mask.get(task) if y_mask is not None else None, image_size)
                raw_conf = self._feature_magnitude_score(task_features, valid_mask)
            conf = self._squash_nyuv2_proxy(task, raw_conf)
            raw_confidences[task] = raw_conf
            confidences[task] = conf
        outputs = {'logits': logits, 'probs': probs, 'confidences': confidences, 'raw_confidences': raw_confidences}
        if return_embeddings:
            outputs['embeddings'] = embeddings
        return outputs

    def compute_task_losses(self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], y_mask: Dict[str, torch.Tensor], seq_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        task_losses: Dict[str, torch.Tensor] = {}
        for task in self.tasks_enabled:
            if task not in logits:
                continue
            pred = logits[task]
            target = targets.get(task)
            mask = y_mask.get(task) if y_mask is not None else None
            if target is None:
                continue
            if task == 'mortality':
                pred_ = pred.squeeze(-1)
                target_ = target.squeeze(-1) if target.dim() > 1 else target
                loss = F.binary_cross_entropy_with_logits(pred_, target_.float(), reduction='none')
            elif task == 'decomp':
                if seq_mask is None or mask is None:
                    raise ValueError('The decomp task requires sequence masks and sequence representations.')
                valid_decomp = seq_mask & mask
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), target.float(), reduction='none')
                mask = valid_decomp
            elif task == 'los':
                los_head = self.task_heads['los'] if 'los' in self.task_heads else None
                if los_head is None or not getattr(los_head, 'use_discretization', True):
                    raise RuntimeError('LOS uses discretized classification targets in this implementation.')
                bucket_target = los_head.discretize_target(target)
                loss = F.cross_entropy(pred, bucket_target, reduction='none')
                if mask is not None and mask.dim() > 1:
                    mask = mask.squeeze(-1)
            elif task == 'phenotype':
                loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
            elif task == 'seg':
                loss = F.cross_entropy(pred, target.long(), reduction='none', ignore_index=-1)
            elif task == 'depth':
                loss = F.l1_loss(pred, target.float(), reduction='none')
            elif task == 'normal':
                target_norm = F.normalize(target.float(), p=2, dim=1)
                loss = 1.0 - F.cosine_similarity(pred, target_norm, dim=1, eps=1e-06).unsqueeze(1)
            else:
                loss = F.cross_entropy(pred, target, reduction='none')
            if mask is not None:
                mask_f = self._align_mask_to_loss(mask, loss, task)
                denom = mask_f.sum().clamp(min=1.0)
                loss = (loss * mask_f).sum() / denom
            else:
                loss = loss.mean()
            task_losses[task] = loss
        if task_losses:
            total_loss = sum(task_losses.values())
        else:
            total_loss = torch.tensor(0.0, device=next(iter(logits.values())).device)
        return (total_loss, task_losses)

    def compute_lipschitz_loss(self, embeddings: torch.Tensor, confidences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        task_embeddings = {task: embeddings for task in self.tasks_enabled}
        return self.lip_regularizer(task_embeddings, confidences)
