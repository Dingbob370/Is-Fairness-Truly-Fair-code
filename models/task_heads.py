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
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MortalityHead(nn.Module):

    def __init__(self, input_dim: int=256, hidden_dim: int=128, dropout: float=0.3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)
        probs = torch.sigmoid(logits).squeeze(-1)
        return (logits, probs)

class DecompHead(nn.Module):

    def __init__(self, input_dim: int=256, hidden_dim: int=128, dropout: float=0.3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z_seq)
        probs = torch.sigmoid(logits).squeeze(-1)
        return (logits, probs)

class LOSHead(nn.Module):

    def __init__(self, input_dim: int=256, hidden_dim: int=128, num_buckets: int=10, bucket_boundaries: Optional[list]=None, dropout: float=0.3):
        super().__init__()
        self.use_discretization = True
        self.num_buckets = num_buckets
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_buckets))
        if bucket_boundaries is None:
            self.bucket_boundaries = [1, 2, 3, 4, 5, 6, 7, 8, 14]
        else:
            self.bucket_boundaries = bucket_boundaries

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)
        probs = F.softmax(logits, dim=-1)
        return (logits, probs)

    def discretize_target(self, los_days: torch.Tensor) -> torch.Tensor:
        los_days = los_days.squeeze(-1) if los_days.dim() > 1 else los_days
        bucket_idx = torch.zeros_like(los_days, dtype=torch.long)
        for boundary in self.bucket_boundaries:
            bucket_idx = bucket_idx + (los_days >= boundary).to(dtype=bucket_idx.dtype)
        return bucket_idx.clamp(max=self.num_buckets - 1)

    def bucket_to_days(self, bucket_idx: torch.Tensor) -> torch.Tensor:
        boundaries: list[Any] = [0] + list(self.bucket_boundaries) + [float('inf')]
        centers = []
        for i in range(len(boundaries) - 1):
            if boundaries[i + 1] == float('inf'):
                centers.append(boundaries[i] + 15)
            else:
                centers.append((boundaries[i] + boundaries[i + 1]) / 2)
        centers_t = torch.tensor(centers, device=bucket_idx.device, dtype=torch.float32)
        return centers_t[bucket_idx]

class PhenotypeHead(nn.Module):

    def __init__(self, input_dim: int=256, hidden_dim: int=128, num_phenotypes: int=25, dropout: float=0.3):
        super().__init__()
        self.num_phenotypes = num_phenotypes
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_phenotypes))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(z)
        probs = torch.sigmoid(logits)
        return (logits, probs)

class SegmentationHead(nn.Module):

    def __init__(self, input_dim: int=512, num_classes: int=13):
        super().__init__()
        self.num_classes = int(num_classes)
        self.decoder = nn.Sequential(nn.Conv2d(input_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, z: torch.Tensor, return_features: bool=False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.decoder(z)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        if return_features:
            return (logits, probs, features)
        return (logits, probs)

class DepthHead(nn.Module):

    def __init__(self, input_dim: int=512):
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(input_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.regressor = nn.Conv2d(64, 1, 1)

    def forward(self, z: torch.Tensor, return_features: bool=False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.decoder(z)
        pred = self.regressor(features)
        if return_features:
            return (pred, pred, features)
        return (pred, pred)

class NormalHead(nn.Module):

    def __init__(self, input_dim: int=512):
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(input_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.regressor = nn.Conv2d(64, 3, 1)

    def forward(self, z: torch.Tensor, return_features: bool=False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.decoder(z)
        pred = self.regressor(features)
        pred = F.normalize(pred, p=2, dim=1)
        if return_features:
            return (pred, pred, features)
        return (pred, pred)

def build_task_heads(dataset_name: str, tasks_enabled: list, backbone_dim: int, **kwargs) -> nn.ModuleDict:
    heads = nn.ModuleDict()
    if dataset_name in ('mimic3', 'eicu'):
        task_head_classes = {'mortality': MortalityHead, 'decomp': DecompHead, 'los': LOSHead, 'phenotype': PhenotypeHead}
    elif dataset_name == 'nyuv2':
        task_head_classes = {'seg': SegmentationHead, 'depth': DepthHead, 'normal': NormalHead}
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    for task in tasks_enabled:
        if task not in task_head_classes:
            raise ValueError(f'Unknown task: {task}')
        heads[task] = task_head_classes[task](input_dim=backbone_dim, **kwargs.get(task, {}))
    return heads
