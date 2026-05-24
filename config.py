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
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BackboneConfig:
    input_dim: int = 17
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    model_name: str = 'resnet50'
    pretrained: bool = True
    allow_random_init: bool = False

@dataclass
class BandControllerConfig:
    r_star: float = 0.16
    r_min: float = 0.15
    r_max: float = 0.35
    eta: float = 0.001
    lambda_init: float = 0.1
    lambda_min: float = 0.01
    lambda_max: float = 1.0
    warmup_steps: int = 100
    ema_alpha: float = 0.1

@dataclass
class LipschitzConfig:
    delta: float = 0.1
    mu: float = 0.05
    eps_d: float = 0.001
    k_prototypes: int = 5
    ema_alpha: float = 0.1

@dataclass
class TrainingConfig:
    seed: int = 42
    data_seed: Optional[int] = None
    device: str = 'cuda'
    num_epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 1e-05
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = 'fp16'
    force_backbone_fp32: bool = True
    log_interval: int = 100
    eval_interval: int = 1
    warn_threshold: int = 5
    early_stop_patience: int = 30
    early_stop_min_delta: float = 0.0005
    save_best_macro: bool = True
    multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None

@dataclass
class ExperimentConfig:
    dataset_name: str = 'mimic3'
    data_root: str = './data/mimic3/bench'
    tasks_enabled: List[str] = field(default_factory=lambda : ['mortality', 'decomp', 'los', 'phenotype'])
    phenotype_num_classes: int = 25
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    band_controller: BandControllerConfig = field(default_factory=BandControllerConfig)
    lipschitz: LipschitzConfig = field(default_factory=LipschitzConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = './outputs'
    experiment_name: str = 'dynamic_lipschitz_mtl'
