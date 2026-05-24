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
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .base import BaseProcessor

class CATSProcessor(BaseProcessor):

    def __init__(self, task_names: List[str], device: str='cuda', alpha: float=0.1):
        super().__init__(task_names, device)
        self.alpha = alpha
        self.task_loss_ema: Dict[str, float] = {task: 1.0 for task in task_names}
        self.step = 0

    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        self.step += 1
        for (task, loss) in task_losses.items():
            if task in self.task_loss_ema:
                loss_val = loss.detach().item()
                self.task_loss_ema[task] = self.alpha * loss_val + (1 - self.alpha) * self.task_loss_ema[task]
        ema_values = [self.task_loss_ema[t] for t in self.task_names if t in task_losses]
        if not ema_values:
            return sum(task_losses.values())
        mean_ema = sum(ema_values) / len(ema_values)
        weights = {}
        for task in self.task_names:
            if task in task_losses and task in self.task_loss_ema:
                weights[task] = self.task_loss_ema[task] / (mean_ema + 1e-08)
        total_loss = torch.tensor(0.0, device=self.device)
        for (task, loss) in task_losses.items():
            weight = weights.get(task, 1.0)
            total_loss = total_loss + weight * loss
        return total_loss

    def get_weights(self) -> Dict[str, float]:
        ema_values = list(self.task_loss_ema.values())
        mean_ema = sum(ema_values) / len(ema_values) if ema_values else 1.0
        return {task: ema / (mean_ema + 1e-08) for (task, ema) in self.task_loss_ema.items()}

    def state_dict(self) -> Dict:
        return {'task_loss_ema': self.task_loss_ema.copy(), 'step': self.step}

    def load_state_dict(self, state: Dict) -> None:
        if 'task_loss_ema' in state:
            self.task_loss_ema = state['task_loss_ema']
        if 'step' in state:
            self.step = state['step']
