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

class UncertaintyProcessor(BaseProcessor):

    def __init__(self, task_names: List[str], device: str='cuda'):
        super().__init__(task_names, device)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks, device=device))

    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        for (i, task) in enumerate(self.task_names):
            if task not in task_losses:
                continue
            loss = task_losses[task]
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + 0.5 * self.log_vars[i]
            total_loss = total_loss + weighted_loss
        return total_loss

    def get_extra_params(self) -> List[nn.Parameter]:
        return [self.log_vars]

    def get_weights(self) -> Dict[str, float]:
        weights = {}
        for (i, task) in enumerate(self.task_names):
            weights[task] = float(torch.exp(-self.log_vars[i]).item())
        return weights

    def state_dict(self) -> Dict:
        return {'log_vars': self.log_vars.data.clone()}

    def load_state_dict(self, state: Dict) -> None:
        if 'log_vars' in state:
            self.log_vars.data = state['log_vars'].to(self.device)
