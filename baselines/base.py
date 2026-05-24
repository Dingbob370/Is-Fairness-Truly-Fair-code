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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn

class BaseProcessor(ABC):

    def __init__(self, task_names: List[str], device: str='cuda'):
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device

    @abstractmethod
    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def requires_manual_backward(self) -> bool:
        return False

    def backward_and_step(self, task_losses: Dict[str, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        total_loss = self.process(task_losses, model)
        total_loss.backward()
        optimizer.step()
        return {'total_loss': total_loss.item()}

    def get_extra_params(self) -> List[nn.Parameter]:
        return []

    def to(self, device: str) -> 'BaseProcessor':
        self.device = device
        for p in self.get_extra_params():
            p.data = p.data.to(device)
            if p.grad is not None:
                p.grad = p.grad.to(device)
        return self

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict) -> None:
        pass
