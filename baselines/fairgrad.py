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
import numpy as np
from .base import BaseProcessor

class FairGradProcessor(BaseProcessor):

    def __init__(self, task_names: List[str], device: str='cuda', alpha: float=0.5, grad_clip_norm: float=1.0, eps: float=1e-08):
        super().__init__(task_names, device)
        self.alpha = alpha
        self.grad_clip_norm = grad_clip_norm
        self.eps = eps
        self.step_count = 0

    def requires_manual_backward(self) -> bool:
        return True

    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        return sum(task_losses.values())

    def backward_and_step(self, task_losses: Dict[str, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.step_count += 1
        optimizer.zero_grad(set_to_none=True)
        valid_task_losses = {}
        total_loss_value = 0.0
        for (task, loss) in task_losses.items():
            if loss is None:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[FairGrad] Step {self.step_count}: Task {task} loss is NaN/Inf, skipping')
                continue
            valid_task_losses[task] = loss
            total_loss_value += loss.item()
        if not valid_task_losses:
            print(f'[FairGrad] Step {self.step_count}: No valid task losses')
            optimizer.step()
            return {'total_loss': total_loss_value}
        task_grads = []
        task_names = []
        param_shapes = []
        param_numels = []
        for p in model.parameters():
            param_shapes.append(p.shape)
            param_numels.append(p.numel())
        total_params = sum(param_numels)
        for (task, loss) in valid_task_losses.items():
            try:
                model.zero_grad()
                loss.backward(retain_graph=True)
                grad_flat = torch.zeros(total_params, device=self.device)
                offset = 0
                for (i, p) in enumerate(model.parameters()):
                    numel = param_numels[i]
                    if p.grad is not None:
                        grad_segment = p.grad.detach().flatten()
                        if torch.isnan(grad_segment).any() or torch.isinf(grad_segment).any():
                            print(f'[FairGrad] Step {self.step_count}: Task {task} grad contains NaN/Inf')
                            grad_segment = torch.zeros_like(grad_segment)
                        grad_norm = grad_segment.norm()
                        if grad_norm > self.grad_clip_norm:
                            grad_segment = grad_segment * (self.grad_clip_norm / grad_norm)
                        grad_flat[offset:offset + numel] = grad_segment
                    offset += numel
                grad_norm = grad_flat.norm().item()
                if grad_norm < 1e-08:
                    print(f'[FairGrad] Step {self.step_count}: Task {task} grad is near zero')
                    continue
                task_grads.append(grad_flat)
                task_names.append(task)
            except Exception as e:
                print(f'[FairGrad] Step {self.step_count}: Error computing grad for task {task}: {e}')
                continue
        if not task_grads:
            print(f'[FairGrad] Step {self.step_count}: No valid gradients')
            optimizer.step()
            return {'total_loss': total_loss_value}
        num_tasks = len(task_grads)
        if num_tasks == 1:
            weighted_grad = task_grads[0]
        else:
            grads_matrix = torch.stack(task_grads, dim=1)
            try:
                GG = torch.mm(grads_matrix.t(), grads_matrix)
                GG_reg = GG + self.eps * torch.eye(num_tasks, device=self.device)
                grad_norms = torch.tensor([g.norm().item() for g in task_grads], device=self.device)
                avg_norm = grad_norms.mean()
                if avg_norm > self.eps:
                    w = avg_norm / (grad_norms + self.eps)
                    w = w / w.sum()
                else:
                    w = torch.ones(num_tasks, device=self.device) / num_tasks
                weighted_grad = torch.zeros_like(grads_matrix[:, 0])
                for i in range(num_tasks):
                    weighted_grad += w[i] * grads_matrix[:, i]
            except Exception as e:
                print(f'[FairGrad] Step {self.step_count}: FairGrad weight computation failed: {e}')
                weighted_grad = torch.stack(task_grads).mean(dim=0)
        if torch.isnan(weighted_grad).any() or torch.isinf(weighted_grad).any():
            print(f'[FairGrad] Step {self.step_count}: Weighted grad contains NaN/Inf, using average')
            weighted_grad = torch.stack(task_grads).mean(dim=0)
        model.zero_grad()
        offset = 0
        for (i, p) in enumerate(model.parameters()):
            numel = param_numels[i]
            if numel > 0 and offset + numel <= len(weighted_grad):
                grad_segment = weighted_grad[offset:offset + numel].view(param_shapes[i])
                if p.grad is None:
                    p.grad = grad_segment.clone()
                else:
                    p.grad.copy_(grad_segment)
                offset += numel
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
        has_nan = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f'[FairGrad] Step {self.step_count}: Final grad contains NaN/Inf, setting to zero')
                    p.grad = torch.zeros_like(p.grad)
                    has_nan = True
        try:
            optimizer.step()
        except Exception as e:
            print(f'[FairGrad] Step {self.step_count}: Optimizer step failed: {e}')
        return {'total_loss': total_loss_value}

    def get_extra_params(self) -> List[nn.Parameter]:
        return []

    def state_dict(self) -> Dict:
        return {'alpha': self.alpha, 'step_count': self.step_count}

    def load_state_dict(self, state: Dict) -> None:
        if 'alpha' in state:
            self.alpha = state['alpha']
        if 'step_count' in state:
            self.step_count = state.get('step_count', 0)
