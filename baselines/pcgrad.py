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
import random
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .base import BaseProcessor

class PCGradProcessor(BaseProcessor):

    def __init__(self, task_names: List[str], device: str='cuda', use_grad_clip: bool=True, grad_clip_norm: float=5.0):
        super().__init__(task_names, device)
        self.use_grad_clip = use_grad_clip
        self.grad_clip_norm = grad_clip_norm
        self.step_count = 0
        self.debug = False

    def requires_manual_backward(self) -> bool:
        return True

    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        return sum(task_losses.values())

    def backward_and_step(self, task_losses: Dict[str, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.step_count += 1
        optimizer.zero_grad(set_to_none=True)
        params = list(model.parameters())
        if not params:
            optimizer.step()
            return {'total_loss': 0.0}
        task_list = []
        loss_list = []
        for (task, loss) in task_losses.items():
            if loss is None:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                if self.debug:
                    print(f'[PCGrad] Step {self.step_count}: Task {task} loss is NaN/Inf')
                continue
            task_list.append(task)
            loss_list.append(loss)
        if not task_list:
            if self.debug:
                print(f'[PCGrad] Step {self.step_count}: No valid task losses')
            optimizer.step()
            return {'total_loss': 0.0}
        task_gradients = []
        for (i, (task, loss)) in enumerate(zip(task_list, loss_list)):
            try:
                model.zero_grad()
                loss.backward(retain_graph=i < len(task_list) - 1)
                task_grad = []
                for p in params:
                    if p.grad is not None:
                        grad = p.grad.detach().clone()
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            if self.debug:
                                print(f'[PCGrad] Step {self.step_count}: Task {task} grad contains NaN/Inf')
                            grad = torch.zeros_like(grad)
                        task_grad.append(grad.flatten())
                    else:
                        task_grad.append(torch.zeros(p.numel(), device=self.device))
                task_grad_flat = torch.cat(task_grad)
                task_gradients.append(task_grad_flat)
                if self.debug and self.step_count <= 10:
                    grad_norm = task_grad_flat.norm().item()
                    print(f'[PCGrad Debug] Task {task}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}')
            except Exception as e:
                if self.debug:
                    print(f'[PCGrad] Step {self.step_count}: Error computing grad for task {task}: {e}')
                task_gradients.append(torch.cat([torch.zeros(p.numel(), device=self.device) for p in params]))
        if len(task_gradients) >= 2:
            task_indices = list(range(len(task_gradients)))
            random.shuffle(task_indices)
            pc_gradients = [g.clone() for g in task_gradients]
            conflict_count = 0
            for i in task_indices:
                for j in task_indices:
                    if i == j:
                        continue
                    g_i = pc_gradients[i]
                    g_j = task_gradients[j]
                    dot_product = torch.dot(g_i, g_j)
                    if dot_product < 0:
                        conflict_count += 1
                        g_j_norm_sq = torch.dot(g_j, g_j)
                        if g_j_norm_sq > 1e-12:
                            alpha = dot_product / g_j_norm_sq
                            pc_gradients[i] = g_i - alpha * g_j
            if self.debug and self.step_count <= 10:
                print(f'[PCGrad Debug] Step {self.step_count}: Found {conflict_count} gradient conflicts')
            projected_grad = torch.stack(pc_gradients).mean(dim=0)
        else:
            projected_grad = task_gradients[0] if task_gradients else None
        if projected_grad is None:
            if self.debug:
                print(f'[PCGrad] Step {self.step_count}: No projected gradient')
            optimizer.step()
            return {'total_loss': sum((l.item() for l in loss_list))}
        model.zero_grad()
        offset = 0
        for p in params:
            numel = p.numel()
            if offset + numel <= len(projected_grad):
                grad_segment = projected_grad[offset:offset + numel].view_as(p)
                if p.grad is None:
                    p.grad = grad_segment.to(p.device)
                else:
                    p.grad.copy_(grad_segment)
                offset += numel
        if self.use_grad_clip and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
            if self.debug and self.step_count <= 10:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'[PCGrad Debug] After clip: total_grad_norm={total_norm:.6f}')
        try:
            optimizer.step()
        except Exception as e:
            if self.debug:
                print(f'[PCGrad] Step {self.step_count}: Optimizer step failed: {e}')
            optimizer.zero_grad()
            total_loss = sum(loss_list)
            total_loss.backward()
            if self.use_grad_clip and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
            optimizer.step()
        total_loss_value = sum((l.item() for l in loss_list))
        if self.step_count % 100 == 0 and self.debug:
            print(f'[PCGrad] Step {self.step_count}: Total loss = {total_loss_value:.6f}')
        return {'total_loss': total_loss_value}

    def get_extra_params(self) -> List[nn.Parameter]:
        return []

    def state_dict(self) -> Dict:
        return {'step_count': self.step_count, 'use_grad_clip': self.use_grad_clip, 'grad_clip_norm': self.grad_clip_norm}

    def load_state_dict(self, state: Dict) -> None:
        self.step_count = state.get('step_count', 0)
        self.use_grad_clip = state.get('use_grad_clip', True)
        self.grad_clip_norm = state.get('grad_clip_norm', 5.0)
