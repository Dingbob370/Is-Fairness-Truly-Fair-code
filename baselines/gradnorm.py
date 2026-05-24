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
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from .base import BaseProcessor

class GradNormProcessor(BaseProcessor):

    def __init__(self, task_names: List[str], device: str='cuda', alpha: float=1.5, grad_clip_norm: float=10.0, eps: float=1e-08):
        super().__init__(task_names, device)
        self.alpha = alpha
        self.grad_clip_norm = grad_clip_norm
        self.eps = eps
        self.log_task_weights = nn.Parameter(torch.zeros(self.num_tasks, device=device))
        self.initial_losses: Optional[Dict[str, float]] = None
        self.step_count = 0

    @property
    def task_weights(self):
        return torch.exp(self.log_task_weights)

    def process(self, task_losses: Dict[str, torch.Tensor], model: Optional[nn.Module]=None, **kwargs) -> torch.Tensor:
        self.step_count += 1
        valid_task_losses = {}
        for (task, loss) in task_losses.items():
            if loss is None:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            valid_task_losses[task] = loss
        if not valid_task_losses:
            return torch.tensor(0.0, device=self.device)
        if self.initial_losses is None:
            self.initial_losses = {}
            for (task, loss) in valid_task_losses.items():
                loss_value = loss.detach().item()
                if loss_value < self.eps:
                    loss_value = 1.0
                self.initial_losses[task] = loss_value
        weighted_losses = {}
        weights = self.task_weights
        for (i, task) in enumerate(self.task_names):
            if task in valid_task_losses:
                weighted_losses[task] = weights[i] * valid_task_losses[task]
        shared_params = self._get_shared_params(model)
        if shared_params is None:
            total_weighted_loss = sum(weighted_losses.values())
            return total_weighted_loss
        (grad_norms, current_losses) = self._compute_gradient_norms(valid_task_losses, shared_params)
        if not grad_norms:
            total_weighted_loss = sum(weighted_losses.values())
            return total_weighted_loss
        r_i = []
        for (idx, task) in enumerate(valid_task_losses.keys()):
            current = current_losses[idx]
            initial = self.initial_losses[task]
            current = max(current, self.eps)
            initial = max(initial, self.eps)
            r = current / initial
            r = min(max(r, 0.01), 100.0)
            r_i.append(r)
        r_avg = sum(r_i) / len(r_i)
        r_avg = max(r_avg, self.eps)
        grad_norms_tensor = torch.stack(grad_norms)
        G_avg = grad_norms_tensor.mean()
        if G_avg.item() < self.eps:
            G_avg = torch.tensor(1.0, device=self.device)
        target_grad_norms = []
        for r in r_i:
            ratio = r / r_avg
            ratio = min(max(ratio, 0.01), 100.0)
            target = G_avg * ratio ** self.alpha
            target = min(max(target, G_avg.item() * 0.01), G_avg.item() * 100.0)
            target_grad_norms.append(target)
        target_grad_norms_tensor = torch.tensor(target_grad_norms, device=self.device)
        gradnorm_loss = torch.mean(torch.abs(grad_norms_tensor - target_grad_norms_tensor))
        if gradnorm_loss.item() > 100.0:
            gradnorm_loss = gradnorm_loss * 0.1
        total_weighted_loss = sum(weighted_losses.values())
        total_loss = total_weighted_loss + gradnorm_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return total_weighted_loss
        if self.step_count % 100 == 0:
            self._renormalize_weights()
        if self.step_count % 200 == 0:
            print(f'[GradNorm] Step {self.step_count}: weights={[w.item() for w in weights]}')
        return total_loss

    def _get_shared_params(self, model: nn.Module) -> Optional[List[nn.Parameter]]:
        if model is None:
            return None
        if hasattr(model, 'backbone'):
            return list(model.backbone.parameters())
        return None

    def _compute_gradient_norms(self, valid_task_losses: Dict[str, torch.Tensor], shared_params: List[nn.Parameter]) -> Tuple[List[torch.Tensor], List[float]]:
        grad_norms = []
        current_losses = []
        original_grads = {}
        for param in shared_params:
            if param.grad is not None:
                original_grads[param] = param.grad.clone()
        try:
            for (i, task) in enumerate(self.task_names):
                if task not in valid_task_losses:
                    continue
                for param in shared_params:
                    param.grad = None
                weight = self.task_weights[i]
                task_loss = valid_task_losses[task]
                weighted_loss = weight * task_loss
                grads = torch.autograd.grad(weighted_loss, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
                grad_norm_sq = 0.0
                has_grad = False
                for grad in grads:
                    if grad is None:
                        continue
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        continue
                    grad_norm = torch.norm(grad)
                    if grad_norm > self.grad_clip_norm:
                        grad = grad * self.grad_clip_norm / grad_norm
                    grad_norm_sq += torch.sum(grad * grad).item()
                    has_grad = True
                if has_grad:
                    if grad_norm_sq < self.eps:
                        grad_norm_sq = self.eps
                    grad_norm = torch.sqrt(torch.tensor(grad_norm_sq, device=self.device))
                    if grad_norm.item() > 100.0:
                        grad_norm = torch.tensor(100.0, device=self.device)
                    elif grad_norm.item() < self.eps:
                        grad_norm = torch.tensor(self.eps, device=self.device)
                    grad_norms.append(grad_norm)
                    current_losses.append(task_loss.detach().item())
                for param in shared_params:
                    param.grad = None
        except Exception as e:
            pass
        for param in shared_params:
            if param in original_grads:
                param.grad = original_grads[param]
        return (grad_norms, current_losses)

    def _renormalize_weights(self):
        with torch.no_grad():
            weights = self.task_weights
            weight_sum = weights.sum().item()
            if abs(weight_sum) < self.eps:
                return
            scaling_factor = self.num_tasks / weight_sum
            new_weights = weights * scaling_factor
            new_log_weights = torch.log(new_weights)
            new_log_weights = torch.clamp(new_log_weights, -5.0, 5.0)
            self.log_task_weights.data.copy_(new_log_weights)

    def get_extra_params(self) -> List[nn.Parameter]:
        return [self.log_task_weights]

    def state_dict(self) -> Dict:
        state = {'log_task_weights': self.log_task_weights.data.clone(), 'alpha': self.alpha, 'step_count': self.step_count, 'initial_losses': self.initial_losses.copy() if self.initial_losses else None}
        return state

    def load_state_dict(self, state: Dict) -> None:
        if 'log_task_weights' in state:
            self.log_task_weights.data = state['log_task_weights'].to(self.device)
        if 'initial_losses' in state:
            self.initial_losses = state['initial_losses']
        if 'alpha' in state:
            self.alpha = state.get('alpha', self.alpha)
        if 'step_count' in state:
            self.step_count = state.get('step_count', self.step_count)
