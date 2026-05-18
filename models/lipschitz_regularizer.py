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
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LipschitzRegularizer(nn.Module):

    def __init__(self, delta: float=0.1, mu: float=0.05, eps_d: float=0.001, k_prototypes: int=5, ema_alpha: float=0.1, eps_conf: float=1e-06, enabled: bool=True, use_quantile_alignment: bool=True, use_multiple_prototypes: bool=True, **_: object):
        super().__init__()
        self.delta = float(delta)
        self.mu = float(mu)
        self.eps_d = float(eps_d)
        self.k_prototypes = int(k_prototypes)
        self.ema_alpha = float(ema_alpha)
        self.eps_conf = float(eps_conf)
        self.enabled = bool(enabled)
        self.use_quantile_alignment = bool(use_quantile_alignment)
        self.use_multiple_prototypes = bool(use_multiple_prototypes)
        self.register_buffer('tau_t', torch.tensor(1.0))
        self.register_buffer('kappa_t', torch.tensor(1.0))
        self.initialized = False

    def rho_delta(self, u: torch.Tensor) -> torch.Tensor:
        abs_u = torch.abs(u)
        return torch.where(abs_u <= self.delta, u ** 2 / (2 * self.delta), abs_u - self.delta / 2)

    def phi_mu(self, z: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(z)
        quadratic = z ** 2 / (2 * self.mu)
        linear = z - self.mu / 2
        out = torch.where(z <= 0, zero, quadratic)
        out = torch.where(z > self.mu, linear, out)
        return out

    def compute_semantic_distance(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        if emb_i.dim() == 1:
            emb_i = emb_i.unsqueeze(0)
        if emb_j.dim() == 1:
            emb_j = emb_j.unsqueeze(0)
        cos_sim = F.cosine_similarity(emb_i, emb_j, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        return (1 - cos_sim) / 2

    def build_prototypes(self, embeddings: torch.Tensor, confidences: torch.Tensor, k: Optional[int]=None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if k is None:
            k = self.k_prototypes
        if not self.use_multiple_prototypes:
            k = 1
        if embeddings.numel() == 0:
            return []
        valid_mask = torch.isfinite(confidences) & (confidences > self.eps_conf)
        n_valid = int(valid_mask.sum().item())
        if n_valid <= 0:
            return []
        k = min(int(k), n_valid, embeddings.size(0))
        if k <= 0:
            return []
        masked_conf = confidences.clone()
        masked_conf[~valid_mask] = -float('inf')
        (_, top_indices) = torch.topk(masked_conf, k=k)
        return [(embeddings[idx], confidences[idx]) for idx in top_indices]

    @torch.no_grad()
    def update_quantile_ema(self, gaps: torch.Tensor, distances: torch.Tensor) -> None:
        if gaps.numel() == 0 or distances.numel() == 0:
            return
        if not torch.isfinite(gaps).all() or not torch.isfinite(distances).all():
            return
        q95_gap = torch.quantile(torch.abs(gaps), 0.95)
        q95_dist = torch.quantile(distances, 0.95)
        if not torch.isfinite(q95_gap) or not torch.isfinite(q95_dist):
            return
        if not self.initialized or not torch.isfinite(self.tau_t) or (not torch.isfinite(self.kappa_t)):
            self.tau_t = q95_gap
            self.kappa_t = q95_dist
            self.initialized = True
        else:
            self.tau_t = self.ema_alpha * q95_gap + (1 - self.ema_alpha) * self.tau_t
            self.kappa_t = self.ema_alpha * q95_dist + (1 - self.ema_alpha) * self.kappa_t
        self.tau_t = torch.clamp(self.tau_t, min=0.0001)
        self.kappa_t = torch.clamp(self.kappa_t, min=0.0001)

    def forward(self, task_embeddings: Dict[str, torch.Tensor], task_confidences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if not self.enabled:
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': 0, 'skipped': 'disabled'})
        task_names = list(task_embeddings.keys())
        if len(task_names) < 2:
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': 0, 'skipped': 'too_few_tasks'})
        task_prototypes: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        for task in task_names:
            task_prototypes[task] = self.build_prototypes(task_embeddings[task], task_confidences[task])
        valid_tasks = [t for t in task_names if len(task_prototypes[t]) > 0]
        if len(valid_tasks) < 2:
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': 0, 'skipped': 'insufficient_prototypes', 'tau_t': float(self.tau_t.item()), 'kappa_t': float(self.kappa_t.item())})
        all_gaps = []
        all_distances = []
        total_pairs = 0
        for i in range(len(valid_tasks)):
            for j in range(i + 1, len(valid_tasks)):
                (task_i, task_j) = (valid_tasks[i], valid_tasks[j])
                for (emb_i, conf_i) in task_prototypes[task_i]:
                    for (emb_j, conf_j) in task_prototypes[task_j]:
                        delta_p = conf_i - conf_j
                        d_ij = self.compute_semantic_distance(emb_i, emb_j)
                        d_prime_ij = torch.clamp(d_ij, min=self.eps_d)
                        if not torch.isfinite(delta_p).all() or not torch.isfinite(d_prime_ij).all():
                            continue
                        all_gaps.append(delta_p)
                        all_distances.append(d_prime_ij)
                        total_pairs += 1
        if total_pairs == 0:
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': 0, 'skipped': 'no_valid_pairs', 'tau_t': float(self.tau_t.item()), 'kappa_t': float(self.kappa_t.item())})
        gaps = torch.stack(all_gaps)
        distances = torch.stack(all_distances)
        if distances.dim() > 1:
            distances = distances.squeeze()
        if distances.dim() == 0:
            distances = distances.unsqueeze(0)
        no_align = not self.use_quantile_alignment or os.environ.get('ABLATION_NO_ALIGN', '') == '1'
        if not no_align:
            self.update_quantile_ema(gaps.detach(), distances.detach())
            normalized_gaps = self.rho_delta(gaps) / self.tau_t
            normalized_dists = distances / self.kappa_t
            tau_used = float(self.tau_t.item())
            kappa_used = float(self.kappa_t.item())
        else:
            normalized_gaps = self.rho_delta(gaps)
            normalized_dists = distances
            tau_used = 1.0
            kappa_used = 1.0
        if not torch.isfinite(normalized_gaps).all() or not torch.isfinite(normalized_dists).all():
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': int(total_pairs), 'skipped': 'non_finite_after_normalization', 'tau_t': tau_used, 'kappa_t': kappa_used})
        penalties = self.phi_mu(normalized_gaps - normalized_dists)
        if not torch.isfinite(penalties).all():
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': int(total_pairs), 'skipped': 'non_finite_penalty', 'tau_t': tau_used, 'kappa_t': kappa_used})
        with torch.no_grad():
            violations = (normalized_gaps > normalized_dists).sum().item()
            violation_rate = float(violations / total_pairs)
        L_lip = penalties.mean()
        if not torch.isfinite(L_lip):
            device = next(iter(task_embeddings.values())).device
            return (torch.tensor(0.0, device=device), {'violation_rate': 0.0, 'n_pairs': int(total_pairs), 'skipped': 'non_finite_loss', 'tau_t': tau_used, 'kappa_t': kappa_used})
        stats = {'violation_rate': violation_rate, 'n_pairs': int(total_pairs), 'n_violations': int(violations), 'mean_gap': float(gaps.abs().mean().item()), 'mean_dist': float(distances.mean().item()), 'tau_t': tau_used, 'kappa_t': kappa_used, 'mean_penalty': float(L_lip.item())}
        return (L_lip, stats)

    def get_scale_stats(self) -> Dict:
        return {'tau_t': float(self.tau_t.item()), 'kappa_t': float(self.kappa_t.item()), 'initialized': self.initialized}
