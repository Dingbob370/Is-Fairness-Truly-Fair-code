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
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class BandControllerDiagnostics:
    lambda_t: float
    r_t_ema: Optional[float]
    step_count: int
    recent_r_mean: float
    recent_r_std: float
    recent_lambda_mean: float
    is_r_in_band: bool
    is_lambda_saturated: bool
    is_warmup: bool
    is_healthy: bool

    def as_dict(self) -> Dict:
        return {'lambda_t': self.lambda_t, 'r_t_ema': self.r_t_ema, 'step_count': self.step_count, 'recent_r_mean': self.recent_r_mean, 'recent_r_std': self.recent_r_std, 'recent_lambda_mean': self.recent_lambda_mean, 'is_r_in_band': self.is_r_in_band, 'is_lambda_saturated': self.is_lambda_saturated, 'is_warmup': self.is_warmup, 'is_healthy': self.is_healthy}

class BandController:

    def __init__(self, r_star: float=0.2, r_min: float=0.15, r_max: float=0.35, eta: float=0.1, lambda_init: float=0.1, lambda_min: float=0.01, lambda_max: float=1.0, warmup_steps: int=100, ema_alpha: float=0.1, enabled: bool=True, **_: object):
        self.r_star = float(r_star)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.eta = float(eta)
        self.lambda_init = float(lambda_init)
        self.lambda_t = float(lambda_init)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.warmup_steps = int(warmup_steps)
        self.ema_alpha = float(ema_alpha)
        self.enabled = bool(enabled)
        self.step_count = 0
        self.r_t_ema: Optional[float] = None
        self.lambda_history: List[float] = []
        self.violation_history: List[float] = []
        self.r_ema_history: List[float] = []

    def update(self, r_t: float) -> float:
        self.step_count += 1
        r_t = float(r_t)
        if self.r_t_ema is None:
            self.r_t_ema = r_t
        else:
            self.r_t_ema = self.ema_alpha * r_t + (1 - self.ema_alpha) * self.r_t_ema
        if not self.enabled:
            self._record_history(r_t)
            return self.lambda_t
        if self.step_count <= self.warmup_steps:
            self._record_history(r_t)
            return self.lambda_t
        error = self.r_t_ema - self.r_star if self.r_t_ema is not None else 0.0
        lambda_new = self.lambda_t * (1 + self.eta * error)
        self.lambda_t = float(np.clip(lambda_new, self.lambda_min, self.lambda_max))
        self._record_history(r_t)
        return self.lambda_t

    def _record_history(self, r_t: float) -> None:
        self.lambda_history.append(self.lambda_t)
        self.violation_history.append(float(r_t))
        self.r_ema_history.append(float(self.r_t_ema) if self.r_t_ema is not None else float(r_t))

    def get_lambda(self) -> float:
        return self.lambda_t

    def get_diagnostics(self) -> BandControllerDiagnostics:
        recent_n = min(50, len(self.violation_history))
        recent_r_mean = float(np.mean(self.violation_history[-recent_n:])) if self.violation_history else 0.0
        recent_r_std = float(np.std(self.violation_history[-recent_n:])) if self.violation_history else 0.0
        recent_lambda_mean = float(np.mean(self.lambda_history[-recent_n:])) if self.lambda_history else float(self.lambda_t)
        r_ema = float(self.r_t_ema) if self.r_t_ema is not None else None
        is_r_in_band = self.r_min <= (r_ema if r_ema is not None else 0.0) <= self.r_max
        is_lambda_saturated = self.lambda_t in (self.lambda_min, self.lambda_max)
        is_warmup = self.step_count <= self.warmup_steps
        is_healthy = self.enabled and is_r_in_band and (not is_lambda_saturated) and (not is_warmup)
        return BandControllerDiagnostics(lambda_t=float(self.lambda_t), r_t_ema=r_ema, step_count=int(self.step_count), recent_r_mean=recent_r_mean, recent_r_std=recent_r_std, recent_lambda_mean=recent_lambda_mean, is_r_in_band=is_r_in_band, is_lambda_saturated=is_lambda_saturated, is_warmup=is_warmup, is_healthy=is_healthy)

    def reset(self) -> None:
        self.lambda_t = float(self.lambda_init)
        self.r_t_ema = None
        self.step_count = 0
        self.lambda_history.clear()
        self.violation_history.clear()
        self.r_ema_history.clear()
