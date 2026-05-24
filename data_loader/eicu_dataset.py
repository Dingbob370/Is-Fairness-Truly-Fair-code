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

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
EICU_FEATURE_NAMES_17: List[str] = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

def _ffill_zero(features: np.ndarray) -> np.ndarray:
    """Forward-fill along time dimension, then fill remaining NaN with 0."""
    x = features
    if x.ndim != 2:
        raise ValueError(f'_ffill_zero expects (T,F), got {x.shape}')
    (T, F) = x.shape
    for f in range(F):
        col = x[:, f]
        for t in range(1, T):
            if not np.isfinite(col[t]):
                col[t] = col[t - 1]
        col[~np.isfinite(col)] = 0.0
        x[:, f] = col
    return x

class EICUMTLDataset(Dataset):

    def __init__(self, root: str, split: str, tasks_enabled: List[str], feature_dim: int=17, normalization: str='train_stats', stats_dir: Optional[str]=None, fillna_method: str='ffill_zero', verbose: bool=False):
        self.root = os.path.expanduser(str(root))
        self.split = str(split)
        self.tasks_enabled = list(tasks_enabled)
        self.feature_dim = int(feature_dim)
        self.normalization = str(normalization)
        self.fillna_method = str(fillna_method)
        self.verbose = bool(verbose)
        if self.split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown split={self.split}, expected train/val/test')
        if self.feature_dim != 17:
            raise ValueError(f'Unsupported eICU feature_dim={self.feature_dim}. Expected 17.')
        for task in self.tasks_enabled:
            if task not in {'mortality', 'los'}:
                raise ValueError(f"Unsupported eICU task '{task}'. Expected one of ['mortality', 'los'].")
        if stats_dir is None:
            stats_dir = os.path.join(self.root, 'stats')
        stats_path = os.path.join(stats_dir, 'stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f'eICU stats.json not found: {stats_path}. Provide a compact benchmark directory with stats/stats.json.')
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        feature_names = list(stats.get('feature_names') or [])
        if feature_names and feature_names != EICU_FEATURE_NAMES_17:
            raise ValueError(f'eICU feature schema mismatch. Got {feature_names}; expected {EICU_FEATURE_NAMES_17}.')
        self.feature_names = EICU_FEATURE_NAMES_17
        self.stats = stats
        self._mean = np.asarray(self.stats.get('mean', [0.0] * self.feature_dim), dtype=np.float32)
        self._std = np.asarray(self.stats.get('std', [1.0] * self.feature_dim), dtype=np.float32)
        if self._mean.shape != (self.feature_dim,) or self._std.shape != (self.feature_dim,):
            raise ValueError(f'stats mean/std shape invalid: mean={self._mean.shape} std={self._std.shape}')
        self._x_raw = np.load(os.path.join(self.root, 'x_raw.npy'), mmap_mode='r')
        self._x_mask = np.load(os.path.join(self.root, 'x_mask.npy'), mmap_mode='r')
        self._seq_len = np.load(os.path.join(self.root, 'seq_len.npy'), mmap_mode='r')
        self._stay_id = np.load(os.path.join(self.root, 'stay_id.npy'), mmap_mode='r')
        self._y_mortality = np.load(os.path.join(self.root, 'y_mortality.npy'), mmap_mode='r')
        self._m_mortality = np.load(os.path.join(self.root, 'y_mask_mortality.npy'), mmap_mode='r')
        self._y_los = np.load(os.path.join(self.root, 'y_los.npy'), mmap_mode='r')
        self._m_los = np.load(os.path.join(self.root, 'y_mask_los.npy'), mmap_mode='r')
        split_idx = np.load(os.path.join(self.root, 'split_idx.npz'))
        key = f'{self.split}_idx'
        if key not in split_idx:
            raise KeyError(f'split_idx.npz missing key={key}. Available: {list(split_idx.keys())}')
        self._indices = split_idx[key].astype(np.int64, copy=False)
        N = int(self._x_raw.shape[0])
        if self._indices.size == 0:
            raise RuntimeError(f"split '{self.split}' has 0 samples. Check your preprocessing split ratios.")
        for (arr_name, arr) in [('x_raw', self._x_raw), ('x_mask', self._x_mask), ('seq_len', self._seq_len), ('stay_id', self._stay_id), ('y_mortality', self._y_mortality), ('y_mask_mortality', self._m_mortality), ('y_los', self._y_los), ('y_mask_los', self._m_los)]:
            if int(arr.shape[0]) != N:
                raise ValueError(f'{arr_name} first dim mismatch: {arr.shape[0]} != {N}')
        if self.verbose:
            print(f'EICUMTLDataset(split={self.split}): n={len(self._indices)} | feature_dim={self.feature_dim} | tasks={self.tasks_enabled}')

    def __len__(self) -> int:
        return int(self._indices.size)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.normalization != 'train_stats':
            return features
        std = np.where(self._std == 0.0, 1.0, self._std)
        return (features - self._mean) / (std + 1e-08)

    def _handle_nan(self, features: np.ndarray) -> np.ndarray:
        if self.fillna_method == 'ffill_zero':
            return _ffill_zero(features)
        if self.fillna_method == 'zero':
            x = features
            x[~np.isfinite(x)] = 0.0
            return x
        raise ValueError(f'Unknown fillna_method={self.fillna_method}')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gidx = int(self._indices[int(idx)])
        seq_len = int(self._seq_len[gidx])
        if seq_len <= 0:
            seq_len = 1
        x_raw = np.array(self._x_raw[gidx, :seq_len, :], dtype=np.float32, copy=True)
        x_mask = np.array(self._x_mask[gidx, :seq_len, :], copy=False).astype(bool, copy=False)
        x_filled = self._handle_nan(x_raw)
        x_norm = self._normalize(x_filled)
        y: Dict[str, torch.Tensor] = {}
        y_mask: Dict[str, torch.Tensor] = {}
        if 'mortality' in self.tasks_enabled:
            yv = float(self._y_mortality[gidx, 0])
            mv = bool(self._m_mortality[gidx, 0])
            y['mortality'] = torch.tensor([yv], dtype=torch.float32)
            y_mask['mortality'] = torch.tensor([mv], dtype=torch.bool)
        if 'los' in self.tasks_enabled:
            yv = float(self._y_los[gidx, 0])
            mv = bool(self._m_los[gidx, 0])
            y['los'] = torch.tensor([yv], dtype=torch.float32)
            y_mask['los'] = torch.tensor([mv], dtype=torch.bool)
        meta = {'idx': int(gidx), 'seq_len': int(seq_len), 'split': self.split}
        return {'x': torch.from_numpy(x_norm).to(dtype=torch.float32), 'x_mask': torch.from_numpy(x_mask), 'y': y, 'y_mask': y_mask, 'meta': meta}
