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

import os
import json
import numpy as np
import torch
from typing import Dict, Any

def save_stats(stats: Dict[str, Any], path: str):
    stats_serializable = {}
    for (key, value) in stats.items():
        if isinstance(value, np.ndarray):
            stats_serializable[key] = value.tolist()
        elif torch.is_tensor(value):
            stats_serializable[key] = value.cpu().numpy().tolist()
        else:
            stats_serializable[key] = value
    with open(path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

def load_stats(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Stats file not found: {path}')
    with open(path, 'r') as f:
        stats = json.load(f)
    return stats

def get_stats(dataset_name: str, root: str, split: str='train') -> Dict[str, Any]:
    stats_file = os.path.join(root, 'stats', f'{dataset_name}_{split}_stats.json')
    if os.path.exists(stats_file):
        return load_stats(stats_file)
    if dataset_name == 'nyuv2':
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    else:
        return {'mean': 0.0, 'std': 1.0}
