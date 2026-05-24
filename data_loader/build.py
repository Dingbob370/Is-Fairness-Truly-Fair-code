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

import torch
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from typing import Optional

def build_dataloader(dataset_name: str, root: str, split: str, batch_size: int, tasks_enabled: list, shuffle: bool, seed: int, dataset_seed: Optional[int]=None, loader_seed: Optional[int]=None, num_workers: int=0, **kwargs):
    dataset_seed = seed if dataset_seed is None else int(dataset_seed)
    loader_seed = seed if loader_seed is None else int(loader_seed)
    random.seed(loader_seed)
    np.random.seed(loader_seed)
    torch.manual_seed(loader_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(loader_seed)
        torch.cuda.manual_seed_all(loader_seed)
    generator = torch.Generator()
    generator.manual_seed(loader_seed)

    def worker_init_fn(worker_id):
        worker_seed = loader_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    if dataset_name == 'nyuv2':
        from .nyuv2_dataset import NYUv2MTLDataset
        from .collate_fn import collate_nyuv2
        dataset = NYUv2MTLDataset(root=root, split=split, tasks_enabled=tasks_enabled, transforms=None, **kwargs)
        collate_fn = collate_nyuv2
    elif dataset_name == 'mimic3':
        from .mimic3_dataset import MIMIC3MTLDataset
        exclude_val_from_train = kwargs.get('exclude_val_from_train', True)
        mimic_kwargs = {'feature_dim': kwargs.get('feature_dim', 17), 'normalization': kwargs.get('normalization', 'train_stats'), 'stats_dir': kwargs.get('stats_dir'), 'fillna_method': kwargs.get('fillna_method', 'ffill_zero'), 'val_ratio': kwargs.get('val_ratio', 0.15), 'seed': dataset_seed, 'verbose': kwargs.get('verbose', False), 'skip_stats': kwargs.get('skip_stats', False)}
        if 'time_row_policy' in kwargs:
            mimic_kwargs['time_row_policy'] = kwargs.get('time_row_policy')
        if 'time_row_policy_test' in kwargs:
            mimic_kwargs['time_row_policy_test'] = kwargs.get('time_row_policy_test')
        dataset = MIMIC3MTLDataset(root=root, split=split, tasks_enabled=tasks_enabled, **mimic_kwargs)
        if split == 'train' and exclude_val_from_train and getattr(dataset, '_derive_val_from_train', False):
            val_stays = getattr(dataset, '_val_stay_ids', None)
            if val_stays:
                keep_indices = [i for (i, ep) in enumerate(getattr(dataset, 'episodes', [])) if ep.get('stay_id') not in val_stays]
                dataset = Subset(dataset, keep_indices)
                if mimic_kwargs.get('verbose', False):
                    print(f'[build_dataloader] Exclude derived val stays from train: keep={len(keep_indices)}, drop={len(val_stays)}')
        from .collate_fn import collate_mimic
        collate_fn = collate_mimic
    elif dataset_name == 'eicu':
        from .eicu_dataset import EICUMTLDataset
        eicu_kwargs = {'feature_dim': kwargs.get('feature_dim', 17), 'normalization': kwargs.get('normalization', 'train_stats'), 'stats_dir': kwargs.get('stats_dir'), 'fillna_method': kwargs.get('fillna_method', 'ffill_zero'), 'verbose': kwargs.get('verbose', False)}
        dataset = EICUMTLDataset(root=root, split=split, tasks_enabled=tasks_enabled, **eicu_kwargs)
        from .collate_fn import collate_mimic
        collate_fn = collate_mimic
    else:
        raise ValueError(f'Unknown or unsupported dataset: {dataset_name}')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn if num_workers > 0 else None, collate_fn=collate_fn, pin_memory=True, drop_last=False, generator=generator)
    return dataloader
