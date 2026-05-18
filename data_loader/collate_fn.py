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
import numpy as np
from typing import Dict, List, Any

def collate_nyuv2(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated = {}
    collated['x'] = torch.stack([item['x'] for item in batch], dim=0)
    collated['y'] = {}
    for task in batch[0]['y'].keys():
        collated['y'][task] = torch.stack([item['y'][task] for item in batch], dim=0)
    collated['y_mask'] = {}
    for task in batch[0]['y_mask'].keys():
        masks = [item['y_mask'][task] for item in batch]
        if masks[0] is not None:
            collated['y_mask'][task] = torch.stack(masks, dim=0)
        else:
            collated['y_mask'][task] = None
    collated['meta'] = {}
    for key in batch[0]['meta'].keys():
        if key == 'idx':
            collated['meta']['idx'] = [item['meta']['idx'] for item in batch]
        else:
            values = [item['meta'][key] for item in batch]
            if all((isinstance(v, (int, str)) for v in values)):
                collated['meta'][key] = values
            else:
                try:
                    collated['meta'][key] = torch.stack(values, dim=0)
                except:
                    collated['meta'][key] = values
    return collated

def collate_mimic(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        return {}
    collated = {}
    max_len = max((item['x'].shape[0] for item in batch))
    batch_size = len(batch)
    feat_dims = {int(item['x'].shape[1]) for item in batch}
    if len(feat_dims) != 1:
        raise RuntimeError(f'collate_mimic: feature_dim mismatch in batch: {sorted(feat_dims)}')
    feat_dim = next(iter(feat_dims))
    x_padded = torch.zeros(batch_size, max_len, feat_dim)
    seq_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for (i, item) in enumerate(batch):
        seq_len = item['x'].shape[0]
        x_padded[i, :seq_len] = item['x']
        seq_mask[i, :seq_len] = True
    collated['x'] = x_padded
    collated['seq_mask'] = seq_mask
    if batch[0]['x_mask'] is not None:
        x_mask_padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.bool)
        for (i, item) in enumerate(batch):
            seq_len = item['x_mask'].shape[0]
            x_mask_padded[i, :seq_len] = item['x_mask']
        collated['x_mask'] = x_mask_padded
    else:
        collated['x_mask'] = None
    collated['y'] = {}
    collated['y_mask'] = {}
    all_tasks = set()
    for item in batch:
        all_tasks.update(item['y'].keys())
    for task in all_tasks:
        first_item = next((item for item in batch if task in item['y']), None)
        if first_item is None:
            continue
        y_shape = first_item['y'][task].shape
        y_dtype = first_item['y'][task].dtype
        if task == 'decomp':
            y_padded = torch.zeros(batch_size, max_len, dtype=y_dtype)
            y_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool)
            for (i, item) in enumerate(batch):
                if task in item['y']:
                    seq_len = item['y'][task].shape[0]
                    y_padded[i, :seq_len] = item['y'][task]
                    if task in item['y_mask'] and item['y_mask'][task] is not None:
                        y_mask_padded[i, :seq_len] = item['y_mask'][task]
        elif len(y_shape) == 0:
            y_padded = torch.zeros(batch_size, dtype=y_dtype)
            y_mask_padded = torch.zeros(batch_size, dtype=torch.bool)
            for (i, item) in enumerate(batch):
                if task in item['y']:
                    y_padded[i] = item['y'][task]
                    if task in item['y_mask'] and item['y_mask'][task] is not None:
                        y_mask_padded[i] = item['y_mask'][task]
        else:
            y_padded = torch.zeros(batch_size, *y_shape, dtype=y_dtype)
            y_mask_padded = torch.zeros(batch_size, *y_shape, dtype=torch.bool)
            for (i, item) in enumerate(batch):
                if task in item['y']:
                    y_padded[i] = item['y'][task]
                    if task in item['y_mask'] and item['y_mask'][task] is not None:
                        y_mask_padded[i] = item['y_mask'][task]
        collated['y'][task] = y_padded
        collated['y_mask'][task] = y_mask_padded
    collated['meta'] = {}
    meta_keys = batch[0]['meta'].keys()
    for key in meta_keys:
        if key == 'idx':
            collated['meta']['idx'] = [item['meta']['idx'] for item in batch]
        elif key == 'seq_len':
            collated['meta']['seq_len'] = torch.tensor([item['meta']['seq_len'] for item in batch])
        elif key == 'split':
            collated['meta']['split'] = batch[0]['meta']['split']
        else:
            values = [item['meta'][key] for item in batch]
            if all((isinstance(v, (int, float)) for v in values)):
                collated['meta'][key] = torch.tensor(values)
            else:
                collated['meta'][key] = values
    return collated
