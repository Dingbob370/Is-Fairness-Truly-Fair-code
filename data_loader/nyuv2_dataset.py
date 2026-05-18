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
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any

class NYUv2MTLDataset(Dataset):

    def __init__(self, root: str, split: str='train', tasks_enabled: Optional[List[str]]=None, transforms: Optional[Any]=None):
        self.root = root
        self.split = split
        self.tasks_enabled = tasks_enabled or ['seg', 'depth', 'normal']
        self.transforms = transforms
        assert split in ['train', 'val'], f'NYUv2 release supports train/val only; got split={split}. Use val for evaluation.'
        self.data_path = os.path.join(root, split)
        self.image_dir = os.path.join(self.data_path, 'image')
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'Missing required file or directory: {self.image_dir}')
        self.sample_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')], key=lambda x: int(x.split('.')[0]))
        self._validate_data_files()
        print(f'NYUv2MTLDataset(split={split}): n={len(self.sample_files)}')
        print(f'Tasks: {self.tasks_enabled}')

    def _validate_data_files(self):
        for i in range(len(self.sample_files)):
            sample_id = self.sample_files[i].replace('.npy', '')
            img_path = os.path.join(self.data_path, 'image', f'{sample_id}.npy')
            if not os.path.exists(img_path):
                raise FileNotFoundError(f'Missing required file or directory: {img_path}')
            for task in self.tasks_enabled:
                if task == 'seg':
                    label_path = os.path.join(self.data_path, 'label', f'{sample_id}.npy')
                elif task == 'depth':
                    label_path = os.path.join(self.data_path, 'depth', f'{sample_id}.npy')
                elif task == 'normal':
                    label_path = os.path.join(self.data_path, 'normal', f'{sample_id}.npy')
                else:
                    raise ValueError(f'Unknown or unsupported task: {task}')
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f'Missing required file or directory: {label_path}')

    def __len__(self):
        return len(self.sample_files)

    def _load_npy_mmap(self, path: str):
        return np.load(path, mmap_mode='r')

    def __getitem__(self, idx):
        sample_id = self.sample_files[idx].replace('.npy', '')
        img_path = os.path.join(self.data_path, 'image', f'{sample_id}.npy')
        image = self._load_npy_mmap(img_path)
        if image.ndim == 3 and image.shape[2] == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        elif image.ndim == 3 and image.shape[0] == 3:
            image = torch.from_numpy(image.copy()).float()
        else:
            raise ValueError(f'Unsupported tensor shape: {image.shape}')
        if image.max() > 1.0:
            image = image / 255.0
        y_dict = {}
        y_mask_dict = {}
        if 'seg' in self.tasks_enabled:
            seg_path = os.path.join(self.data_path, 'label', f'{sample_id}.npy')
            seg = self._load_npy_mmap(seg_path)
            seg_tensor = torch.from_numpy(seg.copy()).long()
            y_dict['seg'] = seg_tensor
            y_mask_dict['seg'] = seg_tensor != -1
        if 'depth' in self.tasks_enabled:
            depth_path = os.path.join(self.data_path, 'depth', f'{sample_id}.npy')
            depth = self._load_npy_mmap(depth_path)
            if depth.ndim == 2:
                depth = depth[np.newaxis, :, :]
            elif depth.ndim == 3 and depth.shape[0] == 1:
                pass
            elif depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth.transpose(2, 0, 1)
            else:
                raise ValueError(f'Unsupported tensor shape: {depth.shape}')
            depth_tensor = torch.from_numpy(depth.copy()).float()
            y_dict['depth'] = depth_tensor
            depth_mask = (depth_tensor > 0).bool()
            y_mask_dict['depth'] = depth_mask
        if 'normal' in self.tasks_enabled:
            normal_path = os.path.join(self.data_path, 'normal', f'{sample_id}.npy')
            normal = self._load_npy_mmap(normal_path)
            if normal.ndim == 3 and normal.shape[2] == 3:
                normal = normal.transpose(2, 0, 1)
            elif normal.ndim == 3 and normal.shape[0] == 3:
                pass
            else:
                raise ValueError(f'Unsupported tensor shape: {normal.shape}')
            normal_tensor = torch.from_numpy(normal.copy()).float()
            y_dict['normal'] = normal_tensor
            normal_norm = torch.sqrt(torch.sum(normal_tensor ** 2, dim=0, keepdim=True))
            normal_mask = (normal_norm > 0).bool()
            y_mask_dict['normal'] = normal_mask
        if self.transforms is not None:
            (image, y_dict, y_mask_dict) = self.transforms(image, y_dict, y_mask_dict)
        return {'x': image, 'y': y_dict, 'y_mask': y_mask_dict, 'meta': {'idx': idx, 'sample_id': sample_id, 'split': self.split}}
