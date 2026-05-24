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
import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List
import torch
from tqdm import tqdm
from config import ExperimentConfig
from data_loader.build import build_dataloader
from models.mtl_model import DynamicLipschitzMTL

def _apply_dataset_defaults(config: ExperimentConfig, *, tasks_overridden: bool, data_root_overridden: bool) -> None:
    if not tasks_overridden:
        config.tasks_enabled = ['seg', 'depth', 'normal']
    if not data_root_overridden:
        config.data_root = './data/nyuv2'
    config.dataset_name = 'nyuv2'
    config.training.force_backbone_fp32 = False
    config.backbone.pretrained = True
    if config.training.batch_size == 64:
        config.training.batch_size = 8

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute NYUv2 proxy calibration stats from an ERM checkpoint')
    p.add_argument('--ckpt', type=str, required=True, help='ERM checkpoint used as calibration reference')
    p.add_argument('--split', type=str, default='val', choices=['val'], help='NYUv2 split used for calibration; this public release uses val')
    p.add_argument('--data_root', type=str, default=None, help='Override NYUv2 root')
    p.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks override (default: seg,depth,normal)')
    p.add_argument('--device', type=str, default='cuda', help='Device, e.g. cuda, cuda:0, cpu; falls back to cpu if CUDA is unavailable')
    p.add_argument('--batch_size', type=int, default=None, help='Batch size')
    p.add_argument('--num_workers', type=int, default=None, help='Num workers')
    p.add_argument('--seed', type=int, default=None, help='Loader seed')
    p.add_argument('--data_seed', type=int, default=None, help='Dataset-side seed')
    p.add_argument('--output', type=str, default=None, help='Output JSON path')
    return p.parse_args()

def _safe_path_label(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    return f'{parent}/{name}' if parent else name

def _safe_saved_label(path: str) -> str:
    return os.path.basename(path)

def _resolve_device(device_name: str) -> torch.device:
    if str(device_name).startswith('cuda') and (not torch.cuda.is_available()):
        return torch.device('cpu')
    return torch.device(device_name)

def _sample_valid_mask(mask: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if mask.dim() >= 3:
        return mask.reshape(mask.size(0), -1).any(dim=1)
    if mask.dim() > 1 and mask.size(-1) == 1:
        return mask.squeeze(-1).bool()
    return mask.bool()

def _tensor_summary(values: torch.Tensor) -> Dict[str, float]:
    (q10, q50, q90) = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype))
    return {'n_valid_samples': int(values.numel()), 'min': float(values.min().item()), 'mean': float(values.mean().item()), 'q10': float(q10.item()), 'q50': float(q50.item()), 'q90': float(q90.item()), 'max': float(values.max().item())}

@torch.no_grad()
def main() -> None:
    args = _parse_args()
    config = ExperimentConfig()
    _apply_dataset_defaults(config, tasks_overridden=args.tasks is not None, data_root_overridden=args.data_root is not None)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.training.num_workers = int(args.num_workers)
    if args.seed is not None:
        config.training.seed = int(args.seed)
    if args.data_seed is not None:
        config.training.data_seed = int(args.data_seed)
    tasks_enabled: List[str] = list(config.tasks_enabled)
    if args.tasks is not None:
        tasks_enabled = [t.strip() for t in str(args.tasks).split(',') if t.strip()]
        if not tasks_enabled:
            raise ValueError('--tasks parsed to an empty list. Provide comma-separated task names.')
        config.tasks_enabled = list(tasks_enabled)
    ckpt_path = os.path.abspath(str(args.ckpt))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    output_path = os.path.abspath(str(args.output)) if args.output is not None else os.path.join(os.path.dirname(ckpt_path), 'nyuv2_proxy_stats.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = _resolve_device(str(args.device))
    data_seed = config.training.seed if config.training.data_seed is None else int(config.training.data_seed)
    loader = build_dataloader(dataset_name='nyuv2', root=config.data_root, split=str(args.split), batch_size=config.training.batch_size, tasks_enabled=tasks_enabled, shuffle=False, seed=int(config.training.seed), dataset_seed=int(data_seed), num_workers=int(config.training.num_workers))
    model = DynamicLipschitzMTL(dataset_name='nyuv2', tasks_enabled=tasks_enabled, force_backbone_fp32=config.training.force_backbone_fp32, backbone_config=asdict(config.backbone), head_config={}, lip_config=asdict(config.lipschitz)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    pools: Dict[str, List[torch.Tensor]] = {task: [] for task in tasks_enabled}
    for batch in tqdm(loader, desc=f'ProxyStats({args.split})'):
        x = batch['x'].to(device)
        y_mask = {k: v.to(device) if v is not None else None for (k, v) in batch['y_mask'].items()}
        outputs = model(x, y_mask=y_mask)
        raw_conf = outputs.get('raw_confidences')
        if raw_conf is None:
            raise RuntimeError('Model output missing raw_confidences; cannot compute NYUv2 proxy stats.')
        for task in tasks_enabled:
            if task not in raw_conf:
                raise KeyError(f"raw_confidences missing task '{task}'")
            sample_valid = _sample_valid_mask(y_mask.get(task), batch_size=x.size(0), device=device)
            if int(sample_valid.sum().item()) == 0:
                continue
            pools[task].append(raw_conf[task].detach().float()[sample_valid].cpu())
    task_stats: Dict[str, Dict[str, float | str | bool]] = {}
    for task in tasks_enabled:
        if not pools[task]:
            raise RuntimeError(f"Task '{task}' has no valid proxy samples on split={args.split}")
        values = torch.cat(pools[task], dim=0)
        stats = _tensor_summary(values)
        if task == 'seg':
            stats['proxy_definition'] = 'valid-region mean of max-softmax'
            stats['squash_rule'] = 'identity'
            stats['already_bounded'] = True
        else:
            stats['proxy_definition'] = 'valid-region mean of channel-wise L2 norm on decoder penultimate feature map'
            stats['squash_rule'] = 'clip((s-q10)/(q90-q10+1e-6), 0, 1)'
            stats['already_bounded'] = False
        task_stats[task] = stats
    summary = {'dataset_name': 'nyuv2', 'split': str(args.split), 'source_ckpt': _safe_path_label(ckpt_path), 'data_root': '<redacted>', 'seed': int(config.training.seed), 'data_seed': int(data_seed), 'backbone': {'model_name': str(config.backbone.model_name), 'pretrained': bool(config.backbone.pretrained)}, 'protocol': {'seg': 'valid-region mean of max-softmax', 'depth': 'valid-region mean of channel-wise L2 norm on decoder penultimate feature map', 'normal': 'valid-region mean of channel-wise L2 norm on decoder penultimate feature map', 'depth_normal_squash': 'clip((s-q10)/(q90-q10+1e-6), 0, 1)'}, 'tasks': task_stats}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved: {_safe_saved_label(output_path)}')
if __name__ == '__main__':
    main()
