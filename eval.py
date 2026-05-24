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
from typing import Dict
import torch
from tqdm import tqdm
from config import ExperimentConfig
from data_loader.build import build_dataloader
from metrics import NYUv2MetricAccumulator, compute_unified_score, evaluate_all_tasks, metric_keys_for_model
from models.mtl_model import DynamicLipschitzMTL

def _apply_dataset_defaults(config: ExperimentConfig, *, tasks_overridden: bool, data_root_overridden: bool) -> None:
    if str(config.dataset_name).lower() != 'nyuv2':
        return
    if not tasks_overridden:
        config.tasks_enabled = ['seg', 'depth', 'normal']
    if not data_root_overridden:
        config.data_root = './data/nyuv2'
    config.training.force_backbone_fp32 = False
    config.backbone.pretrained = True
    if config.training.batch_size == 64:
        config.training.batch_size = 8

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint on val/test split')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pt) saved by trainer.save_checkpoint')
    parser.add_argument('--split', type=str, default=None, choices=['val', 'test'], help='Dataset split to evaluate; defaults to test for clinical datasets and val for NYUv2')
    parser.add_argument('--dataset_name', type=str, default=None, choices=['mimic3', 'eicu', 'nyuv2'], help='Override dataset name (default: config.dataset_name)')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks override (default: config.tasks_enabled)')
    parser.add_argument('--device', type=str, default=None, help='Device, e.g. cuda, cuda:0, cpu (default: config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: config.training.batch_size)')
    parser.add_argument('--num_workers', type=int, default=None, help='Num workers (default: config.training.num_workers)')
    parser.add_argument('--data_root', type=str, default=None, help='Override data root (default: config.data_root)')
    parser.add_argument('--nyuv2_proxy_stats', type=str, default=None, help='NYUv2: fixed proxy calibration JSON produced from ERM val split')
    parser.add_argument('--seed', type=int, default=None, help='Loader seed (shuffle is off for val/test; default: config)')
    parser.add_argument('--data_seed', type=int, default=None, help='Dataset-side seed (controls derived val split for MIMIC3); default: same as --seed')
    parser.add_argument('--time_row_policy', type=str, default=None, choices=['legacy_last', 'hash_random', 'min_period', 'max_period'], help='MIMIC3: how to collapse multi-row LOS/decomp listfiles for train/val (default: legacy_last)')
    parser.add_argument('--time_row_policy_test', type=str, default=None, choices=['legacy_last', 'hash_random', 'min_period', 'max_period'], help='MIMIC3: how to collapse multi-row LOS/decomp listfiles for test (default: hash_random)')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to save eval_*.json (default: ckpt dir)')
    return parser.parse_args()

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

@torch.no_grad()
def main() -> None:
    args = _parse_args()
    config = ExperimentConfig()
    if args.dataset_name is not None:
        config.dataset_name = str(args.dataset_name)
    if args.split is None:
        args.split = 'val' if str(config.dataset_name).lower() == 'nyuv2' else 'test'
    elif str(config.dataset_name).lower() == 'nyuv2' and str(args.split) == 'test':
        raise ValueError('NYUv2 public release has train/val only. Use --split val for NYUv2 evaluation.')
    _apply_dataset_defaults(config, tasks_overridden=args.tasks is not None, data_root_overridden=args.data_root is not None)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.device is not None:
        config.training.device = str(args.device)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.training.num_workers = int(args.num_workers)
    if args.seed is not None:
        config.training.seed = int(args.seed)
    if args.data_seed is not None:
        config.training.data_seed = int(args.data_seed)
    tasks_enabled = config.tasks_enabled
    if args.tasks is not None:
        tasks_enabled = [t.strip() for t in str(args.tasks).split(',') if t.strip()]
        if not tasks_enabled:
            raise ValueError('--tasks parsed to an empty list. Provide comma-separated task names.')
        config.tasks_enabled = tasks_enabled
    ckpt_path = os.path.abspath(str(args.ckpt))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    output_dir = os.path.abspath(args.output_dir) if args.output_dir is not None else os.path.dirname(ckpt_path)
    os.makedirs(output_dir, exist_ok=True)
    device = _resolve_device(config.training.device)
    data_seed = config.training.seed if config.training.data_seed is None else int(config.training.data_seed)
    dataloader_kwargs = {}
    if config.dataset_name in ('mimic3', 'eicu'):
        dataloader_kwargs['feature_dim'] = config.backbone.input_dim
        if config.dataset_name == 'mimic3' and args.time_row_policy is not None:
            dataloader_kwargs['time_row_policy'] = str(args.time_row_policy)
        if config.dataset_name == 'mimic3' and args.time_row_policy_test is not None:
            dataloader_kwargs['time_row_policy_test'] = str(args.time_row_policy_test)
    loader = build_dataloader(dataset_name=config.dataset_name, root=config.data_root, split=str(args.split), batch_size=config.training.batch_size, tasks_enabled=tasks_enabled, shuffle=False, seed=config.training.seed, dataset_seed=data_seed, num_workers=config.training.num_workers, **dataloader_kwargs)
    head_config = {}
    if config.dataset_name == 'mimic3' and 'phenotype' in tasks_enabled:
        phenotype_num_classes = int(getattr(config, 'phenotype_num_classes', 25))
        ds = getattr(loader, 'dataset', None)
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        meta_dim = None
        if ds is not None and hasattr(ds, 'metadata') and isinstance(ds.metadata, dict):
            meta_dim = ds.metadata.get('phenotype_dim')
        if meta_dim is not None and int(meta_dim) != phenotype_num_classes:
            raise ValueError(f"phenotype_num_classes({phenotype_num_classes}) != dataset.metadata['phenotype_dim']({meta_dim}). Check phenotype_labels.csv or config.py.")
        head_config['phenotype'] = {'num_phenotypes': phenotype_num_classes}
    model = DynamicLipschitzMTL(dataset_name=config.dataset_name, tasks_enabled=tasks_enabled, force_backbone_fp32=config.training.force_backbone_fp32, backbone_config=asdict(config.backbone), head_config=head_config, lip_config=asdict(config.lipschitz), nyuv2_proxy_stats_path=args.nyuv2_proxy_stats).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    metric_keys = metric_keys_for_model(raw_model)
    if raw_model.dataset_name == 'nyuv2':
        seg_head = raw_model.task_heads['seg'] if 'seg' in raw_model.task_heads else None
        num_seg_classes = int(getattr(seg_head, 'num_classes', 13))
        accumulator = NYUv2MetricAccumulator(num_seg_classes=num_seg_classes, ignore_label=-1)
        n_valid: Dict[str, int] = {t: 0 for t in raw_model.tasks_enabled}
        n_total: Dict[str, int] = {t: 0 for t in raw_model.tasks_enabled}
        for batch in tqdm(loader, desc=f'Evaluating({args.split})'):
            x = batch['x'].to(device)
            y = {k: v.to(device) for (k, v) in batch['y'].items()}
            y_mask = {k: v.to(device) if v is not None else None for (k, v) in batch['y_mask'].items()}
            outputs = model(x, y_mask=y_mask)
            preds = outputs['probs']
            if 'seg' in raw_model.tasks_enabled and 'seg' in preds and ('seg' in y):
                seg_mask = y_mask.get('seg')
                n_total['seg'] += int(y['seg'].numel())
                n_valid['seg'] += int(seg_mask.sum().item()) if seg_mask is not None else int((y['seg'] != -1).sum().item())
                accumulator.update_seg(preds['seg'].detach().cpu(), y['seg'].detach().cpu(), seg_mask.detach().cpu() if seg_mask is not None else None)
            if 'depth' in raw_model.tasks_enabled and 'depth' in preds and ('depth' in y):
                depth_mask = y_mask.get('depth')
                n_total['depth'] += int(y['depth'].numel())
                n_valid['depth'] += int(depth_mask.sum().item()) if depth_mask is not None else int((y['depth'] > 0).sum().item())
                accumulator.update_depth(preds['depth'].detach().cpu(), y['depth'].detach().cpu(), depth_mask.detach().cpu() if depth_mask is not None else None)
            if 'normal' in raw_model.tasks_enabled and 'normal' in preds and ('normal' in y):
                normal_mask = y_mask.get('normal')
                n_total['normal'] += int(y['normal'].shape[0] * y['normal'].shape[-2] * y['normal'].shape[-1])
                if normal_mask is not None:
                    n_valid['normal'] += int(normal_mask.sum().item())
                else:
                    n_valid['normal'] += int((torch.linalg.norm(y['normal'], dim=1, keepdim=True) > 0).sum().item())
                accumulator.update_normal(preds['normal'].detach().cpu(), y['normal'].detach().cpu(), normal_mask.detach().cpu() if normal_mask is not None else None)
        metrics = accumulator.compute(raw_model.tasks_enabled)
        task_scores = {t: compute_unified_score(t, metrics[t], metric_keys) for t in raw_model.tasks_enabled}
        worst_task = min(task_scores, key=task_scores.get)
        worst_score = float(task_scores[worst_task])
        macro_score = float(sum(task_scores.values()) / max(1, len(task_scores)))
        gap = float(max(task_scores.values()) - min(task_scores.values())) if task_scores else 0.0
        summary = {'ckpt': _safe_path_label(ckpt_path), 'split': str(args.split), 'dataset_name': config.dataset_name, 'data_root': '<redacted>', 'seed': int(config.training.seed), 'data_seed': int(data_seed), 'n_total': {k: int(v) for (k, v) in n_total.items()}, 'n_valid': {k: int(v) for (k, v) in n_valid.items()}, 'metrics': metrics, 'task_scores': {k: float(v) for (k, v) in task_scores.items()}, 'worst_task': worst_task, 'worst_score': worst_score, 'macro_score': macro_score, 'gap': gap}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_path = os.path.join(output_dir, f'eval_{args.split}_{ckpt_tag}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'Saved: {_safe_saved_label(out_path)}')
        return
    los_head = raw_model.task_heads['los'] if 'los' in raw_model.task_heads else None
    agg = {task: {'pred': [], 'target': [], 'mask': []} for task in raw_model.tasks_enabled}
    n_valid: Dict[str, int] = {t: 0 for t in raw_model.tasks_enabled}
    n_total: Dict[str, int] = {t: 0 for t in raw_model.tasks_enabled}
    for batch in tqdm(loader, desc=f'Evaluating({args.split})'):
        x = batch['x'].to(device)
        seq_mask = batch.get('seq_mask')
        if seq_mask is not None:
            seq_mask = seq_mask.to(device)
        y = {k: v.to(device) for (k, v) in batch['y'].items()}
        y_mask = {k: v.to(device) if v is not None else None for (k, v) in batch['y_mask'].items()}
        outputs = model(x, seq_mask, y_mask=y_mask)
        probs = outputs['probs']
        for task in raw_model.tasks_enabled:
            if task not in probs or task not in y:
                continue
            if task == 'decomp':
                if seq_mask is None or y_mask.get('decomp') is None:
                    raise RuntimeError('The decomp task requires sequence masks and sequence representations.')
                valid_decomp = seq_mask & y_mask['decomp']
                n_total[task] += int(seq_mask.sum().item())
                n_valid[task] += int(valid_decomp.sum().item())
                agg[task]['pred'].append(probs[task][valid_decomp].detach().cpu())
                agg[task]['target'].append(y[task][valid_decomp].detach().cpu())
                continue
            mask_t = y_mask.get(task)
            pred_t = probs[task]
            target_t = y[task]
            if mask_t is not None:
                if task == 'phenotype':
                    n_total[task] += int(mask_t.numel())
                    n_valid[task] += int(mask_t.sum().item())
                    agg[task]['mask'].append(mask_t.detach().cpu())
                else:
                    m = mask_t.squeeze(-1) if mask_t.dim() > 1 and mask_t.size(-1) == 1 else mask_t
                    n_total[task] += int(m.numel())
                    n_valid[task] += int(m.sum().item())
                    agg[task]['mask'].append(m.detach().cpu())
            else:
                n_total[task] += int(target_t.numel())
                n_valid[task] += int(target_t.numel())
            agg[task]['pred'].append(pred_t.detach().cpu())
            agg[task]['target'].append(target_t.detach().cpu())
    for task in raw_model.tasks_enabled:
        if not agg[task]['pred']:
            raise RuntimeError(f"evaluate: task '{task}' has no valid samples on split={args.split}")
    predictions = {}
    for task in raw_model.tasks_enabled:
        preds = torch.cat(agg[task]['pred'], dim=0)
        targets = torch.cat(agg[task]['target'], dim=0)
        masks = torch.cat(agg[task]['mask'], dim=0) if agg[task]['mask'] else None
        if task == 'los':
            if los_head is None or not getattr(los_head, 'use_discretization', True):
                raise RuntimeError('LOS uses discretized classification targets in this implementation.')
            target_bucket = los_head.discretize_target(targets.detach().cpu()).detach().cpu()
            try:
                import numpy as np
                raw_all = targets.detach().cpu().numpy().reshape(-1)
                if masks is None:
                    m_all = None
                    raw_v2 = raw_all
                    bucket_v2 = target_bucket.detach().cpu().numpy().reshape(-1)
                else:
                    m_all = masks.detach().cpu().numpy().astype(bool).reshape(-1)
                    raw_v2 = raw_all[m_all]
                    bucket_v2 = target_bucket.detach().cpu().numpy().reshape(-1)[m_all]
                boundaries = list(getattr(los_head, 'bucket_boundaries', []))
                num_buckets = int(getattr(los_head, 'num_buckets', 10))
                manual = np.zeros_like(raw_v2, dtype=np.int64)
                for (i, b) in enumerate(boundaries):
                    manual[raw_v2 >= float(b)] = int(i + 1)
                manual = np.minimum(manual, num_buckets - 1)
                mismatch = int(np.sum(manual != bucket_v2))
                if mismatch > 0:
                    if m_all is None:
                        target_bucket = torch.from_numpy(manual).to(dtype=torch.long)
                    else:
                        full = target_bucket.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=True)
                        full[m_all] = manual
                        target_bucket = torch.from_numpy(full).to(dtype=torch.long)
            except Exception:
                pass
            predictions[task] = {'pred': preds.numpy(), 'target': target_bucket.numpy(), 'mask': masks.numpy() if masks is not None else None}
            continue
        predictions[task] = {'pred': preds.numpy(), 'target': targets.numpy(), 'mask': masks.numpy() if masks is not None else None}
    metrics = evaluate_all_tasks(predictions, raw_model.tasks_enabled)
    task_scores = {t: compute_unified_score(t, metrics[t], metric_keys) for t in raw_model.tasks_enabled}
    worst_task = min(task_scores, key=task_scores.get)
    worst_score = float(task_scores[worst_task])
    macro_score = float(sum(task_scores.values()) / max(1, len(task_scores)))
    gap = float(max(task_scores.values()) - min(task_scores.values())) if task_scores else 0.0
    summary = {'ckpt': _safe_path_label(ckpt_path), 'split': str(args.split), 'dataset_name': config.dataset_name, 'data_root': '<redacted>', 'seed': int(config.training.seed), 'data_seed': int(data_seed), 'n_total': {k: int(v) for (k, v) in n_total.items()}, 'n_valid': {k: int(v) for (k, v) in n_valid.items()}, 'metrics': metrics, 'task_scores': {k: float(v) for (k, v) in task_scores.items()}, 'worst_task': worst_task, 'worst_score': worst_score, 'macro_score': macro_score, 'gap': gap}
    try:
        ds = getattr(loader, 'dataset', None)
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        if ds is not None and hasattr(ds, 'time_row_policy') and hasattr(ds, 'time_row_policy_test'):
            summary['time_row_policy'] = str(getattr(ds, 'time_row_policy'))
            summary['time_row_policy_test'] = str(getattr(ds, 'time_row_policy_test'))
    except Exception:
        pass
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(output_dir, f'eval_{args.split}_{ckpt_tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Saved: {_safe_saved_label(out_path)}')
if __name__ == '__main__':
    main()
