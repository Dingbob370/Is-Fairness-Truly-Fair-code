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
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from config import ExperimentConfig
from data_loader.build import build_dataloader
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
    p = argparse.ArgumentParser(description='Evaluate gradient norms on val/test (no optimizer step)')
    p.add_argument('--runs', nargs='*', default=None, help='Run directories or checkpoint paths (.pt). If a directory is given, --ckpt_name will be appended.')
    p.add_argument('--runs_file', type=str, default=None, help='Text file listing run dirs / ckpt paths (one per line)')
    p.add_argument('--log_files', nargs='*', default=None, help="Parse logs and extract ckpt paths from 'New best model saved:' lines.")
    p.add_argument('--ckpt_name', type=str, default='best_model.pt', help='ckpt filename when --runs provides directories')
    p.add_argument('--split', type=str, default=None, choices=['val', 'test'], help='Which split to run backward on; defaults to test for clinical datasets and val for NYUv2')
    p.add_argument('--dataset_name', type=str, default=None, choices=['mimic3', 'eicu', 'nyuv2'], help='Override dataset name (default: config.dataset_name)')
    p.add_argument('--data_root', type=str, default=None, help='Override data root (default: config.data_root)')
    p.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks override (default: config.tasks_enabled)')
    p.add_argument('--nyuv2_proxy_stats', type=str, default=None, help='NYUv2: fixed proxy calibration JSON produced from ERM val split')
    p.add_argument('--device', type=str, default=None, help='Device: cuda/cuda:0/cpu (default: config.training.device)')
    p.add_argument('--batch_size', type=int, default=None, help='Batch size (default: config.training.batch_size)')
    p.add_argument('--num_workers', type=int, default=None, help='Num workers (default: config.training.num_workers)')
    p.add_argument('--data_seed', type=int, default=None, help='Dataset-side seed controlling derived val split for MIMIC3; if omitted, try parse from run name.')
    p.add_argument('--mode', type=str, default='total', choices=['total', 'task_only'], help='Gradient computation mode. total: compare ||∇(L_task + λ L_lip)|| (uses λ from ckpt unless --lambda_value is set). task_only: force λ=0 and only compute ||∇L_task|| to avoid gradient-cancellation confounds.')
    p.add_argument('--lambda_value', type=float, default=None, help='Override lambda used in total loss. If omitted, use checkpoint band_controller.lambda_t when available, else 0.')
    p.add_argument('--max_batches', type=int, default=50, help='How many batches to run (default: 50)')
    p.add_argument('--output', type=str, default=None, help='Write JSON summary to this file (default: outputs/eval_grad_norm.json)')
    return p.parse_args()

def _read_runs_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            out.append(line)
    return out

def _extract_ckpts_from_logs(paths: Sequence[str], ckpt_name: str) -> List[str]:
    ckpts: List[str] = []
    pat = re.compile('New best model saved:\\s+(?P<path>\\\\S+)')
    for lp in paths:
        with open(lp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                pth = m.group('path')
                if pth.endswith(ckpt_name):
                    ckpts.append(pth)
    return ckpts

def _resolve_ckpt_paths(*, runs: Optional[Sequence[str]], runs_file: Optional[str], log_files: Optional[Sequence[str]], ckpt_name: str) -> List[str]:
    items: List[str] = []
    if runs:
        items.extend(list(runs))
    if runs_file:
        items.extend(_read_runs_file(runs_file))
    if log_files:
        items.extend(_extract_ckpts_from_logs(log_files, ckpt_name))
    if not items:
        raise ValueError('No valid run or checkpoint path was provided.')
    ckpts: List[str] = []
    for it in items:
        it_exp = os.path.expanduser(os.path.expandvars(str(it)))
        if os.path.isdir(it_exp):
            ckpts.append(os.path.join(it_exp, ckpt_name))
            continue
        if os.path.isfile(it_exp) and it_exp.endswith('.pt'):
            ckpts.append(it_exp)
            continue
        m = re.search('(?:^|\\\\s)(\\\\S+%s)(?:\\\\s|$)' % re.escape(ckpt_name), str(it))
        if m:
            ckpts.append(os.path.expanduser(os.path.expandvars(m.group(1))))
            continue
        raise FileNotFoundError(f'No valid run or checkpoint path was provided.{it}')
    seen = set()
    out: List[str] = []
    for p in ckpts:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out

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

def _infer_seed_tags_from_run(path: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(os.path.dirname(path) if path.endswith('.pt') else path)
    m_d = re.search('(?:^|_)dseed(\\\\d+)(?:_|$)', base)
    m_s = re.search('(?:^|_)seed(\\\\d+)(?:_|$)', base)
    dseed = int(m_d.group(1)) if m_d else None
    seed = int(m_s.group(1)) if m_s else None
    return (dseed, seed)

def _grad_norm_from_grads(grads) -> float:
    norms = []
    for g in grads:
        if g is None:
            continue
        norms.append(g.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), p=2).item())

def _disable_dropout_for_determinism(model: torch.nn.Module) -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
    idx = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            snapshot[f'dropout.{idx}'] = float(getattr(m, 'p', 0.0))
            m.p = 0.0
            idx += 1
        if isinstance(m, nn.RNNBase):
            snapshot[f'rnn_dropout.{id(m)}'] = float(getattr(m, 'dropout', 0.0))
            m.dropout = 0.0
    return snapshot

def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'p90': 0.0, 'max': 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {'mean': float(arr.mean()), 'std': float(arr.std()), 'p50': float(np.percentile(arr, 50)), 'p90': float(np.percentile(arr, 90)), 'max': float(arr.max())}

def _build_loader(*, config: ExperimentConfig, tasks_enabled: List[str], split: str, data_seed: int) -> torch.utils.data.DataLoader:
    dataloader_kwargs = {}
    if config.dataset_name in ('mimic3', 'eicu'):
        dataloader_kwargs['feature_dim'] = config.backbone.input_dim
    return build_dataloader(dataset_name=config.dataset_name, root=config.data_root, split=split, batch_size=config.training.batch_size, tasks_enabled=tasks_enabled, shuffle=False, seed=int(config.training.seed), dataset_seed=int(data_seed), num_workers=int(config.training.num_workers), **dataloader_kwargs)

def _build_model(*, config: ExperimentConfig, tasks_enabled: List[str], loader, device: torch.device, nyuv2_proxy_stats_path: Optional[str]) -> DynamicLipschitzMTL:
    head_config: Dict = {}
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
    model = DynamicLipschitzMTL(dataset_name=config.dataset_name, tasks_enabled=tasks_enabled, force_backbone_fp32=config.training.force_backbone_fp32, backbone_config=asdict(config.backbone), head_config=head_config, lip_config=asdict(config.lipschitz), nyuv2_proxy_stats_path=nyuv2_proxy_stats_path).to(device)
    model.eval()
    return model

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}

def compute_grad_norms(*, model: DynamicLipschitzMTL, loader, device: torch.device, max_batches: int, lambda_value: float, mode: str) -> Dict:
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    params = [p for p in raw_model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('model has no trainable parameters')
    _disable_dropout_for_determinism(raw_model)
    model.train()
    for m in raw_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
    task_norms: List[float] = []
    lip_norms: List[float] = []
    total_norms: List[float] = []
    shares: List[float] = []
    skipped_cnt = 0
    for (i, batch) in enumerate(tqdm(loader, desc='Backward(val)')):
        if i >= int(max_batches):
            break
        x = batch['x'].to(device)
        seq_mask = batch.get('seq_mask')
        if seq_mask is not None:
            seq_mask = seq_mask.to(device)
        y = {k: v.to(device) for (k, v) in batch['y'].items()}
        y_mask = {k: v.to(device) if v is not None else None for (k, v) in batch['y_mask'].items()}
        want_embeddings = mode != 'task_only'
        out = model(x, seq_mask, y_mask=y_mask, return_embeddings=want_embeddings)
        logits = out['logits']
        confidences = out.get('confidences', {})
        (L_task, _) = raw_model.compute_task_losses(logits, y, y_mask, seq_mask=seq_mask)
        if mode == 'task_only':
            grads_task = torch.autograd.grad(L_task, params, allow_unused=True)
            task_gn = _grad_norm_from_grads(grads_task)
            lip_gn = 0.0
            total_gn = task_gn
            share = 0.0
        else:
            embeddings = out.get('embeddings')
            if embeddings is None:
                raise RuntimeError('Model output is missing embeddings.')
            (L_lip, lip_stats) = raw_model.compute_lipschitz_loss(embeddings, confidences)
            skipped = bool(lip_stats.get('skipped'))
            if skipped:
                skipped_cnt += 1
            if skipped or not isinstance(L_lip, torch.Tensor) or (not L_lip.requires_grad):
                L_lip_eff = torch.tensor(0.0, device=device)
                lam_eff = 0.0
            else:
                L_lip_eff = L_lip
                lam_eff = float(lambda_value)
            total_loss = L_task + lam_eff * L_lip_eff
            grads_task = torch.autograd.grad(L_task, params, retain_graph=True, allow_unused=True)
            task_gn = _grad_norm_from_grads(grads_task)
            if lam_eff != 0.0 and L_lip_eff.requires_grad:
                grads_lip = torch.autograd.grad(lam_eff * L_lip_eff, params, retain_graph=True, allow_unused=True)
                lip_gn = _grad_norm_from_grads(grads_lip)
            else:
                lip_gn = 0.0
            grads_total = torch.autograd.grad(total_loss, params, allow_unused=True)
            total_gn = _grad_norm_from_grads(grads_total)
            share = lip_gn / (task_gn + lip_gn + 1e-12)
        task_norms.append(float(task_gn))
        lip_norms.append(float(lip_gn))
        total_norms.append(float(total_gn))
        shares.append(float(share))
    return {'n_batches': int(min(int(max_batches), len(task_norms))), 'skipped_batches': int(skipped_cnt), 'mode': str(mode), 'lambda_used': float(0.0 if mode == 'task_only' else lambda_value), 'task_grad_norm': _summarize(task_norms), 'lip_grad_norm': _summarize(lip_norms), 'total_grad_norm': _summarize(total_norms), 'grad_share': _summarize(shares)}

def main() -> None:
    args = _parse_args()
    ckpt_paths = _resolve_ckpt_paths(runs=args.runs, runs_file=args.runs_file, log_files=args.log_files, ckpt_name=str(args.ckpt_name))
    config = ExperimentConfig()
    if args.dataset_name is not None:
        config.dataset_name = str(args.dataset_name)
    _apply_dataset_defaults(config, tasks_overridden=args.tasks is not None, data_root_overridden=args.data_root is not None)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.device is not None:
        config.training.device = str(args.device)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.training.num_workers = int(args.num_workers)
    if args.split is None:
        args.split = 'val' if str(config.dataset_name).lower() == 'nyuv2' else 'test'
    if str(config.dataset_name).lower() == 'nyuv2' and str(args.split) == 'test':
        raise ValueError('NYUv2 public release has train/val only. Use --split val for NYUv2 diagnostics.')
    if str(config.dataset_name).lower() == 'nyuv2' and args.nyuv2_proxy_stats is None and (str(args.mode) != 'task_only'):
        raise ValueError('NYUv2 requires --nyuv2_proxy_stats for fixed proxy calibration.')
    tasks_enabled = list(config.tasks_enabled)
    if args.tasks is not None:
        tasks_enabled = [t.strip() for t in str(args.tasks).split(',') if t.strip()]
        if not tasks_enabled:
            raise ValueError('--tasks parsed to an empty list. Provide comma-separated task names.')
        config.tasks_enabled = list(tasks_enabled)
    device = _resolve_device(config.training.device)
    results: List[Dict] = []
    for ckpt in ckpt_paths:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"ckpt not found: {_safe_path_label(ckpt)}")
        (dseed_from_name, seed_from_name) = _infer_seed_tags_from_run(ckpt)
        if args.data_seed is not None:
            data_seed = int(args.data_seed)
        elif dseed_from_name is not None:
            data_seed = int(dseed_from_name)
        elif seed_from_name is not None:
            data_seed = int(seed_from_name)
        else:
            data_seed = int(config.training.seed)
        loader = _build_loader(config=config, tasks_enabled=tasks_enabled, split=str(args.split), data_seed=int(data_seed))
        model = _build_model(config=config, tasks_enabled=tasks_enabled, loader=loader, device=device, nyuv2_proxy_stats_path=args.nyuv2_proxy_stats)
        ckpt_obj = _load_checkpoint(model, ckpt, device)
        if str(args.mode) == 'task_only':
            lam = 0.0
        elif args.lambda_value is not None:
            lam = float(args.lambda_value)
        else:
            lam = float(ckpt_obj.get('band_controller', {}).get('lambda_t', 0.0)) if isinstance(ckpt_obj, dict) else 0.0
        stats = compute_grad_norms(model=model, loader=loader, device=device, max_batches=int(args.max_batches), lambda_value=float(lam), mode=str(args.mode))
        run_dir = os.path.dirname(ckpt)
        run_name = os.path.basename(run_dir)
        rec = {'run': run_name, 'ckpt': _safe_path_label(ckpt), 'split': str(args.split), 'data_seed': int(data_seed), **stats}
        results.append(rec)
        print(f"[{run_name}] total_grad_norm_mean={rec['total_grad_norm']['mean']:.6g}  lambda={rec['lambda_used']:.6g}")
    out_path = os.path.abspath(args.output) if args.output is not None else os.path.abspath(os.path.join('outputs', 'eval_grad_norm.json'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, ensure_ascii=False, indent=2)
    print(f'Saved: {_safe_saved_label(out_path)}')
if __name__ == '__main__':
    main()
