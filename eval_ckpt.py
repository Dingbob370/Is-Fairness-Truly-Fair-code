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
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
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
    p = argparse.ArgumentParser(description='Evaluate best_model.pt on val/test and compute Lipschitz Bias')
    p.add_argument('--runs', nargs='*', default=None, help='Run directories or checkpoint paths (.pt). If a directory is given, --ckpt_name will be appended.')
    p.add_argument('--runs_file', type=str, default=None, help='A text file containing run dirs / ckpt paths (one per line). Lines starting with # are ignored.')
    p.add_argument('--log_files', nargs='*', default=None, help="Parse training logs and extract ckpt paths from 'New best model saved:' lines.")
    p.add_argument('--ckpt_name', type=str, default='best_model.pt', help='Checkpoint filename to use when --runs provides directories (default: best_model.pt).')
    p.add_argument('--split', type=str, default=None, choices=['val', 'test'], help='Which split to evaluate; defaults to test for clinical datasets and val for NYUv2')
    p.add_argument('--dataset_name', type=str, default=None, choices=['mimic3', 'eicu', 'nyuv2'], help='Override dataset name (default: config.dataset_name)')
    p.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks override (default: config.tasks_enabled). Example: mortality,los')
    p.add_argument('--data_root', type=str, default=None, help='Override data root (default: config.data_root)')
    p.add_argument('--nyuv2_proxy_stats', type=str, default=None, help='NYUv2: fixed proxy calibration JSON produced from ERM val split')
    p.add_argument('--device', type=str, default=None, help='Device, e.g. cuda, cuda:0, cpu (default: config)')
    p.add_argument('--batch_size', type=int, default=None, help='Batch size for inference (default: config)')
    p.add_argument('--num_workers', type=int, default=None, help='Num workers (default: config)')
    p.add_argument('--data_seed', type=int, default=None, help='Dataset-side seed controlling derived val split for MIMIC3; if omitted, try parse from run name.')
    p.add_argument('--pair_seed', type=int, default=42, help='Random seed for the fixed evaluation pools/pairs used by Bias (paper low-variance protocol).')
    p.add_argument('--k_prototypes', type=int, default=None, help='Top-K high-confidence samples used to form each task prototype (default: config.lipschitz.k_prototypes).')
    p.add_argument('--eval_per_task', type=int, default=256, help='Fixed pool size per task used for pair sampling (balanced across tasks).')
    p.add_argument('--pairs_per_task_pair', type=int, default=4096, help='How many (x,y) pairs to sample per unordered task pair (i,j).')
    p.add_argument('--fixed_threshold', type=float, default=None, help='If set, also compute Bias/violation-rate under a unified constant threshold delta for ALL task pairs (used to answer: is low Bias just because distance thresholds are looser?).')
    p.add_argument('--output', type=str, default=None, help='Write aggregated results to this JSON file (default: outputs/eval_ckpt_bias.json).')
    return p.parse_args()

def _read_runs_file(path: str) -> List[str]:
    items: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            items.append(line)
    return items

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
            p = os.path.join(it_exp, ckpt_name)
            ckpts.append(p)
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

def _infer_data_seed_from_run_name(path: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(os.path.dirname(path) if path.endswith('.pt') else path)
    m_d = re.search('(?:^|_)dseed(\\\\d+)(?:_|$)', base)
    m_s = re.search('(?:^|_)seed(\\\\d+)(?:_|$)', base)
    dseed = int(m_d.group(1)) if m_d else None
    seed = int(m_s.group(1)) if m_s else None
    return (dseed, seed)

def _build_loader(*, config: ExperimentConfig, tasks_enabled: List[str], split: str, data_seed: int, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    dataloader_kwargs = {}
    if config.dataset_name in ('mimic3', 'eicu'):
        dataloader_kwargs['feature_dim'] = config.backbone.input_dim
    return build_dataloader(dataset_name=config.dataset_name, root=config.data_root, split=split, batch_size=batch_size, tasks_enabled=tasks_enabled, shuffle=False, seed=int(config.training.seed), dataset_seed=int(data_seed), num_workers=num_workers, **dataloader_kwargs)

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

def _valid_sample_mask(task: str, *, y_mask: Dict[str, torch.Tensor], seq_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if task not in y_mask or y_mask[task] is None:
        raise KeyError(f"Unknown or unsupported task: {task}")
    m = y_mask[task]
    if task == 'decomp':
        if seq_mask is None:
            raise RuntimeError('The decomp task requires sequence masks and sequence representations.')
        valid = (seq_mask & m).any(dim=1)
        return valid
    if task == 'phenotype':
        return m.any(dim=1)
    if m.dim() >= 3:
        return m.reshape(m.size(0), -1).any(dim=1)
    if m.dim() > 1 and m.size(-1) == 1:
        return m.squeeze(-1).bool()
    return m.bool()

@torch.no_grad()
def compute_lipschitz_bias(*, model: DynamicLipschitzMTL, loader, tasks_enabled: List[str], k_prototypes: int, eval_per_task: int, pairs_per_task_pair: int, pair_seed: int, fixed_threshold: Optional[float], device: torch.device, split: str) -> Dict:
    pools: Dict[str, Dict[str, List[torch.Tensor]]] = {t: {'emb': [], 'p': []} for t in tasks_enabled}
    for batch in tqdm(loader, desc=f'Collecting({split})'):
        x = batch['x'].to(device)
        seq_mask = batch.get('seq_mask')
        if seq_mask is not None:
            seq_mask = seq_mask.to(device)
        y_mask = {k: v.to(device) if v is not None else None for (k, v) in batch['y_mask'].items()}
        out = model(x, seq_mask, y_mask=y_mask, return_embeddings=True)
        emb = out.get('embeddings')
        if emb is None:
            raise RuntimeError('Model output is missing embeddings.')
        emb_cpu = emb.detach().to(dtype=torch.float32).cpu()
        conf = out.get('confidences', {})
        for task in tasks_enabled:
            if task not in conf:
                raise KeyError(f"Model output is missing confidence for task '{task}'.")
            valid = _valid_sample_mask(task, y_mask=y_mask, seq_mask=seq_mask).detach().cpu()
            if valid.numel() != emb_cpu.size(0):
                raise RuntimeError(f'valid mask shape mismatch for task={task}: {tuple(valid.shape)} vs B={emb_cpu.size(0)}')
            if int(valid.sum().item()) == 0:
                continue
            p_cpu = conf[task].detach().to(dtype=torch.float32).cpu()
            p_cpu = p_cpu.view(-1)
            pools[task]['emb'].append(emb_cpu[valid])
            pools[task]['p'].append(p_cpu[valid])
    task_emb: Dict[str, torch.Tensor] = {}
    task_p: Dict[str, torch.Tensor] = {}
    for t in tasks_enabled:
        if not pools[t]['p']:
            raise RuntimeError(f"Task '{t}' has no valid samples on split={getattr(loader.dataset, 'split', 'unknown')}. Check masks and data.")
        task_emb[t] = torch.cat(pools[t]['emb'], dim=0)
        task_p[t] = torch.cat(pools[t]['p'], dim=0)
    prototypes: Dict[str, torch.Tensor] = {}
    topk_info: Dict[str, Dict] = {}
    for t in tasks_enabled:
        p = task_p[t]
        k = int(min(int(k_prototypes), int(p.numel())))
        if k <= 0:
            raise RuntimeError(f"Task '{t}' cannot construct prototypes from an empty sample set.")
        top_idx = torch.topk(p, k=k, largest=True).indices
        proto = task_emb[t][top_idx].mean(dim=0)
        proto = proto / proto.norm(p=2).clamp(min=1e-12)
        prototypes[t] = proto
        topk_info[t] = {'n_valid': int(p.numel()), 'k': k, 'p_min_topk': float(p[top_idx].min().item())}
    tasks_sorted = list(tasks_enabled)
    d_ij: Dict[Tuple[str, str], float] = {}
    for (i, j) in combinations(tasks_sorted, 2):
        qi = prototypes[i]
        qj = prototypes[j]
        cos = float(torch.clamp(torch.dot(qi, qj), -1.0, 1.0).item())
        d = float((1.0 - cos) / 2.0)
        d_ij[i, j] = d
    rng = np.random.default_rng(int(pair_seed))
    eval_pool: Dict[str, np.ndarray] = {}
    for t in tasks_sorted:
        p = task_p[t].numpy()
        if eval_per_task <= 0 or eval_per_task >= p.shape[0]:
            eval_pool[t] = p.astype(np.float32, copy=False)
            continue
        if p.shape[0] >= eval_per_task:
            idx = rng.choice(p.shape[0], size=int(eval_per_task), replace=False)
        else:
            idx = rng.choice(p.shape[0], size=int(eval_per_task), replace=True)
        eval_pool[t] = p[idx].astype(np.float32, copy=False)
    total_sum = 0.0
    total_cnt = 0
    per_pair_mean: Dict[str, float] = {}
    per_pair_raw_gap_mean: Dict[str, float] = {}
    per_pair_violation_rate: Dict[str, float] = {}
    total_raw_gap_sum = 0.0
    total_dist_sum = 0.0
    total_violations = 0.0
    fixed_total_sum = 0.0
    fixed_total_dist_sum = 0.0
    fixed_total_violations = 0.0
    per_pair_mean_fixed: Dict[str, float] = {}
    per_pair_violation_rate_fixed: Dict[str, float] = {}
    for (i, j) in combinations(tasks_sorted, 2):
        pi = eval_pool[i]
        pj = eval_pool[j]
        if pi.size == 0 or pj.size == 0:
            raise RuntimeError(f'The evaluation pool is empty.{i}({pi.size}), {j}({pj.size})')
        a = rng.integers(0, pi.size, size=int(pairs_per_task_pair), dtype=np.int64)
        b = rng.integers(0, pj.size, size=int(pairs_per_task_pair), dtype=np.int64)
        gap = np.abs(pi[a] - pj[b])
        d = d_ij[i, j]
        hinge = np.maximum(gap - d, 0.0)
        violations = (gap > d).astype(np.float32, copy=False)
        m = float(hinge.mean()) if hinge.size > 0 else 0.0
        per_pair_mean[f'{i}__{j}'] = m
        per_pair_raw_gap_mean[f'{i}__{j}'] = float(gap.mean()) if gap.size > 0 else 0.0
        per_pair_violation_rate[f'{i}__{j}'] = float(violations.mean()) if violations.size > 0 else 0.0
        total_sum += float(hinge.sum())
        total_raw_gap_sum += float(gap.sum())
        total_dist_sum += float(d) * float(gap.size)
        total_violations += float(violations.sum())
        total_cnt += int(hinge.size)
        if fixed_threshold is not None:
            d_fix = float(fixed_threshold)
            hinge_fix = np.maximum(gap - d_fix, 0.0)
            violations_fix = (gap > d_fix).astype(np.float32, copy=False)
            per_pair_mean_fixed[f'{i}__{j}'] = float(hinge_fix.mean()) if hinge_fix.size > 0 else 0.0
            per_pair_violation_rate_fixed[f'{i}__{j}'] = float(violations_fix.mean()) if violations_fix.size > 0 else 0.0
            fixed_total_sum += float(hinge_fix.sum())
            fixed_total_dist_sum += float(d_fix) * float(gap.size)
            fixed_total_violations += float(violations_fix.sum())
    bias = float(total_sum / max(1, total_cnt))
    mean_raw_gap = float(total_raw_gap_sum / max(1, total_cnt))
    mean_distance = float(total_dist_sum / max(1, total_cnt))
    violation_rate = float(total_violations / max(1, total_cnt))
    bias_fixed = None
    mean_distance_fixed = None
    violation_rate_fixed = None
    if fixed_threshold is not None:
        bias_fixed = float(fixed_total_sum / max(1, total_cnt))
        mean_distance_fixed = float(fixed_total_dist_sum / max(1, total_cnt))
        violation_rate_fixed = float(fixed_total_violations / max(1, total_cnt))
    out = {'bias': bias, 'mean_raw_gap': mean_raw_gap, 'mean_distance': mean_distance, 'violation_rate': violation_rate, 'per_pair_bias_mean': per_pair_mean, 'per_pair_raw_gap_mean': per_pair_raw_gap_mean, 'per_pair_violation_rate': per_pair_violation_rate, 'distances': {f'{i}__{j}': float(d) for ((i, j), d) in d_ij.items()}, 'topk_info': topk_info, 'eval_pool_size': {t: int(eval_pool[t].size) for t in tasks_sorted}, 'pairs_per_task_pair': int(pairs_per_task_pair), 'pair_seed': int(pair_seed)}
    if fixed_threshold is not None:
        out.update({'fixed_threshold': float(fixed_threshold), 'bias_fixed_threshold': float(bias_fixed), 'mean_distance_fixed_threshold': float(mean_distance_fixed), 'violation_rate_fixed_threshold': float(violation_rate_fixed), 'per_pair_bias_mean_fixed_threshold': per_pair_mean_fixed, 'per_pair_violation_rate_fixed_threshold': per_pair_violation_rate_fixed})
    return out

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)

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
        raise ValueError('NYUv2 public release has train/val only. Use --split val for NYUv2 auditing.')
    if str(config.dataset_name).lower() == 'nyuv2' and args.nyuv2_proxy_stats is None:
        raise ValueError('NYUv2 requires --nyuv2_proxy_stats for fixed proxy calibration.')
    tasks_enabled = list(config.tasks_enabled)
    if args.tasks is not None:
        tasks_enabled = [t.strip() for t in str(args.tasks).split(',') if t.strip()]
        if not tasks_enabled:
            raise ValueError('--tasks parsed to an empty list. Provide comma-separated task names.')
        config.tasks_enabled = list(tasks_enabled)
    device = _resolve_device(config.training.device)
    k_prototypes = int(args.k_prototypes) if args.k_prototypes is not None else int(config.lipschitz.k_prototypes)
    results: List[Dict] = []
    for ckpt in ckpt_paths:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"ckpt not found: {_safe_path_label(ckpt)}")
        (dseed_from_name, seed_from_name) = _infer_data_seed_from_run_name(ckpt)
        if args.data_seed is not None:
            data_seed = int(args.data_seed)
        elif dseed_from_name is not None:
            data_seed = int(dseed_from_name)
        elif seed_from_name is not None:
            data_seed = int(seed_from_name)
        else:
            data_seed = int(config.training.seed)
        loader = _build_loader(config=config, tasks_enabled=tasks_enabled, split=str(args.split), data_seed=int(data_seed), batch_size=int(config.training.batch_size), num_workers=int(config.training.num_workers))
        model = _build_model(config=config, tasks_enabled=tasks_enabled, loader=loader, device=device, nyuv2_proxy_stats_path=args.nyuv2_proxy_stats)
        _load_checkpoint(model, ckpt, device)
        bias_info = compute_lipschitz_bias(model=model, loader=loader, tasks_enabled=tasks_enabled, k_prototypes=int(k_prototypes), eval_per_task=int(args.eval_per_task), pairs_per_task_pair=int(args.pairs_per_task_pair), pair_seed=int(args.pair_seed), fixed_threshold=float(args.fixed_threshold) if args.fixed_threshold is not None else None, device=device, split=str(args.split))
        run_dir = os.path.dirname(ckpt)
        run_name = os.path.basename(run_dir)
        rec = {'run': run_name, 'ckpt': _safe_path_label(ckpt), 'split': str(args.split), 'data_seed': int(data_seed), 'k_prototypes': int(k_prototypes), **bias_info}
        results.append(rec)
        msg = f"[{run_name}] bias={rec['bias']:.6f}  raw_gap={rec['mean_raw_gap']:.6f}  dist={rec['mean_distance']:.6f}  vr={rec['violation_rate']:.3f}"
        if 'bias_fixed_threshold' in rec:
            msg += f"  |  fixedδ={rec['fixed_threshold']:.3f}: bias={rec['bias_fixed_threshold']:.6f} dist={rec['mean_distance_fixed_threshold']:.6f} vr={rec['violation_rate_fixed_threshold']:.3f}"
        msg += f"  ckpt={rec['ckpt']}"
        print(msg)
    out_path = os.path.abspath(args.output) if args.output is not None else os.path.abspath(os.path.join('outputs', 'eval_ckpt_bias.json'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, ensure_ascii=False, indent=2)
    print(f'Saved: {_safe_saved_label(out_path)}')
if __name__ == '__main__':
    main()
