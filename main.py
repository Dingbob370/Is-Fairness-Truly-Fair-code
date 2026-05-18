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
import logging
import os
from dataclasses import asdict
import torch
from config import ExperimentConfig
from data_loader.build import build_dataloader
from metrics import compute_unified_score, get_task_scores, metric_keys_for_model
from models.mtl_model import DynamicLipschitzMTL
from trainer import DynamicLipschitzTrainer

def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log'), encoding='utf-8'), logging.StreamHandler()])

def _format_rstar_tag(r_star: float) -> str:
    return f'{int(round(float(r_star) * 100)):03d}'

def _format_hparam_tag(value: float) -> str:
    """
    Format a float into a short, filesystem-friendly tag.
    Examples:
      0.1   -> "0p1"
      0.01  -> "0p01"
      1e-3  -> "1e-3"
      5e-4  -> "5e-4"
    """
    v = float(value)
    if v == 0.0:
        return '0'
    if abs(v) < 0.01:
        s = f'{v:.0e}'
        s = s.replace('e-0', 'e-').replace('e+0', 'e+').replace('e+', 'e')
        return s
    return f'{v:g}'.replace('.', 'p')

def _public_config_snapshot(config: ExperimentConfig) -> dict:
    snapshot = asdict(config)
    snapshot['data_root'] = '<redacted>'
    return snapshot

def _resolve_output_dir(config: ExperimentConfig) -> str:
    r_star = float(getattr(config.band_controller, 'r_star', 0.0))
    rstar_tag = _format_rstar_tag(r_star)
    seed = int(getattr(config.training, 'seed', 0))
    data_seed_attr = getattr(config.training, 'data_seed', None)
    data_seed = seed if data_seed_attr is None else int(data_seed_attr)
    eta = float(getattr(config.band_controller, 'eta', 0.1))
    lambda_init = float(getattr(config.band_controller, 'lambda_init', 0.1))
    use_amp = bool(getattr(config.training, 'use_amp', True))
    amp_dtype = str(getattr(config.training, 'amp_dtype', 'fp16')).lower()
    dataset_prefix = ''
    if str(getattr(config, 'dataset_name', 'mimic3')).lower() != 'mimic3':
        dataset_prefix = f"{str(getattr(config, 'dataset_name')).lower()}_"
    run_name = f'{dataset_prefix}{config.experiment_name}_rstar{rstar_tag}'
    if abs(eta - 0.1) > 1e-12:
        run_name += f'_eta{_format_hparam_tag(eta)}'
    if abs(lambda_init - 0.1) > 1e-12:
        run_name += f'_lam{_format_hparam_tag(lambda_init)}'
    if not use_amp:
        run_name += '_ampoff'
    elif amp_dtype != 'fp16':
        run_name += f'_amp{amp_dtype}'
    if data_seed_attr is not None:
        run_name += f'_dseed{data_seed}'
    run_name += f'_seed{seed}'
    return os.path.join(config.output_dir, run_name)

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
    if abs(config.training.learning_rate - 0.001) < 1e-12:
        config.training.learning_rate = 0.0001

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dynamic Lipschitz MTL training entrypoint')
    parser.add_argument('--dataset_name', type=str, default=None, choices=['mimic3', 'eicu', 'nyuv2'], help='Dataset name to train on')
    parser.add_argument('--data_root', type=str, default=None, help='Override dataset root path (default: config.data_root)')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated tasks override (default: config.tasks_enabled). Example: mortality,los')
    parser.add_argument('--seed', type=int, default=None, help='Override training seed (for sweeps)')
    parser.add_argument('--data_seed', type=int, default=None, help='Dataset-side seed (e.g., fixed train/val split for MIMIC3); default: same as --seed')
    parser.add_argument('--r_star', type=float, default=None, help='Override band controller r_star (for sweeps)')
    parser.add_argument('--eta', type=float, default=None, help='Override band controller eta (for sweeps)')
    parser.add_argument('--lambda_init', type=float, default=None, help='Override band controller lambda_init')
    parser.add_argument('--amp_dtype', type=str, default=None, choices=['fp16', 'bf16'], help='Automatic mixed precision dtype')
    parser.add_argument('--device', type=str, default=None, help='Override training device, e.g. cuda, cuda:0, cpu')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Override num_workers')
    parser.add_argument('--num_epochs', type=int, default=None, help='Override num_epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--nyuv2_proxy_stats', type=str, default=None, help='NYUv2: fixed proxy calibration JSON produced from ERM val split')
    parser.add_argument('--output_dir', type=str, default=None, help='Base output directory (run subdir will be created)')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name used in output directory')
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = ExperimentConfig()
    if args.dataset_name is not None:
        config.dataset_name = str(args.dataset_name)
    _apply_dataset_defaults(config, tasks_overridden=args.tasks is not None, data_root_overridden=args.data_root is not None)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.tasks is not None:
        tasks = [t.strip() for t in str(args.tasks).split(',') if t.strip()]
        if not tasks:
            raise ValueError('--tasks parsed to an empty list. Provide comma-separated task names.')
        config.tasks_enabled = tasks
    if args.seed is not None:
        config.training.seed = int(args.seed)
    if args.data_seed is not None:
        config.training.data_seed = int(args.data_seed)
    if args.r_star is not None:
        config.band_controller.r_star = float(args.r_star)
    if args.eta is not None:
        eta = float(args.eta)
        if eta <= 0:
            raise ValueError(f'band_controller.eta must be > 0, got {eta}')
        config.band_controller.eta = eta
    if args.lambda_init is not None:
        lambda_init = float(args.lambda_init)
        if not config.band_controller.lambda_min <= lambda_init <= config.band_controller.lambda_max:
            raise ValueError(f'band_controller.lambda_init must be within [lambda_min,lambda_max]=[{config.band_controller.lambda_min},{config.band_controller.lambda_max}], got {lambda_init}')
        config.band_controller.lambda_init = lambda_init
    if args.amp_dtype is not None:
        config.training.amp_dtype = str(args.amp_dtype).lower()
    if args.device is not None:
        config.training.device = str(args.device)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.training.num_workers = int(args.num_workers)
    if args.num_epochs is not None:
        config.training.num_epochs = int(args.num_epochs)
    if args.learning_rate is not None:
        config.training.learning_rate = float(args.learning_rate)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.experiment_name is not None:
        config.experiment_name = str(args.experiment_name)
    if str(config.dataset_name).lower() == 'nyuv2' and args.nyuv2_proxy_stats is None:
        raise ValueError('NYUv2 requires --nyuv2_proxy_stats for fixed proxy calibration.')
    config.output_dir = _resolve_output_dir(config)
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f'Config: {_public_config_snapshot(config)}')
    if not config.band_controller.r_min <= config.band_controller.r_star <= config.band_controller.r_max:
        logger.warning('band_controller.r_star is outside the configured controller band.', float(config.band_controller.r_star), float(config.band_controller.r_min), float(config.band_controller.r_max))
    if str(config.training.device).startswith('cuda') and (not torch.cuda.is_available()):
        logger.warning('CUDA is not available; switching to CPU.')
        config.training.device = 'cpu'
        config.training.multi_gpu = False
    else:
        num_gpus = torch.cuda.device_count()
        logger.info(f'CUDA devices available: {num_gpus}')
        if num_gpus <= 1:
            config.training.multi_gpu = False
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    data_seed = config.training.seed if config.training.data_seed is None else int(config.training.data_seed)
    effective_batch_size = config.training.batch_size
    if config.training.multi_gpu and torch.cuda.is_available():
        num_gpus = len(config.training.gpu_ids) if config.training.gpu_ids else torch.cuda.device_count()
        logger.info(f'Effective batch size={effective_batch_size}; per-GPU batch size={effective_batch_size // num_gpus}')
    dataloader_kwargs = {}
    if config.dataset_name in ('mimic3', 'eicu'):
        dataloader_kwargs['feature_dim'] = config.backbone.input_dim
    train_loader = build_dataloader(dataset_name=config.dataset_name, root=config.data_root, split='train', batch_size=effective_batch_size, tasks_enabled=config.tasks_enabled, shuffle=True, seed=config.training.seed, dataset_seed=data_seed, num_workers=config.training.num_workers, **dataloader_kwargs)
    val_loader = build_dataloader(dataset_name=config.dataset_name, root=config.data_root, split='val', batch_size=effective_batch_size, tasks_enabled=config.tasks_enabled, shuffle=False, seed=config.training.seed, dataset_seed=data_seed, num_workers=config.training.num_workers, **dataloader_kwargs)
    head_config = {}
    if config.dataset_name == 'mimic3' and 'phenotype' in config.tasks_enabled:
        phenotype_num_classes = int(getattr(config, 'phenotype_num_classes', 25))
        ds = getattr(val_loader, 'dataset', None)
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        meta_dim = None
        if ds is not None and hasattr(ds, 'metadata') and isinstance(ds.metadata, dict):
            meta_dim = ds.metadata.get('phenotype_dim')
        if meta_dim is not None and int(meta_dim) != phenotype_num_classes:
            raise ValueError(f"phenotype_num_classes({phenotype_num_classes}) != dataset.metadata['phenotype_dim']({meta_dim}). Check phenotype_labels.csv or config.py.")
        head_config['phenotype'] = {'num_phenotypes': phenotype_num_classes}
    model = DynamicLipschitzMTL(dataset_name=config.dataset_name, tasks_enabled=config.tasks_enabled, force_backbone_fp32=config.training.force_backbone_fp32, backbone_config=asdict(config.backbone), head_config=head_config, lip_config=asdict(config.lipschitz), nyuv2_proxy_stats_path=args.nyuv2_proxy_stats)
    logger.info(f'Model parameters: {sum((p.numel() for p in model.parameters())):,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    trainer = DynamicLipschitzTrainer(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, device=config.training.device, band_config=asdict(config.band_controller), use_amp=config.training.use_amp, amp_dtype=config.training.amp_dtype, gradient_clip=config.training.gradient_clip, log_interval=config.training.log_interval, warn_threshold=config.training.warn_threshold, multi_gpu=config.training.multi_gpu, gpu_ids=config.training.gpu_ids)
    best_worst = float('-inf')
    best_worst_epoch = 0
    no_improve_rounds = 0
    best_macro = float('-inf')
    best_macro_epoch = 0
    early_stop_patience = int(getattr(config.training, 'early_stop_patience', 0))
    early_stop_min_delta = float(getattr(config.training, 'early_stop_min_delta', 0.0))
    save_best_macro = bool(getattr(config.training, 'save_best_macro', True))
    metric_keys = metric_keys_for_model(trainer._get_raw_model())
    for epoch in range(config.training.num_epochs):
        train_stats = trainer.train_epoch()
        logger.info(f'Epoch {epoch + 1} Train: {train_stats}')
        if (epoch + 1) % config.training.eval_interval == 0:
            val_metrics = trainer.evaluate()
            logger.info(f'Epoch {epoch + 1} Val: {val_metrics}')
            if not val_metrics:
                raise RuntimeError('No valid metrics or samples were produced; check masks and data.')
            task_scores = {task: compute_unified_score(task, val_metrics[task], metric_keys) for task in config.tasks_enabled}
            worst_task = min(task_scores, key=task_scores.get)
            worst_score = float(task_scores[worst_task])
            macro_score = float(sum(task_scores.values()) / max(1, len(task_scores)))
            gap = float(max(task_scores.values()) - min(task_scores.values())) if task_scores else 0.0
            logger.info(f'Task scores: {task_scores}, Worst: {worst_task}={worst_score:.4f}, Macro={macro_score:.4f}, Gap={gap:.4f}')
            improved_worst = worst_score > best_worst + early_stop_min_delta
            if improved_worst:
                best_worst = worst_score
                best_worst_epoch = epoch + 1
                no_improve_rounds = 0
                ckpt_path = os.path.join(config.output_dir, 'best_model.pt')
                trainer.save_checkpoint(ckpt_path)
                logger.info(f'New best model saved: {ckpt_path} (worst-task {worst_task}={worst_score:.4f})')
            else:
                no_improve_rounds += 1
            if save_best_macro:
                if macro_score > best_macro + early_stop_min_delta:
                    best_macro = macro_score
                    best_macro_epoch = epoch + 1
                    ckpt_path = os.path.join(config.output_dir, 'best_macro.pt')
                    trainer.save_checkpoint(ckpt_path)
                    logger.info(f'New best macro model saved: {ckpt_path} (macro={macro_score:.4f})')
            if early_stop_patience > 0 and no_improve_rounds >= early_stop_patience:
                logger.info(f'Early stopping triggered: patience={early_stop_patience} eval_rounds, min_delta={early_stop_min_delta}, best_worst={best_worst:.4f}@epoch{best_worst_epoch}, best_macro={best_macro:.4f}@epoch{best_macro_epoch}')
                break
    logger.info(f'Training completed! best_worst={best_worst:.4f}@epoch{best_worst_epoch}, best_macro={best_macro:.4f}@epoch{best_macro_epoch}')
    results_path = os.path.join(config.output_dir, 'results.txt')
    best_ckpt = os.path.join(config.output_dir, 'best_model.pt')
    if os.path.exists(best_ckpt):
        trainer.load_checkpoint(best_ckpt)
        final_metrics = trainer.evaluate()
    else:
        final_metrics = {}
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f'Method: {config.experiment_name}\n')
        f.write(f'Seed: {config.training.seed}\n')
        f.write(f'Best epoch: {best_worst_epoch}\n')
        for task in config.tasks_enabled:
            key = metric_keys[task]
            f.write(f'{task}_{key}: {final_metrics[task][key]:.6f}\n')
        task_scores = get_task_scores(final_metrics, config.tasks_enabled)
        for task in config.tasks_enabled:
            f.write(f'{task}_score: {task_scores[task]:.6f}\n')
        worst_task = min(task_scores, key=task_scores.get)
        worst_score = float(task_scores[worst_task])
        macro_score = float(sum(task_scores.values()) / len(task_scores))
        f.write(f'worst-task: {worst_task}\n')
        f.write(f'worst-task-score: {worst_score:.6f}\n')
        f.write(f'macro-score: {macro_score:.6f}\n')
if __name__ == '__main__':
    main()
