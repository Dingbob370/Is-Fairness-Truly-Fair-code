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
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.band_controller import BandController
from models.mtl_model import DynamicLipschitzMTL
from metrics import NYUv2MetricAccumulator, evaluate_all_tasks

class BatchValidator:

    def __init__(self, tasks_enabled: list, warn_threshold: int=5):
        self.tasks_enabled = tasks_enabled
        self.warn_threshold = int(warn_threshold)
        self.empty_batch_count = {task: 0 for task in tasks_enabled}

    def _mask_sum(self, task: str, y_mask: Dict[str, torch.Tensor], seq_mask: Optional[torch.Tensor]) -> int:
        if task not in y_mask or y_mask[task] is None:
            return 0
        m = y_mask[task]
        if task == 'decomp':
            if seq_mask is None:
                return 0
            valid_decomp = seq_mask & m
            return int(valid_decomp.sum().item())
        return int(m.sum().item()) if isinstance(m, torch.Tensor) else int(bool(m))

    def check(self, y_mask: Dict[str, torch.Tensor], seq_mask: Optional[torch.Tensor], step: int, logger) -> None:
        warnings = []
        for task in self.tasks_enabled:
            valid = self._mask_sum(task, y_mask, seq_mask)
            if valid == 0:
                self.empty_batch_count[task] += 1
                if self.empty_batch_count[task] <= 3:
                    warnings.append(f'{task} has no valid samples')
                if self.empty_batch_count[task] >= self.warn_threshold:
                    raise RuntimeError(f"Task '{task}' had {self.warn_threshold} consecutive empty batches. Check your data loading!")
            else:
                self.empty_batch_count[task] = 0
        if warnings:
            logger.warning(f"Step {step}: {', '.join(warnings)}")

class DynamicLipschitzTrainer:

    def __init__(self, model: DynamicLipschitzMTL, train_loader: DataLoader, val_loader: Optional[DataLoader]=None, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, device: str='cuda', band_config: Optional[Dict]=None, processor=None, use_lipschitz: bool=True, use_amp: bool=True, amp_dtype: str='fp16', gradient_clip: float=1.0, log_interval: int=100, warn_threshold: int=5, multi_gpu: bool=False, gpu_ids: Optional[List[int]]=None):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.use_amp = bool(use_amp) and str(device).startswith('cuda')
        self.amp_dtype = self._parse_amp_dtype(amp_dtype)
        if self.use_amp and self.amp_dtype == torch.bfloat16:
            bf16_supported = getattr(torch.cuda, 'is_bf16_supported', None)
            if callable(bf16_supported) and torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()):
                self.logger.warning('bf16 autocast is not supported on this GPU; falling back to fp16.')
                self.amp_dtype = torch.float16
        self.gradient_clip = float(gradient_clip)
        self.log_interval = int(log_interval)
        self.multi_gpu = multi_gpu
        self.gpu_ids = gpu_ids
        self.processor = processor
        self.use_lipschitz = bool(use_lipschitz)
        self.model = self._setup_model(model)
        if self.processor is not None and hasattr(self.processor, 'to'):
            self.processor.to(self.device)
        elif self.processor is not None and hasattr(self.processor, 'device'):
            self.processor.device = self.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        if optimizer is None:
            params = list(self._get_raw_model().parameters())
            if self.processor is not None and hasattr(self.processor, 'get_extra_params'):
                params.extend(list(self.processor.get_extra_params()))
            optimizer = torch.optim.Adam(params, lr=0.001)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=self.use_grad_scaler)
        self.amp_overflow_count = 0
        self.band_controller = BandController(**band_config or {})
        self.global_step = 0
        self.epoch = 0
        self.batch_validator = BatchValidator(self._get_raw_model().tasks_enabled, warn_threshold=warn_threshold)
        self._log_gpu_info()

    def _setup_model(self, model: DynamicLipschitzMTL) -> nn.Module:
        if not torch.cuda.is_available():
            self.logger.warning('CUDA is not available; switching to CPU.')
            self.multi_gpu = False
            return model.to('cpu')
        num_gpus = torch.cuda.device_count()
        if self.multi_gpu and num_gpus > 1:
            if self.gpu_ids is None:
                self.gpu_ids = list(range(num_gpus))
            else:
                original_gpu_ids = list(self.gpu_ids)
                self.gpu_ids = [i for i in self.gpu_ids if 0 <= i < num_gpus]
                if not self.gpu_ids:
                    raise ValueError(f'Invalid gpu_ids={original_gpu_ids}; available CUDA device ids are 0..{num_gpus - 1}.')
            if len(self.gpu_ids) > 1:
                self.logger.info(f'Using DataParallel on GPU ids: {self.gpu_ids}')
                model = model.to(f'cuda:{self.gpu_ids[0]}')
                model = nn.DataParallel(model, device_ids=self.gpu_ids)
                self.device = f'cuda:{self.gpu_ids[0]}'
                return model
            if len(self.gpu_ids) == 1:
                self.multi_gpu = False
                self.device = f'cuda:{self.gpu_ids[0]}'
                self.logger.info(f'Using GPU id: {self.gpu_ids[0]}')
                return model.to(self.device)
        self.multi_gpu = False
        model = model.to(self.device)
        return model

    def _get_raw_model(self) -> DynamicLipschitzMTL:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _log_gpu_info(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.logger.info(f'CUDA devices available: {num_gpus}')
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024 ** 3
                self.logger.info(f'  GPU {i}: {props.name} ({mem_gb:.1f} GB)')
            if self.multi_gpu:
                self.logger.info(f'Using DataParallel on GPU ids: {self.gpu_ids}')
            else:
                self.logger.info(f'Using device: {self.device}')
        else:
            self.logger.info('CUDA is not available; using CPU.')

    @staticmethod
    def _parse_amp_dtype(amp_dtype: str) -> torch.dtype:
        value = str(amp_dtype).strip().lower()
        if value in {'fp16', 'float16', 'half', '16'}:
            return torch.float16
        if value in {'bf16', 'bfloat16'}:
            return torch.bfloat16
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype!r} (expected 'fp16' or 'bf16')")

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
        tensor_detached = tensor.detach()
        nan_count = int(torch.isnan(tensor_detached).sum().item())
        inf_count = int(torch.isinf(tensor_detached).sum().item())
        finite_mask = torch.isfinite(tensor_detached)
        stats: Dict[str, Any] = {'dtype': str(tensor_detached.dtype), 'shape': tuple(tensor_detached.shape), 'nan': nan_count, 'inf': inf_count}
        if finite_mask.any():
            finite_vals = tensor_detached[finite_mask]
            stats['min'] = float(finite_vals.min().item())
            stats['max'] = float(finite_vals.max().item())
        else:
            stats['min'] = float('nan')
            stats['max'] = float('nan')
        return stats

    @staticmethod
    def _meta_brief(meta: Optional[Dict]) -> str:
        if not isinstance(meta, dict) or not meta:
            return ''
        parts: List[str] = []
        split = meta.get('split')
        if split is not None:
            parts.append(f'split={split}')
        return ', '.join(parts)

    def _assert_finite(self, name: str, tensor: torch.Tensor, meta: Optional[Dict]=None) -> None:
        if torch.isfinite(tensor).all():
            return
        stats = self._tensor_stats(tensor)
        meta_str = self._meta_brief(meta)
        msg = f'Non-finite tensor: {name} at step={self.global_step}. {stats}'
        if meta_str:
            msg += f' | meta: {meta_str}'
        self.logger.error(msg)
        raise FloatingPointError(msg)

    def train_epoch(self) -> Dict:
        self.model.train()
        self.epoch += 1
        epoch_stats = {'task_loss': [], 'lip_loss': [], 'total_loss': [], 'lambda_t': [], 'violation_rate': []}
        per_task_losses: Dict[str, List[float]] = {t: [] for t in self._get_raw_model().tasks_enabled}
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for (_, batch) in enumerate(pbar):
            self.global_step += 1
            x = batch['x'].to(self.device)
            seq_mask = batch.get('seq_mask')
            if seq_mask is not None:
                seq_mask = seq_mask.to(self.device)
            y = {k: v.to(self.device) for (k, v) in batch['y'].items()}
            y_mask = {k: v.to(self.device) if v is not None else None for (k, v) in batch['y_mask'].items()}
            meta = batch.get('meta')
            self.batch_validator.check(y_mask, seq_mask, self.global_step, self.logger)
            raw_model = self._get_raw_model()
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(x, seq_mask, y_mask=y_mask, return_embeddings=self.use_lipschitz)
                logits = outputs['logits']
            self._assert_finite('x', x, meta)
            for (task_name, task_logits) in logits.items():
                self._assert_finite(f'logits[{task_name}]', task_logits, meta)
            logits_fp32 = {k: v.float() for (k, v) in logits.items()}
            (task_total, task_losses) = raw_model.compute_task_losses(logits_fp32, y, y_mask, seq_mask=seq_mask)
            if self.processor is None:
                L_task = task_total
                is_grad_surgery = False
            elif getattr(self.processor, 'requires_manual_backward', lambda : False)():
                L_task = sum(task_losses.values())
                is_grad_surgery = True
            else:
                L_task = self.processor.process(task_losses, model=raw_model)
                is_grad_surgery = False
            self._assert_finite('L_task', L_task, meta)
            for (task_name, loss_value) in task_losses.items():
                per_task_losses.setdefault(task_name, []).append(float(loss_value.detach().item()))
            if self.use_lipschitz:
                confidences = outputs['confidences']
                embeddings = outputs['embeddings']
                with autocast(enabled=False):
                    embeddings_fp32 = embeddings.float()
                    confidences_fp32 = {k: v.float() for (k, v) in confidences.items()}
                    (L_lip, lip_stats) = raw_model.compute_lipschitz_loss(embeddings_fp32, confidences_fp32)
                self._assert_finite('embeddings', embeddings_fp32, meta)
                for (task_name, conf) in confidences_fp32.items():
                    self._assert_finite(f'confidence[{task_name}]', conf, meta)
                self._assert_finite('L_lip', L_lip, meta)
                if lip_stats.get('skipped'):
                    r_t = float(self.band_controller.r_t_ema) if self.band_controller.r_t_ema is not None else 0.0
                    lambda_t = float(self.band_controller.lambda_t)
                    total_loss = L_task
                else:
                    r_t = float(lip_stats.get('violation_rate', 0.0))
                    lambda_t = float(self.band_controller.update(r_t))
                    total_loss = L_task + lambda_t * L_lip
            else:
                L_lip = torch.tensor(0.0, device=self.device)
                lip_stats = {'violation_rate': 0.0, 'n_pairs': 0}
                r_t = 0.0
                lambda_t = 0.0
                total_loss = L_task
            self._assert_finite('total_loss', total_loss, meta)
            if self.use_lipschitz and self.global_step % self.log_interval == 0 and (not is_grad_surgery) and (not lip_stats.get('skipped')):
                raw_params = [p for p in self._get_raw_model().parameters() if p.requires_grad]

                def _grad_norm(grads) -> float:
                    norms = []
                    for g in grads:
                        if g is None:
                            continue
                        norms.append(g.detach().float().norm(2))
                    if not norms:
                        return 0.0
                    return float(torch.norm(torch.stack(norms), p=2).item())
                try:
                    grads_task = torch.autograd.grad(L_task, raw_params, retain_graph=True, allow_unused=True)
                    grads_lip = torch.autograd.grad(lambda_t * L_lip, raw_params, retain_graph=True, allow_unused=True)
                    grad_task_norm = _grad_norm(grads_task)
                    grad_lip_norm = _grad_norm(grads_lip)
                    grad_share = grad_lip_norm / (grad_task_norm + grad_lip_norm + 1e-12)
                except RuntimeError as e:
                    self.logger.warning(f'grad_share computation failed at step={self.global_step}: {e}')
                    (grad_task_norm, grad_lip_norm, grad_share) = (0.0, 0.0, 0.0)
                lip_stats = dict(lip_stats)
                lip_stats['grad_task_norm'] = grad_task_norm
                lip_stats['grad_lip_norm'] = grad_lip_norm
                lip_stats['grad_share'] = grad_share
            if not is_grad_surgery:
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_grad_scaler:
                    scale_before = float(self.scaler.get_scale())
                    self.scaler.scale(total_loss).backward()
                    if self.gradient_clip and self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self._get_raw_model().parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = float(self.scaler.get_scale())
                    if scale_after < scale_before:
                        self.amp_overflow_count += 1
                else:
                    total_loss.backward()
                    if self.gradient_clip and self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._get_raw_model().parameters(), self.gradient_clip)
                    self.optimizer.step()
            else:
                stats = self.processor.backward_and_step(task_losses, raw_model, self.optimizer)
                total_loss = torch.tensor(stats.get('total_loss', 0.0), device=self.device)
            epoch_stats['task_loss'].append(float(L_task.item()))
            epoch_stats['lip_loss'].append(float(L_lip.item()))
            epoch_stats['total_loss'].append(float(total_loss.item()))
            epoch_stats['lambda_t'].append(lambda_t)
            epoch_stats['violation_rate'].append(r_t)
            pbar.set_postfix({'L_task': f'{L_task.item():.4f}', 'L_lip': f'{L_lip.item():.4f}', 'λ': f'{lambda_t:.3f}', 'r_t': f'{r_t:.3f}'})
            if self.global_step % self.log_interval == 0:
                if self.use_lipschitz:
                    self._log_diagnostics(lip_stats)
        if self.scheduler is not None:
            self.scheduler.step()
        stats = {k: float(np.mean(v)) for (k, v) in epoch_stats.items()}
        stats.update({f'task_loss/{k}': float(np.mean(v)) if v else 0.0 for (k, v) in per_task_losses.items()})
        return stats

    @torch.no_grad()
    def evaluate(self) -> Dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        raw_model = self._get_raw_model()
        if raw_model.dataset_name == 'nyuv2':
            seg_head = raw_model.task_heads['seg'] if 'seg' in raw_model.task_heads else None
            num_seg_classes = int(getattr(seg_head, 'num_classes', 13))
            accumulator = NYUv2MetricAccumulator(num_seg_classes=num_seg_classes, ignore_label=-1)
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                x = batch['x'].to(self.device)
                y = {k: v.to(self.device) for (k, v) in batch['y'].items()}
                y_mask = {k: v.to(self.device) if v is not None else None for (k, v) in batch['y_mask'].items()}
                outputs = self.model(x, y_mask=y_mask)
                preds = outputs['probs']
                if 'seg' in raw_model.tasks_enabled and 'seg' in preds and ('seg' in y):
                    accumulator.update_seg(preds['seg'].detach().cpu(), y['seg'].detach().cpu(), y_mask.get('seg').detach().cpu() if y_mask.get('seg') is not None else None)
                if 'depth' in raw_model.tasks_enabled and 'depth' in preds and ('depth' in y):
                    accumulator.update_depth(preds['depth'].detach().cpu(), y['depth'].detach().cpu(), y_mask.get('depth').detach().cpu() if y_mask.get('depth') is not None else None)
                if 'normal' in raw_model.tasks_enabled and 'normal' in preds and ('normal' in y):
                    accumulator.update_normal(preds['normal'].detach().cpu(), y['normal'].detach().cpu(), y_mask.get('normal').detach().cpu() if y_mask.get('normal') is not None else None)
            return accumulator.compute(raw_model.tasks_enabled)
        los_head = raw_model.task_heads['los'] if 'los' in raw_model.task_heads else None
        phenotype_head = raw_model.task_heads['phenotype'] if 'phenotype' in raw_model.task_heads else None
        mortality_head = raw_model.task_heads['mortality'] if 'mortality' in raw_model.task_heads else None
        decomp_head = raw_model.task_heads['decomp'] if 'decomp' in raw_model.task_heads else None
        agg = {task: {'pred': [], 'target': [], 'mask': []} for task in raw_model.tasks_enabled}
        for batch in tqdm(self.val_loader, desc='Evaluating'):
            x = batch['x'].to(self.device)
            seq_mask = batch.get('seq_mask')
            if seq_mask is not None:
                seq_mask = seq_mask.to(self.device)
            y = {k: v.to(self.device) for (k, v) in batch['y'].items()}
            y_mask = {k: v.to(self.device) if v is not None else None for (k, v) in batch['y_mask'].items()}
            outputs = self.model(x, seq_mask, y_mask=y_mask)
            probs = outputs['probs']
            for task in raw_model.tasks_enabled:
                if task not in probs or task not in y:
                    continue
                if task == 'decomp':
                    valid_decomp = seq_mask & y_mask['decomp']
                    pred_flat = probs[task][valid_decomp].detach().cpu()
                    target_flat = y[task][valid_decomp].detach().cpu()
                    agg[task]['pred'].append(pred_flat)
                    agg[task]['target'].append(target_flat)
                    continue
                mask_t = y_mask.get(task)
                pred_t = probs[task]
                target_t = y[task]
                if mask_t is not None:
                    if task == 'phenotype':
                        agg[task]['mask'].append(mask_t.detach().cpu())
                    else:
                        m = mask_t.squeeze(-1) if mask_t.dim() > 1 and mask_t.size(-1) == 1 else mask_t
                        agg[task]['mask'].append(m.detach().cpu())
                agg[task]['pred'].append(pred_t.detach().cpu())
                agg[task]['target'].append(target_t.detach().cpu())
        for task in raw_model.tasks_enabled:
            if not agg[task]['pred']:
                raise RuntimeError(f"evaluate(): task '{task}No valid metrics or samples were produced; check masks and data.")
        predictions: Dict[str, Dict] = {}
        for task in raw_model.tasks_enabled:
            preds = torch.cat(agg[task]['pred'], dim=0)
            targets = torch.cat(agg[task]['target'], dim=0)
            masks = torch.cat(agg[task]['mask'], dim=0) if agg[task]['mask'] else None
            if task == 'los':
                if los_head is None or not getattr(los_head, 'use_discretization', True):
                    raise RuntimeError('LOS uses discretized classification targets in this implementation.')
                target_bucket = los_head.discretize_target(targets.detach().cpu()).detach().cpu()
                predictions[task] = {'pred': preds.numpy(), 'target': target_bucket.numpy(), 'mask': masks.numpy() if masks is not None else None}
                continue
            predictions[task] = {'pred': preds.numpy(), 'target': targets.numpy(), 'mask': masks.numpy() if masks is not None else None}
        return evaluate_all_tasks(predictions, raw_model.tasks_enabled)

    def _log_diagnostics(self, lip_stats: Dict) -> None:
        bc_diag = self.band_controller.get_diagnostics().as_dict()
        skipped = lip_stats.get('skipped')
        skipped_str = f', skipped={skipped}' if skipped else ''
        amp_str = ''
        if self.use_amp:
            dtype_str = 'bf16' if self.amp_dtype == torch.bfloat16 else 'fp16'
            if self.use_grad_scaler:
                scale = float(self.scaler.get_scale())
                amp_str = f', amp_dtype={dtype_str}, amp_scale={scale:.3g}, amp_overflow={self.amp_overflow_count}'
            else:
                amp_str = f', amp_dtype={dtype_str}'
        r_t = float(lip_stats.get('violation_rate', 0.0))
        r_ema = bc_diag.get('r_t_ema')
        r_ema_str = f'{float(r_ema):.3f}' if r_ema is not None else 'None'
        r_err = float(r_ema) - float(self.band_controller.r_star) if r_ema is not None else 0.0
        grad_share = lip_stats.get('grad_share')
        grad_share_str = 'None' if grad_share is None else f'{float(grad_share) * 100:.1f}%'
        self.logger.info(f"Step {self.global_step}: r_t={r_t:.3f}, r_ema={r_ema_str}, lambda={bc_diag['lambda_t']:.4f}, err={r_err:+.4f}, in_band={bc_diag['is_r_in_band']}, saturated={bc_diag['is_lambda_saturated']}, warmup={bc_diag['is_warmup']}, grad_share={grad_share_str}, tau={lip_stats.get('tau_t', 0):.4f}, kappa={lip_stats.get('kappa_t', 0):.4f}, healthy={bc_diag['is_healthy']}{skipped_str}{amp_str}")

    def save_checkpoint(self, path: str) -> None:
        raw_model = self._get_raw_model()
        torch.save({'epoch': self.epoch, 'global_step': self.global_step, 'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'processor_state_dict': self.processor.state_dict() if self.processor is not None else None, 'band_controller': {'lambda_t': self.band_controller.lambda_t, 'r_t_ema': self.band_controller.r_t_ema, 'step_count': self.band_controller.step_count}}, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        raw_model = self._get_raw_model()
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.processor is not None and checkpoint.get('processor_state_dict') is not None:
            self.processor.load_state_dict(checkpoint['processor_state_dict'])
        bc = checkpoint.get('band_controller', {})
        self.band_controller.lambda_t = bc.get('lambda_t', self.band_controller.lambda_t)
        self.band_controller.r_t_ema = bc.get('r_t_ema', self.band_controller.r_t_ema)
        self.band_controller.step_count = bc.get('step_count', self.band_controller.step_count)
