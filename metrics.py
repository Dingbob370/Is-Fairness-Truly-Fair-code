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
from typing import Dict, List
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
METRIC_KEYS = {'mortality': 'auc', 'decomp': 'auc', 'los': 'kappa', 'phenotype': 'auc', 'seg': 'miou', 'depth': 'rmse', 'normal': 'mean'}

def metric_keys_for_model(model) -> Dict[str, str]:
    return {task: METRIC_KEYS[task] for task in model.tasks_enabled}

def _to_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, 'detach'):
        array = array.detach().cpu().numpy()
    return np.asarray(array)

def _squeeze_channel_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    return mask

def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray=None) -> float:
    if mask is not None:
        y_true = y_true[mask]
        y_prob = y_prob[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5

def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray=None) -> float:
    num_labels = y_true.shape[1]
    aucs = []
    for k in range(num_labels):
        if mask is not None:
            valid_k = mask[:, k].astype(bool)
        else:
            valid_k = np.ones(len(y_true), dtype=bool)
        if valid_k.sum() == 0:
            continue
        y_k = y_true[valid_k, k]
        p_k = y_prob[valid_k, k]
        if len(np.unique(y_k)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_k, p_k)))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5

def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray=None) -> float:
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 0.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights='linear'))
    except Exception:
        return 0.0

def compute_segmentation_metrics(pred, target, mask=None, num_classes: int=13, ignore_label: int=-1) -> Dict[str, float]:
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target).astype(np.int64)
    if pred_np.ndim == 4:
        pred_np = pred_np.argmax(axis=1)
    pred_np = pred_np.astype(np.int64)
    if mask is None:
        valid = target_np != ignore_label
    else:
        valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
        valid = valid & (target_np != ignore_label)
    valid = valid & (target_np >= 0) & (target_np < num_classes) & (pred_np >= 0) & (pred_np < num_classes)
    if valid.sum() == 0:
        return {'miou': 0.0, 'pix_acc': 0.0}
    hist = np.bincount(num_classes * target_np[valid] + pred_np[valid], minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(hist).astype(np.float64)
    denom = hist.sum(axis=1) + hist.sum(axis=0) - tp
    valid_classes = denom > 0
    ious = tp[valid_classes] / np.clip(denom[valid_classes], a_min=1.0, a_max=None)
    pix_acc = float(tp.sum() / np.clip(hist.sum(), a_min=1.0, a_max=None))
    return {'miou': float(ious.mean()) if valid_classes.any() else 0.0, 'pix_acc': pix_acc}

def compute_depth_metrics(pred, target, mask=None) -> Dict[str, float]:
    pred_np = _to_numpy(pred).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    if pred_np.ndim == 4 and pred_np.shape[1] == 1:
        pred_np = pred_np[:, 0]
    if target_np.ndim == 4 and target_np.shape[1] == 1:
        target_np = target_np[:, 0]
    if mask is None:
        valid = target_np > 0
    else:
        valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
    if valid.sum() == 0:
        return {'rmse': float('inf')}
    diff = pred_np[valid] - target_np[valid]
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    return {'rmse': rmse}

def compute_normal_metrics(pred, target, mask=None) -> Dict[str, float]:
    pred_np = _to_numpy(pred).astype(np.float64)
    target_np = _to_numpy(target).astype(np.float64)
    pred_norm = np.linalg.norm(pred_np, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target_np, axis=1, keepdims=True)
    pred_unit = pred_np / np.clip(pred_norm, a_min=1e-12, a_max=None)
    target_unit = target_np / np.clip(target_norm, a_min=1e-12, a_max=None)
    dots = np.clip(np.sum(pred_unit * target_unit, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    if mask is None:
        valid = target_norm[:, 0] > 0
    else:
        valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
    valid = valid & (target_norm[:, 0] > 0) & (pred_norm[:, 0] > 0)
    if valid.sum() == 0:
        return {'mean': float('inf'), '11.25': 0.0, '22.5': 0.0, '30': 0.0}
    valid_angles = angles[valid]
    return {'mean': float(valid_angles.mean()), '11.25': float((valid_angles <= 11.25).mean()), '22.5': float((valid_angles <= 22.5).mean()), '30': float((valid_angles <= 30.0).mean())}

class NYUv2MetricAccumulator:

    def __init__(self, num_seg_classes: int=13, ignore_label: int=-1):
        self.num_seg_classes = int(num_seg_classes)
        self.ignore_label = int(ignore_label)
        self.seg_confmat = np.zeros((self.num_seg_classes, self.num_seg_classes), dtype=np.int64)
        self.seg_total = 0
        self.depth_sse = 0.0
        self.depth_count = 0
        self.normal_sum_angle = 0.0
        self.normal_count = 0
        self.normal_within_1125 = 0
        self.normal_within_225 = 0
        self.normal_within_30 = 0

    def update_seg(self, pred, target, mask=None) -> None:
        pred_np = _to_numpy(pred)
        target_np = _to_numpy(target).astype(np.int64)
        if pred_np.ndim == 4:
            pred_np = pred_np.argmax(axis=1)
        pred_np = pred_np.astype(np.int64)
        if mask is None:
            valid = target_np != self.ignore_label
        else:
            valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
            valid = valid & (target_np != self.ignore_label)
        valid = valid & (target_np >= 0) & (target_np < self.num_seg_classes) & (pred_np >= 0) & (pred_np < self.num_seg_classes)
        if valid.sum() == 0:
            return
        hist = np.bincount(self.num_seg_classes * target_np[valid] + pred_np[valid], minlength=self.num_seg_classes * self.num_seg_classes).reshape(self.num_seg_classes, self.num_seg_classes)
        self.seg_confmat += hist
        self.seg_total += int(valid.sum())

    def update_depth(self, pred, target, mask=None) -> None:
        pred_np = _to_numpy(pred).astype(np.float64)
        target_np = _to_numpy(target).astype(np.float64)
        if pred_np.ndim == 4 and pred_np.shape[1] == 1:
            pred_np = pred_np[:, 0]
        if target_np.ndim == 4 and target_np.shape[1] == 1:
            target_np = target_np[:, 0]
        if mask is None:
            valid = target_np > 0
        else:
            valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
        if valid.sum() == 0:
            return
        diff = pred_np[valid] - target_np[valid]
        self.depth_sse += float(np.square(diff).sum())
        self.depth_count += int(valid.sum())

    def update_normal(self, pred, target, mask=None) -> None:
        pred_np = _to_numpy(pred).astype(np.float64)
        target_np = _to_numpy(target).astype(np.float64)
        pred_norm = np.linalg.norm(pred_np, axis=1, keepdims=True)
        target_norm = np.linalg.norm(target_np, axis=1, keepdims=True)
        pred_unit = pred_np / np.clip(pred_norm, a_min=1e-12, a_max=None)
        target_unit = target_np / np.clip(target_norm, a_min=1e-12, a_max=None)
        dots = np.clip(np.sum(pred_unit * target_unit, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        if mask is None:
            valid = target_norm[:, 0] > 0
        else:
            valid = _squeeze_channel_mask(_to_numpy(mask).astype(bool))
        valid = valid & (target_norm[:, 0] > 0) & (pred_norm[:, 0] > 0)
        if valid.sum() == 0:
            return
        valid_angles = angles[valid]
        self.normal_sum_angle += float(valid_angles.sum())
        self.normal_count += int(valid_angles.size)
        self.normal_within_1125 += int((valid_angles <= 11.25).sum())
        self.normal_within_225 += int((valid_angles <= 22.5).sum())
        self.normal_within_30 += int((valid_angles <= 30.0).sum())

    def compute(self, tasks_enabled: List[str]) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        if 'seg' in tasks_enabled:
            tp = np.diag(self.seg_confmat).astype(np.float64)
            denom = self.seg_confmat.sum(axis=1) + self.seg_confmat.sum(axis=0) - tp
            valid_classes = denom > 0
            ious = tp[valid_classes] / np.clip(denom[valid_classes], a_min=1.0, a_max=None)
            pix_acc = float(tp.sum() / np.clip(self.seg_confmat.sum(), a_min=1.0, a_max=None))
            metrics['seg'] = {'miou': float(ious.mean()) if valid_classes.any() else 0.0, 'pix_acc': pix_acc}
        if 'depth' in tasks_enabled:
            rmse = float(np.sqrt(self.depth_sse / self.depth_count)) if self.depth_count > 0 else float('inf')
            metrics['depth'] = {'rmse': rmse}
        if 'normal' in tasks_enabled:
            if self.normal_count > 0:
                mean_angle = float(self.normal_sum_angle / self.normal_count)
                metrics['normal'] = {'mean': mean_angle, '11.25': float(self.normal_within_1125 / self.normal_count), '22.5': float(self.normal_within_225 / self.normal_count), '30': float(self.normal_within_30 / self.normal_count)}
            else:
                metrics['normal'] = {'mean': float('inf'), '11.25': 0.0, '22.5': 0.0, '30': 0.0}
        return metrics

def compute_unified_score(task: str, task_metrics: Dict, metric_keys: Dict[str, str]) -> float:
    key = metric_keys.get(task, 'auc')
    if key not in task_metrics:
        raise KeyError(f"Task '{task}' missing required metric key '{key}'. Got: {list(task_metrics.keys())}")
    value = float(task_metrics[key])
    if not np.isfinite(value):
        return 0.0
    if task == 'los' and key == 'kappa':
        value = (value + 1.0) / 2.0
        value = float(np.clip(value, 0.0, 1.0))
    elif task == 'depth' and key == 'rmse':
        value = float(1.0 / (1.0 + max(value, 0.0)))
    elif task == 'normal' and key == 'mean':
        value = float(1.0 - np.clip(value, 0.0, 180.0) / 180.0)
    return value

def compute_worst_task_score(val_metrics: Dict[str, Dict], tasks: List[str]) -> float:
    task_scores = get_task_scores(val_metrics, tasks)
    if not task_scores:
        return 0.0
    return min(task_scores.values())

def compute_macro_score(val_metrics: Dict[str, Dict], tasks: List[str]) -> float:
    task_scores = get_task_scores(val_metrics, tasks)
    if not task_scores:
        return 0.0
    return sum(task_scores.values()) / len(task_scores)

def get_worst_task_name(val_metrics: Dict[str, Dict], tasks: List[str]) -> str:
    task_scores = get_task_scores(val_metrics, tasks)
    if not task_scores:
        return ''
    return min(task_scores, key=task_scores.get)

def get_task_scores(val_metrics: Dict[str, Dict], tasks: List[str]) -> Dict[str, float]:
    metric_keys = {t: METRIC_KEYS[t] for t in tasks}
    task_scores: Dict[str, float] = {}
    for task in tasks:
        if task not in val_metrics:
            raise KeyError(f"val_metrics missing task '{task}'. Got: {list(val_metrics.keys())}")
        task_scores[task] = compute_unified_score(task, val_metrics[task], metric_keys)
    return task_scores

def evaluate_all_tasks(predictions: Dict[str, Dict], tasks_enabled: List[str]) -> Dict[str, Dict]:
    metrics = {}
    for task in tasks_enabled:
        if task not in predictions:
            continue
        pred_data = predictions[task]
        pred = pred_data['pred']
        target = pred_data['target']
        mask = pred_data.get('mask')
        if task == 'mortality':
            auc = compute_auc(target.flatten(), pred.flatten(), mask.flatten() if mask is not None else None)
            metrics[task] = {'auc': auc}
        elif task == 'decomp':
            auc = compute_auc(target.flatten(), pred.flatten())
            metrics[task] = {'auc': auc}
        elif task == 'los':
            if pred.ndim == 2:
                pred_bucket = pred.argmax(axis=-1)
            else:
                pred_bucket = pred
            target_bucket = target.astype(int)
            valid_mask = mask.flatten() if mask is not None else None
            kappa = compute_kappa(target_bucket.flatten(), pred_bucket.flatten(), valid_mask)
            metrics[task] = {'kappa': kappa}
        elif task == 'phenotype':
            auc = compute_macro_auc(target, pred, mask)
            metrics[task] = {'auc': auc}
        elif task == 'seg':
            metrics[task] = compute_segmentation_metrics(pred, target, mask)
        elif task == 'depth':
            metrics[task] = compute_depth_metrics(pred, target, mask)
        elif task == 'normal':
            metrics[task] = compute_normal_metrics(pred, target, mask)
    return metrics
