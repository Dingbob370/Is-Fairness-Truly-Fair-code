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
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import warnings
import re
import hashlib
GCS_COMPONENT_COLS: List[str] = ['Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale verbal response']
MIMIC3_FEATURE_NAMES_17: List[str] = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
MIMIC3_FEATURE_NAMES_14: List[str] = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale total', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
DEFAULT_MIMIC3_FEATURE_NAMES: List[str] = list(MIMIC3_FEATURE_NAMES_17)

class MIMIC3MTLDataset(Dataset):

    def __init__(self, root: str, split: str, tasks_enabled: List[str], feature_dim: int=17, normalization: str='train_stats', stats_dir: Optional[str]=None, fillna_method: str='ffill_zero', val_ratio: float=0.15, seed: int=42, time_row_policy: str='legacy_last', time_row_policy_test: str='hash_random', verbose: bool=False, skip_stats: bool=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.tasks_enabled = tasks_enabled
        self.feature_dim = int(feature_dim)
        if self.feature_dim == 17:
            self.feature_names = list(MIMIC3_FEATURE_NAMES_17)
        elif self.feature_dim == 14:
            self.feature_names = list(MIMIC3_FEATURE_NAMES_14)
        else:
            raise ValueError(f'Unsupported MIMIC3 feature_dim={feature_dim}. Expected 17 (full) or 14 (ablation).')
        self.normalization = normalization
        self.fillna_method = fillna_method
        self.val_ratio = val_ratio
        self.seed = seed
        self.time_row_policy = str(time_row_policy)
        self.time_row_policy_test = str(time_row_policy_test)
        valid_policies = {'legacy_last', 'hash_random', 'min_period', 'max_period'}
        if self.time_row_policy not in valid_policies:
            raise ValueError(f'Unknown time_row_policy={self.time_row_policy}, must be one of {sorted(valid_policies)}')
        if self.time_row_policy_test not in valid_policies:
            raise ValueError(f'Unknown time_row_policy_test={self.time_row_policy_test}, must be one of {sorted(valid_policies)}')
        self.verbose = verbose
        self.skip_stats = skip_stats
        self._schema_monitor: Dict[str, Any] = {'files_seen': 0, 'missing_cols_files': 0, 'extra_cols_files': 0, 'gcs_text_files': 0, 'examples': {'missing': [], 'extra': [], 'gcs_text': []}}
        self._schema_monitor_max_examples: int = 20
        self._schema_monitor_printed: int = 0
        valid_tasks = ['mortality', 'decomp', 'los', 'phenotype']
        for task in tasks_enabled:
            if task not in valid_tasks:
                raise ValueError(f'Unknown task: {task}. Valid tasks: {valid_tasks}')
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'Bench directory not found: {self.root}')
        if stats_dir is None:
            stats_dir = os.path.join(self.root, 'stats')
        self.stats_dir = stats_dir
        os.makedirs(stats_dir, exist_ok=True)
        self.episodes = self._build_episode_list()
        if self.skip_stats:
            self.stats = {'mean': 0.0, 'std': 1.0, 'feature_names': list(self.feature_names)}
            if self.verbose:
                print('Using unit feature normalization because skip_stats=True.')
        else:
            self.stats = self._align_stats_schema(self._load_or_compute_stats())
        self.metadata = self._load_metadata()
        if self.verbose:
            print(f'MIMIC3MTLDataset: {len(self.episodes)} episodes in {split} split')
            print(f'Tasks enabled: {tasks_enabled}')
            if hasattr(self, 'stats') and 'feature_names' in self.stats:
                print(f"Feature stats cover {len(self.stats['feature_names'])} features.")

    def _build_episode_list(self) -> List[Dict]:
        episodes_dict = {}
        if self.verbose:
            print(f'Building MIMIC-III episode list from root={self.root}')
            print(f'Enabled tasks: {self.tasks_enabled}')
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'Missing required file or directory: {self.root}')
        base_split = self.split
        derive_val_from_train = False
        if self.split in ('train', 'val'):
            val_listfiles = []
            for task in self.tasks_enabled:
                task_dir = os.path.join(self.root, task)
                if not os.path.exists(task_dir):
                    continue
                val_listfiles.append(os.path.join(task_dir, 'val_listfile.csv'))
            has_any_val = any((os.path.exists(p) for p in val_listfiles))
            has_all_val = bool(val_listfiles) and all((os.path.exists(p) for p in val_listfiles))
            if has_any_val and (not has_all_val):
                raise RuntimeError('Inconsistent val_listfile.csv presence across tasks. Either provide val_listfile.csv for all enabled tasks or none.')
            if self.split == 'val':
                if has_all_val:
                    base_split = 'val'
                else:
                    base_split = 'train'
                    derive_val_from_train = True
            else:
                base_split = 'train'
                derive_val_from_train = not has_any_val
        self._base_split = base_split
        self._derive_val_from_train = derive_val_from_train
        if self.verbose and self.split == 'val' and (base_split == 'train'):
            print('val split uses train_listfile.csv and will be carved from train by stay_id to avoid leakage.')
        for task in self.tasks_enabled:
            task_dir = os.path.join(self.root, task)
            if self.verbose:
                print(f'Loading task={task} from {task_dir}')
            if not os.path.exists(task_dir):
                print(f'Skipping missing task directory: {task_dir}')
                continue
            actual_split = base_split
            if self.verbose:
                print(f'Using listfile split={actual_split} for task={task}')
            listfile_path = os.path.join(task_dir, f'{actual_split}_listfile.csv')
            if self.verbose:
                print(f'Listfile candidate: {listfile_path}')
            if not os.path.exists(listfile_path):
                print(f'Missing required file or directory: {listfile_path}')
                continue
            try:
                with open(listfile_path, 'r') as f:
                    lines = f.readlines()
                if self.verbose:
                    print(f'Loaded {len(lines)} lines from {listfile_path}')
                line_count = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    self._parse_listfile_line(line, episodes_dict, task, actual_split)
                    line_count += 1
                    if self.verbose and line_count <= 5:
                        print(f'Listfile preview line {line_count}: parsed.')
                if self.verbose:
                    print(f'Parsed task={task}: lines={line_count}, accumulated episodes={len(episodes_dict)}')
            except Exception as e:
                print(f'Failed to parse listfile {listfile_path}: {e}')
                import traceback
                traceback.print_exc()
        for ep in episodes_dict.values():
            for info in ep.get('tasks', {}).values():
                if isinstance(info, dict) and '_row_score' in info:
                    info.pop('_row_score', None)

        def _sort_key(ep: Dict):
            stay_id_raw = str(ep.get('stay_id', ''))
            try:
                stay_id = int(stay_id_raw)
            except Exception:
                stay_id = stay_id_raw
            episode_file = str(ep.get('episode_file', ''))
            m = re.search('_episode(\\\\d+)', episode_file)
            episode_id = int(m.group(1)) if m else 0
            return (stay_id, episode_id, episode_file)
        episodes = sorted(list(episodes_dict.values()), key=_sort_key)
        if self.verbose:
            print('Episode list summary:')
            print(f'Episodes parsed: {len(episodes)}')
            if episodes:
                print('First parsed episodes:')
                for (i, ep) in enumerate(episodes[:3]):
                    print(f"  {i}: tasks={list(ep['tasks'].keys())}")
        if episodes and self.split in ('train', 'val') and getattr(self, '_derive_val_from_train', False):
            stay_ids = []
            seen = set()
            for ep in episodes:
                sid = ep.get('stay_id')
                if sid not in seen:
                    seen.add(sid)
                    stay_ids.append(sid)
            if len(stay_ids) > 1 and float(self.val_ratio) > 0:
                rng = np.random.RandomState(self.seed)
                val_stay_count = max(1, int(len(stay_ids) * self.val_ratio))
                val_stay_count = min(val_stay_count, len(stay_ids) - 1)
                val_indices = rng.choice(len(stay_ids), size=val_stay_count, replace=False)
                val_stays = set((stay_ids[i] for i in val_indices))
                self._val_stay_ids = val_stays
                if self.split == 'val':
                    episodes = [ep for ep in episodes if ep.get('stay_id') in val_stays]
                if self.verbose:
                    print(f'Derived train/val split (by stay_id): total_stays={len(stay_ids)}, val_stays={len(val_stays)}, split={self.split}, episodes={len(episodes)}')
            elif self.verbose and self.split == 'val':
                print('WARNING: not enough unique stay_id to create a val split; using all episodes as val.')
        if False and self.split == 'val' and episodes:
            np.random.seed(self.seed)
            val_size = int(len(episodes) * self.val_ratio)
            indices = np.random.choice(len(episodes), val_size, replace=False)
            episodes = [episodes[i] for i in sorted(indices)]
            if self.verbose:
                print(f'Derived validation subset: source_episodes={len(episodes_dict)}, val_size={val_size}')
        return episodes

    def _add_episodes_from_directory(self, episodes_dict: Dict, task: str, split: str):
        task_dir = os.path.join(self.root, task)
        split_dir = os.path.join(task_dir, split)
        if not os.path.exists(split_dir):
            return
        for filename in os.listdir(split_dir):
            if filename.endswith('.csv') and '_timeseries' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    stay_id = parts[0]
                    episode_key = (stay_id, filename)
                    if episode_key not in episodes_dict:
                        episodes_dict[episode_key] = {'stay_id': stay_id, 'episode_file': filename, 'timeseries_path': os.path.join(split_dir, filename), 'tasks': {}}
                    episodes_dict[episode_key]['tasks'][task] = {'has_label': False, 'labels': []}

    def _parse_listfile_line(self, line: str, episodes_dict: Dict, task: str, split: str):
        line = line.strip()
        if 'stay' in line.lower() or 'tay' in line.lower():
            if self.verbose and task == 'phenotype':
                print(f'Skipping phenotype header line: {line[:100]}...')
            return
        if ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        if len(parts) < 2:
            if self.verbose:
                print(f'Skipping malformed listfile line: {line[:50]}...')
            return
        period_length: Optional[float] = None

        def _looks_like_timeseries_file(value: str) -> bool:
            v = str(value).strip().lower()
            return v.endswith('.csv') or '_timeseries' in v

        def _is_number(value: str) -> bool:
            try:
                float(str(value).strip())
                return True
            except Exception:
                return False

        def _resolve_timeseries_path(task_name: str, split_name: str, episode_file_name: str) -> str:
            ep = str(episode_file_name).strip()
            ep_base = os.path.basename(ep)
            candidates = []
            if '/' in ep or '\\' in ep:
                candidates.append(os.path.join(self.root, ep))
            candidates.extend([os.path.join(self.root, split_name, ep_base), os.path.join(self.root, task_name, split_name, ep_base), os.path.join(self.root, task_name, ep_base), os.path.join(self.root, ep_base)])
            for p in candidates:
                if os.path.exists(p):
                    return p
            return candidates[0]
        if task == 'phenotype':
            filename = parts[0].strip()
            if '_episode' in filename:
                stay_id = filename.split('_')[0]
            else:
                stay_id = filename
            episode_file = os.path.basename(filename)
            label_start = 1
            if len(parts) >= 27 and _is_number(parts[1]):
                try:
                    period_length = float(parts[1])
                    label_start = 2
                except Exception:
                    period_length = None
            labels = [p.strip() for p in parts[label_start:] if str(p).strip() != '']
            timeseries_path = _resolve_timeseries_path(task, split, episode_file)
        else:
            fields = [p.strip() for p in parts]
            file_idx = None
            if _looks_like_timeseries_file(fields[0]):
                file_idx = 0
            else:
                for i in range(1, len(fields)):
                    if _looks_like_timeseries_file(fields[i]):
                        file_idx = i
                        break
            if file_idx is not None:
                episode_file = os.path.basename(fields[file_idx])
                stay_id = episode_file.split('_')[0] if '_episode' in episode_file else fields[0]
                if file_idx == 0 and len(fields) >= 3 and _is_number(fields[1]) and _is_number(fields[2]):
                    period_length = float(fields[1])
                    labels = [fields[2]]
                else:
                    if len(fields) >= 3 and _is_number(fields[1]):
                        try:
                            period_length = float(fields[1])
                        except Exception:
                            period_length = None
                    labels = [p for p in fields[file_idx + 1:] if str(p).strip() != '']
            else:
                stay_id = fields[0]
                episode_file = f'{stay_id}_episode1_timeseries.csv'
                labels = [fields[1]] if len(fields) == 2 else [fields[-1]]
                if len(fields) >= 3 and _is_number(fields[1]):
                    try:
                        period_length = float(fields[1])
                    except Exception:
                        period_length = None
            timeseries_path = _resolve_timeseries_path(task, split, episode_file)
        if not os.path.exists(timeseries_path):
            if self.verbose:
                print(f'Missing required file or directory: {timeseries_path}')
            return
        episode_key = (stay_id, episode_file)
        if episode_key not in episodes_dict:
            episodes_dict[episode_key] = {'stay_id': stay_id, 'episode_file': episode_file, 'timeseries_path': timeseries_path, 'tasks': {}}
        new_task_info = {'has_label': len(labels) > 0, 'labels': labels, 'period_length': period_length}
        existing = episodes_dict[episode_key]['tasks'].get(task)
        policy = self.time_row_policy_test if split == 'test' else self.time_row_policy
        if task in ('los', 'decomp') and period_length is not None and (policy != 'legacy_last'):
            if policy == 'hash_random':
                seed = int(getattr(self, 'seed', 42))
                cand = f'{seed}|{stay_id}|{episode_file}|{task}|{float(period_length):.6f}'
                score = int(hashlib.md5(cand.encode('utf-8')).hexdigest(), 16)
                new_task_info['_row_score'] = score
                prev_score = None if existing is None else existing.get('_row_score')
                if prev_score is None or score < int(prev_score):
                    episodes_dict[episode_key]['tasks'][task] = new_task_info
            else:
                try:
                    new_pl = float(period_length)
                except Exception:
                    new_pl = None
                old_pl = None if existing is None else existing.get('period_length')
                try:
                    old_pl_v = float(old_pl) if old_pl is not None else None
                except Exception:
                    old_pl_v = None
                take = False
                if old_pl_v is None or new_pl is None:
                    take = True if old_pl_v is None and new_pl is not None else False
                elif policy == 'min_period':
                    take = new_pl < old_pl_v
                elif policy == 'max_period':
                    take = new_pl > old_pl_v
                if take:
                    episodes_dict[episode_key]['tasks'][task] = new_task_info
        else:
            episodes_dict[episode_key]['tasks'][task] = new_task_info
        if self.verbose and len(episodes_dict) < 5:
            print(f'Parsed label row for task={task}, labels={len(labels)}')

    def _load_metadata(self) -> Dict:
        metadata = {'phenotype_dim': 25}
        labels_file = os.path.join(self.root, 'phenotype_labels.csv')
        if os.path.exists(labels_file):
            try:
                df = pd.read_csv(labels_file, nrows=1)
                cols = list(df.columns)
                if cols:
                    first = str(cols[0]).strip().lower()
                    has_id_col = first in {'stay', 'stay_id', 'icustay_id', 'subject_id', 'hadm_id'} or first.startswith('stay')
                    metadata['phenotype_dim'] = max(1, len(cols) - 1) if has_id_col else len(cols)
                if self.verbose:
                    print(f"Detected phenotype_dim={metadata['phenotype_dim']} from phenotype_labels.csv")
            except:
                pass
        return metadata

    def _load_or_compute_stats(self) -> Dict:
        stats_file = os.path.join(self.stats_dir, 'stats.json')
        if self.split == 'train':
            stats = self._compute_train_stats()
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            if self.verbose:
                print(f'Wrote train feature stats to {stats_file}')
        elif os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            if self.verbose:
                print(f'Loaded feature stats from {stats_file}')
        else:
            warnings.warn(f'Feature stats file not found: {stats_file}; using unit normalization.')
            stats = {'mean': 0.0, 'std': 1.0, 'feature_names': list(self.feature_names)}
        return stats

    def _align_stats_schema(self, stats: Dict) -> Dict:
        """
        Ensure stats schema matches the fixed feature contract used by this project.
        This avoids feature_dim drift caused by pandas dtype inference (e.g., columns as object vs float).
        """
        desired = list(self.feature_names)
        feature_names = stats.get('feature_names') or []
        mean_raw = stats.get('mean', 0.0)
        std_raw = stats.get('std', 1.0)
        mean = np.asarray(mean_raw, dtype=np.float32)
        std = np.asarray(std_raw, dtype=np.float32)
        if not isinstance(feature_names, list) or len(feature_names) == 0 or mean.ndim == 0 or (std.ndim == 0):
            return {'mean': 0.0, 'std': 1.0, 'feature_names': desired}
        name_to_idx = {str(n): i for (i, n) in enumerate(feature_names)}
        new_mean: List[float] = []
        new_std: List[float] = []
        for name in desired:
            idx = name_to_idx.get(name)
            if idx is None or idx >= mean.shape[0] or idx >= std.shape[0]:
                new_mean.append(0.0)
                new_std.append(1.0)
            else:
                new_mean.append(float(mean[idx]))
                v = float(std[idx])
                new_std.append(v if v != 0.0 else 1.0)
        return {'mean': new_mean, 'std': new_std, 'feature_names': desired}

    def _compute_train_stats(self) -> Dict:
        if self.verbose:
            print('Computing train feature statistics.')
        feature_names = list(self.feature_names)
        if len(self.episodes) == 0:
            return {'mean': 0.0, 'std': 1.0, 'feature_names': feature_names}
        all_features: List[np.ndarray] = []
        max_samples = min(1000, len(self.episodes))
        indices = np.random.choice(len(self.episodes), max_samples, replace=False)
        for idx in indices:
            ep = self.episodes[idx]
            try:
                ts_data = pd.read_csv(ep['timeseries_path'])
                df = ts_data.reindex(columns=feature_names).copy()
                for c in GCS_COMPONENT_COLS:
                    if c in df.columns:
                        df[c] = self._coerce_gcs_component_column(df[c])
                df = df.apply(pd.to_numeric, errors='coerce')
                features = df.to_numpy(dtype=np.float32)
                features = np.where(np.isfinite(features), features, np.nan)
                features = self._handle_nan(features)
                all_features.append(features)
            except Exception as e:
                if self.verbose:
                    print(f'Failed to read one timeseries file while computing stats: {e}')
        if len(all_features) == 0:
            return {'mean': 0.0, 'std': 1.0, 'feature_names': feature_names}
        try:
            all_features_array = np.vstack(all_features)
        except ValueError as e:
            if self.verbose:
                print(f'Failed to stack feature arrays: {e}')
            return {'mean': 0.0, 'std': 1.0, 'feature_names': feature_names}
        mean = np.nanmean(all_features_array, axis=0)
        std = np.nanstd(all_features_array, axis=0)
        std = np.where(std == 0.0, 1.0, std)
        if self.verbose:
            print(f'Computed train stats: mean shape={mean.shape}, std shape={std.shape}')
        return {'mean': mean.tolist(), 'std': std.tolist(), 'feature_names': feature_names}
        all_feature_names_list = []
        max_samples_for_names = min(100, len(self.episodes))
        if max_samples_for_names == 0:
            return {'mean': 0.0, 'std': 1.0, 'feature_names': []}
        indices = np.random.choice(len(self.episodes), max_samples_for_names, replace=False)
        for idx in indices:
            ep = self.episodes[idx]
            try:
                ts_data = pd.read_csv(ep['timeseries_path'])
                numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
                exclude_cols = ['Hours', 'time', 'Unnamed: 0', 'hours']
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                if len(numeric_cols) > 0:
                    all_feature_names_list.append(set(numeric_cols))
            except Exception as e:
                if self.verbose:
                    print(f'Failed to inspect one timeseries file: {e}')
        if len(all_feature_names_list) == 0:
            return {'mean': 0.0, 'std': 1.0, 'feature_names': []}
        common_features = set.intersection(*all_feature_names_list)
        common_features = sorted(list(common_features))
        if len(common_features) == 0:
            first_ep = self.episodes[0]
            try:
                ts_data = pd.read_csv(first_ep['timeseries_path'])
                numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
                exclude_cols = ['Hours', 'time', 'Unnamed: 0', 'hours']
                common_features = [col for col in numeric_cols if col not in exclude_cols]
                if self.verbose:
                    print(f'Fallback numeric feature count: {len(common_features)}')
            except:
                common_features = []
        if self.verbose:
            print(f'Common numeric feature count: {len(common_features)}')
            if len(common_features) <= 10:
                print(f'Common numeric features: {common_features}')
        all_features = []
        max_samples = min(1000, len(self.episodes))
        indices = np.random.choice(len(self.episodes), max_samples, replace=False)
        for idx in indices:
            ep = self.episodes[idx]
            try:
                ts_data = pd.read_csv(ep['timeseries_path'])
                available_features = [col for col in common_features if col in ts_data.columns]
                if len(available_features) > 0:
                    features = ts_data[available_features].values
                    features = self._handle_nan(features)
                    all_features.append(features)
            except Exception as e:
                if self.verbose:
                    print(f'Failed to read one timeseries file while computing stats: {e}')
        if len(all_features) == 0:
            return {'mean': 0.0, 'std': 1.0, 'feature_names': common_features}
        try:
            all_features_array = np.vstack(all_features)
        except ValueError as e:
            if self.verbose:
                print(f'Failed to stack feature arrays: {e}')
                for (i, feat) in enumerate(all_features):
                    print(f'Feature array {i}: shape={feat.shape}')
            return {'mean': 0.0, 'std': 1.0, 'feature_names': common_features}
        mean = np.nanmean(all_features_array, axis=0)
        std = np.nanstd(all_features_array, axis=0)
        std[std == 0] = 1.0
        if self.verbose:
            print(f'Computed train stats: mean shape={mean.shape}, std shape={std.shape}')
        return {'mean': mean.tolist(), 'std': std.tolist(), 'feature_names': common_features}

    def _handle_nan(self, data: np.ndarray) -> np.ndarray:
        if self.fillna_method == 'ffill_zero':
            df = pd.DataFrame(data)
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            return df.values
        elif self.fillna_method == 'zero':
            data = np.nan_to_num(data, nan=0.0)
            return data
        else:
            raise ValueError(f'Unknown fillna_method: {self.fillna_method}')

    def __len__(self) -> int:
        return len(self.episodes)

    def get_schema_monitor(self) -> Dict[str, Any]:
        """Return best-effort schema drift statistics collected during _load_timeseries()."""
        return self._schema_monitor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]
        (x, x_mask) = self._load_timeseries(ep['timeseries_path'])
        seq_len = x.shape[0]
        period_length = ep.get('period_length')
        y = {}
        y_mask = {}
        for task in self.tasks_enabled:
            if task in ep['tasks']:
                (task_y, task_mask) = self._load_task_label(task, ep, seq_len)
            else:
                (task_y, task_mask) = self._create_empty_label(task, seq_len)
            y[task] = task_y
            y_mask[task] = task_mask
        assert torch.isfinite(x).all(), 'NaN/Inf found in one clinical sample.'
        meta = {'idx': idx, 'split': self.split, 'seq_len': seq_len, 'period_length': period_length}
        return {'x': x, 'x_mask': x_mask, 'y': y, 'y_mask': y_mask, 'meta': meta}

    @staticmethod
    def _looks_like_label_column(name: str) -> bool:
        """
        Best-effort guard against label leakage: timeseries should not contain outcome/label columns.
        """
        raw = str(name).strip().lower()
        tokens = [t for t in re.split('[^a-z0-9]+', raw) if t]
        joined = ''.join(tokens)
        strong = ('mortality', 'hospitalexpire', 'icuexpire', 'lengthofstay', 'outcome', 'ytrue')
        if any((k in joined for k in strong)):
            return True
        if any((t in {'death', 'expire', 'label', 'target'} for t in tokens)):
            return True
        if 'los' in tokens:
            return True
        return False

    @staticmethod
    def _coerce_gcs_component_column(series: 'pd.Series') -> 'pd.Series':
        try:
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors='coerce')
        except Exception:
            pass
        s = series.astype(str)
        extracted = s.str.extract('(\\d+(?:\\.\\d+)?)', expand=False)
        return pd.to_numeric(extracted, errors='coerce')

    def _load_timeseries(self, path: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(path, dict):
            episode = path
            path = episode.get('timeseries_path') or episode.get('path')
            if path is None:
                raise TypeError(f"_load_timeseries() got a dict but cannot find 'timeseries_path'. Available keys: {list(episode.keys())}")
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"_load_timeseries() expects a path-like object or an episode dict with 'timeseries_path'. Got: {type(path)}")
        path = os.fspath(path)
        try:
            ts_data = pd.read_csv(path)
        except Exception as e:
            raise IOError(f'Failed to load one timeseries file: {e}')
        leak_cols = [c for c in ts_data.columns if self._looks_like_label_column(c)]
        if leak_cols:
            shown = ', '.join(map(str, leak_cols[:20]))
            more = '' if len(leak_cols) <= 20 else f'...(+{len(leak_cols) - 20})'
            raise RuntimeError(f'Potential label leakage columns in timeseries: [{shown}{more}]')
        if hasattr(self, 'stats') and 'feature_names' in self.stats and (len(self.stats['feature_names']) > 0):
            feature_names = self.stats['feature_names']
            available_features = []
            for feat in feature_names:
                if feat in ts_data.columns:
                    available_features.append(feat)
                elif self.verbose:
                    print(f'Missing feature column {feat} in one timeseries file')
            if len(available_features) == 0:
                numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
                exclude_cols = ['Hours', 'time', 'Unnamed: 0', 'hours']
                available_features = [col for col in numeric_cols if col not in exclude_cols]
        else:
            numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Hours', 'time', 'Unnamed: 0', 'hours']
            available_features = [col for col in numeric_cols if col not in exclude_cols]
        if len(available_features) == 0:
            available_features = list(ts_data.columns)
            exclude_cols = ['Hours', 'time', 'Unnamed: 0', 'hours']
            available_features = [col for col in available_features if col not in exclude_cols]
        feature_names = list(self.stats.get('feature_names') or self.feature_names)
        try:
            self._schema_monitor['files_seen'] += 1
            exclude_cols = {'Hours', 'time', 'hours', 'Unnamed: 0'}
            missing_cols = [c for c in feature_names if c not in ts_data.columns]
            extra_cols = [c for c in ts_data.columns if c not in feature_names and str(c) not in exclude_cols]
            gcs_text_cols = [c for c in GCS_COMPONENT_COLS if c in ts_data.columns and (not pd.api.types.is_numeric_dtype(ts_data[c]))]
            if missing_cols:
                self._schema_monitor['missing_cols_files'] += 1
                if len(self._schema_monitor['examples']['missing']) < self._schema_monitor_max_examples:
                    self._schema_monitor['examples']['missing'].append({'missing': missing_cols})
            if extra_cols:
                self._schema_monitor['extra_cols_files'] += 1
                if len(self._schema_monitor['examples']['extra']) < self._schema_monitor_max_examples:
                    self._schema_monitor['examples']['extra'].append({'extra': list(map(str, extra_cols[:20]))})
            if gcs_text_cols:
                self._schema_monitor['gcs_text_files'] += 1
                if len(self._schema_monitor['examples']['gcs_text']) < self._schema_monitor_max_examples:
                    self._schema_monitor['examples']['gcs_text'].append({'cols': list(gcs_text_cols)})
            if self.verbose and self._schema_monitor_printed < 10 and (missing_cols or extra_cols or gcs_text_cols):
                self._schema_monitor_printed += 1
                print(f'[schema-monitor] missing={len(missing_cols)} extra={len(extra_cols)} gcs_text={gcs_text_cols}')
        except Exception:
            pass
        df = ts_data.reindex(columns=feature_names).copy()
        for c in GCS_COMPONENT_COLS:
            if c in df.columns:
                df[c] = self._coerce_gcs_component_column(df[c])
        df = df.apply(pd.to_numeric, errors='coerce')
        features = df.to_numpy(dtype=np.float32)
        features = np.where(np.isfinite(features), features, np.nan)
        x_mask = np.isfinite(features)
        seq_len = features.shape[0]
        features = self._handle_nan(features)
        if self.normalization == 'train_stats' and hasattr(self, 'stats') and ('mean' in self.stats) and ('std' in self.stats):
            mean = np.asarray(self.stats['mean'], dtype=np.float32)
            std = np.asarray(self.stats['std'], dtype=np.float32)
            if mean.ndim == 0 or std.ndim == 0:
                mean_scalar = float(mean)
                std_scalar = float(std)
                if std_scalar == 0.0:
                    std_scalar = 1.0
                features = (features - mean_scalar) / (std_scalar + 1e-08)
            elif mean.shape[0] == features.shape[1] and std.shape[0] == features.shape[1]:
                std = np.where(std == 0.0, 1.0, std)
                features = (features - mean) / (std + 1e-08)
            elif self.verbose:
                warnings.warn(f'Stats shape mismatch: mean={mean.shape}, std={std.shape}, feature_dim={features.shape[1]}')
        x = torch.FloatTensor(features)
        x_mask = torch.BoolTensor(x_mask)
        return (x, x_mask)

    def _load_task_label(self, task: str, episode: Dict, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task_info = episode['tasks'][task]
        if task == 'mortality':
            if task_info['has_label'] and len(task_info['labels']) > 0:
                try:
                    label = float(task_info['labels'][0])
                except:
                    label = 0.0
                mask = True
            else:
                label = 0.0
                mask = False
            return (torch.FloatTensor([label]), torch.BoolTensor([mask]))
        elif task == 'decomp':
            y = torch.zeros(seq_len, dtype=torch.float32)
            mask = torch.zeros(seq_len, dtype=torch.bool)
            if task_info['has_label'] and len(task_info['labels']) > 0 and (seq_len > 0):
                try:
                    label = float(task_info['labels'][-1])
                except Exception:
                    label = 0.0
                y[seq_len - 1] = label
                mask[seq_len - 1] = True
            return (y, mask)
        elif task == 'los':
            if task_info['has_label'] and len(task_info['labels']) > 0:
                try:
                    remaining_hours = float(task_info['labels'][0])
                except Exception:
                    remaining_hours = 0.0
                remaining_days = remaining_hours / 24.0
                mask = True
            else:
                remaining_days = 0.0
                mask = False
            return (torch.FloatTensor([remaining_days]), torch.BoolTensor([mask]))
        elif task == 'phenotype':
            if task_info['has_label'] and len(task_info['labels']) > 0:
                try:
                    labels = [float(l) for l in task_info['labels']]
                except:
                    labels = []
                if len(labels) < self.metadata['phenotype_dim']:
                    labels = labels + [0.0] * (self.metadata['phenotype_dim'] - len(labels))
                elif len(labels) > self.metadata['phenotype_dim']:
                    labels = labels[:self.metadata['phenotype_dim']]
                mask = torch.ones(self.metadata['phenotype_dim'], dtype=torch.bool)
            else:
                labels = torch.zeros(self.metadata['phenotype_dim'])
                mask = torch.zeros(self.metadata['phenotype_dim'], dtype=torch.bool)
            return (torch.FloatTensor(labels), mask)
        else:
            raise ValueError(f'Unknown task: {task}')

    def _create_empty_label(self, task: str, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if task == 'mortality':
            return (torch.FloatTensor([0.0]), torch.BoolTensor([False]))
        elif task == 'decomp':
            return (torch.zeros(seq_len), torch.zeros(seq_len, dtype=torch.bool))
        elif task == 'los':
            return (torch.FloatTensor([0.0]), torch.BoolTensor([False]))
        elif task == 'phenotype':
            return (torch.zeros(self.metadata['phenotype_dim']), torch.zeros(self.metadata['phenotype_dim'], dtype=torch.bool))
        else:
            raise ValueError(f'Unknown task: {task}')
