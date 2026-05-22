# Is Fairness Truly Fair? Code

This repository contains the reference implementation for the paper's ReLiF experiments: fixed-threshold fairness auditing and reliability-aware Lipschitz fairness regularization for multi-task learning.

Included experiment settings:

- MIMIC-III multi-task ICU benchmark
- eICU mortality and length-of-stay setting
- NYUv2 dense prediction setting
- ERM, Uncertainty, GradNorm, PCGrad, FairGrad, CATS, and ReLiF variants

No datasets, checkpoints, logs, or generated experiment outputs are included.

## What Is Included

- Core training entrypoints for ReLiF and baseline methods.
- Dataset loaders for MIMIC-III, eICU, and NYUv2.
- Model, metric, controller, and Lipschitz regularization code.
- Evaluation scripts for utility and fixed-threshold bias auditing.
- NYUv2 proxy-calibration code.
- A dependency-light smoke check for repository integrity.

## What Is Not Included

- Raw or processed clinical datasets.
- NYUv2 `.npy` arrays.
- Checkpoints, logs, generated outputs, or result workbooks.
- Paper-internal audit spreadsheets or local experiment logs.
- Scripts that construct benchmark tensors from restricted clinical database tables.

## Data

Split convention:

- MIMIC-III and eICU use the standard train/validation/test workflow; final clinical evaluation examples below use `--split test`. For MIMIC-III, if all enabled tasks provide `val_listfile.csv`, the loader uses those validation listfiles; otherwise it derives a stay-level validation split from train listfiles using `--data_seed`.
- The released NYUv2 preprocessing layout provides `train` and `val` only. NYUv2 is used as a non-medical vision dense-prediction cross-domain validation setting, and should not be described as a NYUv2 test-set result.

### MIMIC-III

MIMIC-III is a restricted clinical dataset. Obtain access through the official [MIMIC-III Clinical Database v1.4 PhysioNet page](https://physionet.org/content/mimiciii/1.4/), complete the required credentialing, and follow the [PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). The code expects a benchmark-style directory with task subdirectories and listfiles.

Example root:

```bash
./data/mimic3/bench
```

### eICU

eICU is also a restricted clinical dataset. Obtain access through the official [eICU Collaborative Research Database v2.0 PhysioNet page](https://physionet.org/content/eicu-crd/2.0/) and follow the [PhysioNet Credentialed Health Data License](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). This repository does not redistribute restricted eICU materials or scripts that construct benchmark tensors from database tables. The dataloader expects a compact benchmark directory produced in an institution-approved environment.

Expected compact files:

```text
data/eicu_bench/
  x_raw.npy
  x_mask.npy
  seq_len.npy
  stay_id.npy
  y_mortality.npy
  y_mask_mortality.npy
  y_los.npy
  y_mask_los.npy
  split_idx.npz
  stats/stats.json
```

### NYUv2

NYUv2 is publicly available from the original NYU Depth V2 project page and common benchmark mirrors, but this code expects a preprocessed multi-task `.npy` layout:

```text
data/nyuv2/
  train/image/{id}.npy
  train/label/{id}.npy
  train/depth/{id}.npy
  train/normal/{id}.npy
  val/image/{id}.npy
  val/label/{id}.npy
  val/depth/{id}.npy
  val/normal/{id}.npy
```

The released NYUv2 loader supports `train` and `val` splits.
Use `val` for NYUv2 utility evaluation, proxy calibration, and fixed-threshold auditing.
File names must be integer sample ids such as `1.npy`, `2.npy`, and so on; the same ids must exist across `image`, `label`, `depth`, and `normal`.
Images are expected as float arrays in channel-first or channel-last format accepted by the loader; labels, depth, and normals should match the spatial resolution used for training.
This repository does not include a NYUv2 conversion script because public NYUv2 releases differ in packaging. Record the source release, split construction, preprocessing resolution, and any checksums in your local experiment log before reporting numbers.

## Repository Structure

```text
.
├── baselines/                  # ERM, task-balancing, gradient, and fairness baselines
├── data_loader/                # Dataset builders and dataset-specific loaders
├── models/                     # Backbone, task heads, controller, and ReLiF model
├── scripts/smoke_check.py      # Dependency-light repository sanity check
├── main.py                     # ReLiF training entrypoint
├── run_baseline.py             # Baseline and ablation training entrypoint
├── eval.py                     # Utility evaluation
├── eval_ckpt.py                # Fixed-threshold bias audit
├── eval_grad_norm.py           # Gradient diagnostic utility
├── compute_nyuv2_proxy_stats.py
├── config.py
└── requirements.txt
```

## Entry Points

| Purpose | Script |
| --- | --- |
| ReLiF training | `main.py` |
| Baseline and ablation training | `run_baseline.py` |
| Utility evaluation | `eval.py` |
| Fixed-threshold bias audit | `eval_ckpt.py` |
| Gradient diagnostics | `eval_grad_norm.py` |
| NYUv2 proxy calibration | `compute_nyuv2_proxy_stats.py` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For GPU runs, install the PyTorch build matching your CUDA environment.

## Training Examples

MIMIC-III ReLiF:

```bash
PYTHONPATH=. python3 main.py \
  --dataset_name mimic3 \
  --data_root ./data/mimic3/bench \
  --seed 42 \
  --data_seed 42 \
  --amp_dtype bf16
```

eICU ReLiF:

```bash
PYTHONPATH=. python3 main.py \
  --dataset_name eicu \
  --data_root ./data/eicu_bench \
  --tasks mortality,los \
  --seed 42 \
  --data_seed 42 \
  --amp_dtype bf16
```

NYUv2 ERM baseline:

```bash
PYTHONPATH=. python3 run_baseline.py \
  --method erm \
  --dataset_name nyuv2 \
  --data_root ./data/nyuv2 \
  --seed 42 \
  --data_seed 42
```

Generate the fixed NYUv2 proxy calibration from an ERM checkpoint:

```bash
PYTHONPATH=. python3 compute_nyuv2_proxy_stats.py \
  --ckpt ./outputs/nyuv2_erm_seed42/best_model.pt \
  --data_root ./data/nyuv2 \
  --split val \
  --output ./outputs/nyuv2_proxy_stats.json
```

NYUv2 ReLiF training entrypoint:

```bash
PYTHONPATH=. python3 main.py \
  --dataset_name nyuv2 \
  --data_root ./data/nyuv2 \
  --nyuv2_proxy_stats ./outputs/nyuv2_proxy_stats.json \
  --seed 42 \
  --data_seed 42
```

## Evaluation

Clinical utility evaluation:

```bash
PYTHONPATH=. python3 eval.py \
  --ckpt /path/to/run/best_model.pt \
  --dataset_name mimic3 \
  --data_root ./data/mimic3/bench \
  --split test
```

Clinical fixed-threshold bias audit:

```bash
PYTHONPATH=. python3 eval_ckpt.py \
  --runs /path/to/run \
  --dataset_name mimic3 \
  --data_root ./data/mimic3/bench \
  --split test \
  --fixed_threshold 0.275
```

For NYUv2, use `--split val` and pass the shared proxy calibration file:

```bash
PYTHONPATH=. python3 eval_ckpt.py \
  --runs /path/to/nyuv2_run \
  --dataset_name nyuv2 \
  --data_root ./data/nyuv2 \
  --split val \
  --nyuv2_proxy_stats ./outputs/nyuv2_proxy_stats.json
```

## Notes

- The code writes outputs under `./outputs` by default.
- Default dataset paths are relative placeholders and should be overridden with `--data_root`.
- Pass an appropriate `--device` for your machine. Multi-GPU users can set `training.multi_gpu` and `training.gpu_ids` in `config.py`.
- Clinical raw data, processed benchmark files, and derived compact tensors must not be redistributed unless the applicable data-use agreement explicitly permits it.
- This repository does not include scripts that construct benchmark tensors from restricted clinical database tables or commands for downloading restricted clinical datasets.
- No third-party source code is vendored. External packages, datasets, and pretrained weights remain under their own licenses and access terms.
- Generated output paths are redacted where practical, but users should still review artifacts before sharing them.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

The license applies to the code in this repository. Datasets, pretrained weights, and external packages remain under their own licenses and access terms.

## Smoke Check

The repository includes a dependency-light syntax and packaging check:

```bash
python3 scripts/smoke_check.py
```

Full training and evaluation require the dependencies in `requirements.txt` and the corresponding datasets.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ding2026fairness,
  title = {Is Fairness Truly Fair? Towards Reliable Lipschitz Fairness in Multi-Task Learning via Fixed-{$\delta$} Alignment},
  author = {Ding, Junbo and Zang, Xin and Pan, Chenchen and Song, Donghao and Zhu, Jiaxin and Guo, Danhuai},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year = {2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  isbn = {979-8-4007-2259-2/2026/08},
  doi = {10.1145/3770855.3817938},
  url = {https://doi.org/10.1145/3770855.3817938}
}
```

For dataset citations, follow the official citation instructions on the PhysioNet pages linked above for MIMIC-III and eICU.
