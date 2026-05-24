# Environment

Recommended:

- Python 3.10+
- PyTorch 2.0+
- torchvision for NYUv2 / ResNet backbones
- CUDA-capable GPU for full experiments

Install:

```bash
python3 -m pip install -r requirements.txt
```

Dependency-light smoke checks:

```bash
python3 scripts/smoke_check.py
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q .
```

Post-install CLI checks:

```bash
PYTHONPATH=. python3 main.py --help
PYTHONPATH=. python3 run_baseline.py --help
PYTHONPATH=. python3 eval.py --help
PYTHONPATH=. python3 eval_ckpt.py --help
PYTHONPATH=. python3 compute_nyuv2_proxy_stats.py --help
```

The code can parse and import on CPU-only machines, but full training was designed for GPU execution.
