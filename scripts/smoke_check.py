#!/usr/bin/env python3
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

import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ".zenodo.json",
    "README.md",
    "LICENSE",
    "NOTICE",
    "RELEASE_AUDIT.md",
    "ENVIRONMENT.md",
    "requirements.txt",
    "config.py",
    "main.py",
    "run_baseline.py",
    "eval.py",
    "eval_ckpt.py",
    "eval_grad_norm.py",
    "compute_nyuv2_proxy_stats.py",
    "trainer.py",
    "metrics.py",
    "models/mtl_model.py",
    "models/lipschitz_regularizer.py",
    "models/band_controller.py",
    "data_loader/build.py",
    "data_loader/mimic3_dataset.py",
    "data_loader/eicu_dataset.py",
    "data_loader/nyuv2_dataset.py",
    "baselines/erm.py",
    "baselines/gradnorm.py",
    "baselines/pcgrad.py",
    "baselines/fairgrad.py",
    "baselines/uncertainty.py",
    "baselines/cats.py",
]

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".xlsx",
    ".pdf",
    ".log",
    ".pyc",
}

FORBIDDEN_NAMES = {"__pycache__", ".DS_Store", "preprocess_" + "eicu.py"}
FORBIDDEN_PATH_FRAGMENTS = {
    "raw" + "_root",
    "patient" + ".csv.gz",
    "vital" + "Periodic" + ".csv.gz",
    "apache" + "Aps" + "Var" + ".csv.gz",
}
APACHE_HEADER = 'Licensed under the Apache License, Version 2.0'


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    forbidden = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES or str(rel).endswith(".csv.gz"):
            forbidden.append(str(rel))
    if forbidden:
        print("Forbidden release artifacts found:")
        for path in forbidden:
            print(f"  - {path}")
        return 1

    syntax_errors = []
    missing_headers = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if APACHE_HEADER not in text[:800]:
            missing_headers.append(str(rel))
        try:
            compile(text, str(rel), "exec")
        except SyntaxError as exc:
            syntax_errors.append((str(rel), exc))
    if missing_headers:
        print("Python files missing Apache header:")
        for rel in missing_headers:
            print(f"  - {rel}")
        return 1
    if syntax_errors:
        print("Python syntax check failed:")
        for rel, exc in syntax_errors:
            print(f"  - {rel}: {exc}")
        return 1

    placeholder_hits = []
    for rel_name in ("LICENSE", "NOTICE", "README.md"):
        text = (ROOT / rel_name).read_text(encoding="utf-8")
        for marker in ("[" + "yyyy" + "]", "[" + "name of copyright owner" + "]", "Copyright " + "["):
            if marker in text:
                placeholder_hits.append(f"{rel_name}: {marker}")
    if placeholder_hits:
        print("Unreplaced license placeholders found:")
        for hit in placeholder_hits:
            print(f"  - {hit}")
        return 1

    sensitive_patterns = [
        "preprocess_" + "eicu",
        "patient" + ".csv.gz",
        "vital" + "Periodic" + ".csv.gz",
        "apache" + "Aps" + "Var" + ".csv.gz",
        "raw" + "_root",
        "stay" + "_id=",
        "episode" + "_file=",
        "Author list" + " to be updated",
        "To " + "appear",
    ]
    content_hits = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in sensitive_patterns:
            if pattern in text:
                content_hits.append(f"{rel}: {pattern}")
    if content_hits:
        print("Sensitive or placeholder content found:")
        for hit in content_hits:
            print(f"  - {hit}")
        return 1

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_override_markers = ("--r_" + "star", "--eta")
    override_hits = [marker for marker in readme_override_markers if marker in readme]
    if override_hits:
        print("README should not expose experiment-specific controller overrides:")
        for hit in override_hits:
            print(f"  - {hit}")
        return 1

    git_dir = ROOT / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "log", "--all", "--name-only", "--pretty=format:"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            history_text = result.stdout
            history_hits = sorted(fragment for fragment in FORBIDDEN_PATH_FRAGMENTS if fragment in history_text)
            history_hits.extend(sorted(line.strip() for line in history_text.splitlines() if Path(line.strip()).name in FORBIDDEN_NAMES))
            if history_hits:
                print("Forbidden content appears in git history:")
                for hit in history_hits:
                    print(f"  - {hit}")
                return 1

    file_checks = {
        ".zenodo.json": [
            '"upload_type": "software"',
            '"license": "apache-2.0"',
            '"identifier": "https://doi.org/10.1145/3770855.3817938"',
        ],
        "README.md": [
            "@inproceedings{ding2026fairness",
            "doi = {10.1145/3770855.3817938}",
            "url = {https://doi.org/10.1145/3770855.3817938}",
        ],
        "eval_ckpt.py": ["default=None, choices=['val', 'test']", "args.split = 'val' if str(config.dataset_name).lower() == 'nyuv2' else 'test'", "Collecting({split})"],
        "eval_grad_norm.py": ["default=None, choices=['val', 'test']", "args.split = 'val' if str(config.dataset_name).lower() == 'nyuv2' else 'test'"],
        "models/backbone.py": ["raise RuntimeError", "allow_random_init"],
        "eval.py": ["data_root': '<redacted>'", "_safe_saved_label"],
        "eval_ckpt.py": ["_safe_saved_label"],
        "eval_grad_norm.py": ["_safe_saved_label"],
        "compute_nyuv2_proxy_stats.py": ["data_root': '<redacted>'", "_safe_saved_label"],
    }
    semantic_hits = []
    for rel_name, markers in file_checks.items():
        text = (ROOT / rel_name).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                semantic_hits.append(f"{rel_name}: missing {marker}")
    if semantic_hits:
        print("Release semantic checks failed:")
        for hit in semantic_hits:
            print(f"  - {hit}")
        return 1

    obsolete_readme_markers = [
        "citation as pending",
        "complete BibTeX entry",
        "metadata is not finalized",
    ]
    obsolete_hits = [marker for marker in obsolete_readme_markers if marker in readme]
    if obsolete_hits:
        print("Obsolete pending-citation text remains in README:")
        for hit in obsolete_hits:
            print(f"  - {hit}")
        return 1

    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
