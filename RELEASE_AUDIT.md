# Release Audit

This release directory is a sanitized mirror of the internal training code.

Allowed differences from the internal training source:

- Local absolute data paths were replaced with relative placeholder paths.
- Chinese comments and docstrings were removed from source files.
- Runtime messages, exception messages, and CLI help text were translated or generalized for public release.
- MIMIC-III and eICU retain their train/validation/test workflow; public examples use `--split test` for clinical evaluation.
- `eval.py` defaults to clinical `test` but switches the default to `val` for NYUv2 and rejects explicit NYUv2 `test`, preventing accidental NYUv2 test-set claims.
- `eval_ckpt.py` and `eval_grad_norm.py` reject explicit NYUv2 `test` for the same public split-contract reason.
- The public NYUv2 proxy-calibration CLI exposes only `--split val`, matching the released NYUv2 dataloader and README.
- README, environment notes, dependency pins, git ignore rules, a dependency-light smoke check, and this audit file were added for distribution.

Disallowed content check:

- GitHub source trees and generated release archives must not contain a `.git` directory. A local git checkout necessarily contains `.git`, so history is audited separately.
- No `__pycache__`, `.pyc`, `.DS_Store`, checkpoints, raw data, logs, generated result files, spreadsheets, or PDFs.
- No local user paths, server IPs, usernames, private credentials, or private workspace names.
- No script that constructs benchmark tensors from restricted clinical database tables is included.
- No clinical table download, extraction, or cohort-construction code is included.
- Git history is checked for removed restricted-data preparation scripts before publication.
- LOS debug-only raw distribution summaries were removed from the public evaluation output.
- Clinical exception and verbose diagnostic messages are sanitized to avoid printing raw stay ids or episode filenames by default.

Code-equivalence check:

- 32 Python source files were compared against the internal training source.
- The comparison normalized comments, docstrings, all string literals, f-strings, and approved path placeholders.
- Intentional public-release behavior differences are limited to NYUv2 split-contract guards, removal of restricted-data table-preparation utilities and debug-only clinical distribution summaries, release documentation, and the release-only `scripts/smoke_check.py`.
- No training loop, model, loss, metric, dataloader data transformation, or baseline optimizer algorithm change is intended.

Clinical-data note:

- Restricted clinical datasets require authorized access and institutional data-use compliance.
- The public package includes clinical dataloaders for authorized benchmark-style inputs but omits utilities that build benchmark tensors from restricted clinical database tables.
- Users must not redistribute raw clinical data, processed clinical benchmark files, or derived compact tensors unless the applicable data-use agreement explicitly permits it.
- The clinical loaders are provided only as reproducibility interfaces for users who already have compliant access and locally prepared benchmark files.

License note:

- `LICENSE` is included and uses Apache License 2.0.
- `NOTICE` is included with `Copyright 2026 Junbo Ding`.
- Python source files include Apache License boilerplate comments.
- Dependencies, datasets, and externally downloaded pretrained weights remain under their own licenses and access terms.
- No third-party source code is vendored in this release.

NYUv2 note:

- The public loader supports `train` and `val` splits.
- The public proxy-calibration entrypoint intentionally exposes `val` only, consistent with the released NYUv2 split contract.
- NYUv2 should be described as non-medical vision dense-prediction cross-domain validation, not as a separate NYUv2 test-set result.
- NYUv2 ReLiF and NYUv2 fixed-bias auditing require a shared `nyuv2_proxy_stats.json` generated from an ERM reference checkpoint.
