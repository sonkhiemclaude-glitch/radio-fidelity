# Radio Fidelity

Experiments on radio-simulator fidelity for wireless localization.

This repository contains scientific source code, saved aggregate results and
CSV exports. The companion data package supplies processed radio/map inputs,
per-query predictions, bootstrap replicates and result figures. Manuscript
files, reviewer correspondence and author contact details are not distributed here.

## Data package

[Download the replication package and raw inputs](https://drive.google.com/drive/folders/1Le9suMBBG_EYtQN4XrNiqDLnkGkE5FJJ).

Extract `radio-fidelity-replication.zip`, which contains a `radio-fidelity/`
directory with code, data, results, figures and checksums. To combine it with a
clone of this repository, copy the extracted directory's contents into the
clone. The separately supplied `raw/` inputs can be copied into the project
`raw/` directory; alternatively run `code/download_data.py` to fetch the same
public sources and verify their pinned SHA256 hashes.

`SHA256SUMS.json` in the Drive folder verifies the ZIP and raw downloads.
`ARTIFACT_SHA256.json` inside the ZIP verifies each package member.
The source-tree manifest in this repository is `SOURCE_SHA256.json`.

## Quick verification

Use Linux with Python 3.12 and an NVIDIA CUDA GPU for training. The recorded
environment used an A10G, 8 vCPUs and approximately 32 GiB RAM.

```bash
bash code/setup_ec2.sh
.venv/bin/python code/test_revision.py
.venv/bin/python code/test_spatial_bootstrap.py
.venv/bin/python code/validate_revision.py
.venv/bin/python code/export_results.py
```

Validation requires the companion data package, including saved predictions.
It checks 324 earlier synthetic evaluation rows and nine real-only rows, 72
map/zero-raster synthetic rows and six real-only rows, 74 paired spatial interval
rows, 610,000 link-admission campaigns, 432,000 downstream campaigns, 108 indoor
evaluations, and agreement with 3,000 scalar vector-geometry references.

## Reproduce

```bash
bash code/run_revision.sh
```

This downloads/verifies public raw inputs, checks preprocessing, runs the
additional experiments, rebuilds the dated-map result tables and validates the
outputs. It reuses checkpointed model results. To retrain, move the relevant
`exp13_models.json` or `exp17_map_localizer.json` and matching `data/pred_*`
files aside first. Geometry caches must match their map and resolution.
The original complete experiment suite can be rerun with `bash code/runall.sh`;
its outputs are isolated under `results/legacy_rerun/`.

| Experiments | Purpose |
|---|---|
| `exp1`--`exp10` | Likelihood, fidelity ladder, theory, transfer and robustness |
| `exp11` | Corrected admission using whole-message and spatial sampling units |
| `exp12`, `exp16` | 5/2/1 m raster sensitivity and full vector intersections |
| `exp13` | k-NN, residual MLP and gateway-geometry Transformer |
| `exp14` | Admission on downstream clipped mean positioning loss |
| `exp15` | Indoor device holdout with training-only calibration |
| `exp17` | Direct map-raster CNN and paired zero-raster ablation |
| `exp18` | Paired spatial-block intervals for positioning comparisons |

`results/` holds authoritative JSON. `reports/tables/` contains lossless CSV
exports. Archived earlier outputs are in `results/submission/`; intermediate
superseded outputs, supplied only in the data package, are not final evidence.
`raw/manifest.json` records public sources and hashes. `DATA_SOURCES.txt` gives
attribution and licence notices. No new software licence has been assigned.

## Interpretation

The range of differences between losses clipped to `[0,L]` is `2L`. Revised
admission guarantees are conditional on independent draws from the fixed
empirical corpus, not on independent future deployment observations. A safe
prefix selector need not recover a minimum beyond a harmful intermediate level.
The original interpretation of 1,000 correlated links as a certified budget is
withdrawn; archived earlier outputs must be read with this correction.

Likelihood-optimal calibration and localization-optimal fidelity can differ.
Architecture and training protocol affect rankings. The map-raster branch does
not consistently outperform its paired zero-raster ablation. Spatial-bootstrap
intervals hold trained seeds fixed, are pointwise and approximate, and do not
cover retraining, architecture or new-deployment uncertainty. Fitted-coefficient
and training-only indoor calibration arms use real calibration labels.

The original raw OSM snapshot date and shared statistical helper were unavailable.
Original processed inputs/results are preserved, while dated-map reconstructions
and newly specified helper methods are reported separately.
