# PMV-CRN

Multi-source temporal and evidence-map classification with concept-rule representations.

## Version Status

## Contents

| File | Purpose |
| --- | --- |
| `pmv_crn/models/` | Temporal and image encoders, concept-rule layers, fusion, and networks |
| `pmv_crn/data/` | Evidence representations, statistical features, splits, and datasets |
| `pmv_crn/training/` | Losses, batch forwarding, training engine, and balanced entry point |
| `pmv_crn/evaluation/` | Classification metrics, threshold selection, and prediction |
| `pmv_crn/visualization/` | Training curves, embeddings, and evidence exports |
| `pmv_crn/baselines/` | Comparison models, data, training, evaluation, and plots |
| `pmv_crn/config/` | Existing command-line configuration defaults |
| `pmv_crn/utils/` | Random seed setup |
| `scripts/` | Training and baseline script entry points |
| `configs/` | Definition-integrity manifest for the refactor |
| `tests/` | Structural and numerical regression checks |
| `docs/` | Architecture and compatibility documentation |
## Installation

Use Python 3.10 or newer in a virtual environment.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Activate the environment before installing or running. Install the appropriate PyTorch build for the intended CPU or CUDA environment. `umap-learn` is optional for UMAP visualizations.

## Input

Provide a CSV containing `AE`, `EMR`, `GAS`, and `LABEL`. An optional `date` column is parsed and used to sort the records. `LABEL` is binary: 0 and 1.

The input contains synchronized numeric monitoring sequences and existing labels.

## Training

Run commands from the repository directory after activating the environment.

```bash
python train.py --input-csv data/monitoring.csv --out-dir results/full --encoder inception --image-map-type sensor_trajectory9 --model-mode full --split-mode blocked_random_stratified --window 256 --img-size 32 --sample-stride 1 --epochs 60 --batch-size 256 --lr 0.0006 --dropout 0.20 --seed 2026
```

```bash
python train.py --help
```


## Baselines

```bash
python run_classifier_baselines.py --input-csv data/monitoring.csv --out-dir results/baselines --window 256 --sample-stride 1 --epochs 60 --batch-size 256 --lr 0.0006 --seed 2026
```

## Local Checks

```bash
python -m unittest discover -s tests -v
```
## Publication

This repository retains the GPL-3.0 license selected in the existing GitHub repository. See `LICENSE` for the complete terms. The authors remain responsible for permission to release the included implementation. Public repository: https://github.com/CUMT-EMR/PMV-CRN.
