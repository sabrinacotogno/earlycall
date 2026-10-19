# EarlyCall

**Early monitoring of repeated industrial runs: bioprocess batches and solar inverter runs.**

During a run, EarlyCall answers two questions:
1. Which past runs does this run look like?
2. Will it end badly?

## Idea

Past runs are grouped by the shape of their trajectories: two runs are similar if their curves are correlated on every parameter. Each group has a reference profile and known outcomes. A new run is matched to a group using only the data seen so far.

- **Bioprocess:** the matched group gives an early estimate of the final titer, and its reference profile is a target trajectory to steer the batch toward.
- **Solar:** the matched group says whether the run looks like past faulty runs, and which kind of fault.

Each prediction points back to real past runs, so it can be checked and explained.

## Methods

| Method | Role |
|---|---|
| `BatchGrouper` | groups of correlated past runs; the method of this project |
| `BaseRate` | ignores the run (sanity check) |
| `RidgeOnFeatures`, `MultiwayPLSRegression` | predictive models, compared on bioprocess |
| `LogisticOnFeatures`, `MultiwayPLS` | predictive models, compared on solar |

Details: [docs/DETAILS.md](docs/DETAILS.md).

## Data

- Simulated bioprocess runs (a simplified cell-culture model written for this project).
- Real solar inverter data from Kaggle (not redistributed here), with known faults injected. See [data/README.md](data/README.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
#or
conda create -n venv python=3.11 -y
conda activate venv

pip install -e ".[app,dev]"

# Step 1: raw data
python scripts/step1_makedata_cho.py         # simulated CHO lab exports -> data/raw/cho/<scenario>/
python scripts/step1_makedata_synthetic.py   # simple curves -> data/processed/ (already aligned)
#   solar: download the Kaggle files into data/raw/solar/ (see data/README.md)

# Step 2: aligned tables -> data/processed/<name>_values.csv and <name>_batches.csv
python scripts/step2_processdata_cho.py
python scripts/step2_processdata_solar.py

# Step 3: compare the methods -> reports/results.csv, reports/figures/auc_<name>.png
python scripts/step3_run_models.py             # all datasets (about 30 s)
python scripts/step3_run_models.py solar_plant2

streamlit run app/streamlit_app.py
pytest
```

## Background

The CHO simulator is a simplified mechanistic model written for this project, not a validated process model.

The correlation-grouping method is inspired by an approach I developed for bioprocess monitoring in a consulting project. This repository is an independent re-implementation: no code nor data from that project are used.

## License

MIT
