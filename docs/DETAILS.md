# EarlyCall: details

This document explains the problem, the methods and the evaluation.

## 1. Problem

Many industrial processes run as **batches**: repeated runs with a start, an end and a final outcome.

| Domain | One batch | Final outcome |
|---|---|---|
| Bioprocess (fed-batch culture) | one bioreactor run, 14 days | final titer (product concentration) |
| Solar PV | one inverter over one run period (06:00–18:30) | faulty yes/no (faults injected into real runs) |

Questions an operator asks during the run:
1. **Which known behaviour does this run follow?** (grouping / reference profiles)
2. **How likely is it to end low, and in which quality range?** (early risk assessment)

### Bioprocess: early prediction and a target to steer toward

BatchGrouper matches a running batch to groups of similar past runs. It predicts early how the batch will end.

The key output is the golden tunnel: the group with the best yield. It is the target trajectory to steer the batch back to. Combined with tools that test and recommend corrective actions, it is one building block of a closed-loop batch assistant.

**Validation.** BatchGrouper meets the criteria of GAMP validation (the pharma guideline for computerised systems):
- **Traceable:** each prediction points to real past runs.
- **Transparent:** no hidden weights. The threshold, the groups and their members can be inspected.
- **Reproducible:** the same data always gives the same groups.

### Solar: detection and diagnosis

Each inverter is compared with its peers at the same moment. This removes the effect of the weather. Faults are injected into real runs, so the true answer is known.

Two tools, two jobs:
- **Detection:** models retrained at each hour flag a fault as soon as it starts, at any time of day.
- **Diagnosis:** BatchGrouper matches a flagged run to similar past runs. It works for faults that leave a regular, recognizable footprint in the data, such as a stuck tracker.

## 2. Methods

Every method is trained on past runs. During a new run, it uses **only the data seen so far** (up to `t_index`).

### Grouping: BatchGrouper

1. **Similarity:** for each parameter, the Pearson correlation between every pair of past runs. Two runs are linked if the correlation is above a threshold on **every** parameter.
2. **Groups:** sets of runs that are all linked to each other. The search is exact: a run found in several groups stays in the group with the highest average correlation. The same data always gives the same groups. If the exact search is too slow (very large datasets), a faster greedy search is used and a warning is shown.
3. **Each group** has a reference profile (mean curve), a band (spread) and the outcomes of its runs.
4. **During a new run:** the run is compared with each group's reference profile, cut at the current time. It joins the best group that fits on every parameter, or no group.
5. **Diagnosis** (`diagnose`): the matched group, its past runs, how many ended low and, for solar, the most common fault type among them.

### What each method returns

| | Bioprocess (outcome: titer from group of reference) | Solar (outcome: faulty yes/no) |
|---|---|---|
| Output | estimated final titer | probability of a fault |
| `BaseRate` | none (no information) | share of faulty runs in training |
| `BatchGrouper` | mean titer of the matched group | share of faulty runs in the matched group |
| Summary features + regression | `RidgeOnFeatures`: predicted titer | `LogisticOnFeatures`: probability |
| Multiway PLS (Nomikos & MacGregor): each run unfolded into one row | `MultiwayPLSRegression`: predicted titer | `MultiwayPLS`: PLS components, then logistic regression |
| Alarm | estimated titer below the threshold (30th percentile of training titers) | probability above the share of faulty runs in training |
| No group fits (BatchGrouper) | not enough information: no alarm | not enough information: no alarm |

- Summary features: last value, mean, min, max and recent change of each parameter.
- Ridge, logistic regression and PLS train **one model per cut time**, on the training runs cut at that same time. Nothing about the future is invented.
- Compared for bioprocess: `BaseRate`, `RidgeOnFeatures`, `BatchGrouper`, `MultiwayPLSRegression`.
- Compared for solar: `BaseRate`, `LogisticOnFeatures`, `BatchGrouper`, `MultiwayPLS`.

## 3. Data

- **Simulated CHO fed-batch runs** (`src/earlycall/data/cho_simulator.py`): a simplified mechanistic model (cell growth and death, glucose, lactate, glutamine, ammonium, titer) with pH, oxygen and temperature control. It writes realistic lab exports: online sensors every 15 min, daily offline samples, measurement noise, missing values and labelled faults (logger gap, pH probe drift, missed feed...). 3 recipes with different final titers; stress scenarios `level_shift` and `time_shift`. Ground truth (recipe, faults) is known.
- **Simple synthetic curves** (`src/earlycall/data/simulate.py`): fast generator used by the unit tests.
- **Real solar PV**: "Solar Power Generation Data" (Kaggle, © original authors, not redistributed here). Two plants, ~34 days, 15-min data, ~22 inverters per plant. One batch = one run: one inverter from 06:00 to 18:30. See [`data/README.md`](data/README.md).

## 4. Evaluation protocol

- Splits are made by **run**, never by time point.
- Thresholds, references and models are fitted on training runs only.
- The outcome is never used as an input. 
- Scores are computed at several moments of the run: how early can each method be trusted?
- Scores:
  - **area under the ROC curve:** how well the method ranks low runs above good ones (no threshold);
  - **recall:** share of low runs that raise an alarm;
  - **false alarm rate:** share of good runs that raise an alarm;
  - solar only: area under the ROC curve for each fault type.

## 5. Project structure

```
earlycall/
├── data/                  raw/ (not versioned), processed/ (generated CSV)
├── src/earlycall/
│   ├── paths.py           project folders (work from scripts, notebooks, tests)
│   ├── batch_data.py      BatchData: X (n_batches, n_params, n_time) in memory, two CSV tables on disk
│   ├── preprocessing.py   fill short gaps, drop incomplete batches
│   ├── data/              cho_simulator.py + cho.py (simulated bioprocess), solar.py (Kaggle),
│   │                      simulate.py (simple curves for tests)
│   ├── methods/           base.py (shared interface), baseline.py, correlation.py, multiway.py
│   ├── evaluation.py      grouped splits, scores vs time into the run
│   └── plotting.py
├── scripts/               step1_*, step2_*, step3_* command-line entry points
├── app/                   Streamlit demo
├── notebooks/             exploration only, no logic
├── docs/                  DETAILS.md (this page)
├── tests/
└── reports/figures/
```

## 6. Limitations

- Pearson correlation is blind to level and scale; the reference band compensates partially.
- Batches are aligned on clock time. Bioprocess runs should be aligned on process phases (e.g. when temperature reaches its new setpoint); solar runs on sunrise–sunset over a full year.
- Thresholds (`corr_threshold`, `min_group_size`) are set by hand; sensitivity is shown in the app.
- Not yet included: nearest-neighbour analogs, multiway PCA control charts (T², reconstruction error), time warping.
- With hundreds of runs, normal-behaviour models (gradient boosting, autoencoders) become an option.

