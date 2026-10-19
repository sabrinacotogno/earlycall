# Data

Nothing in `raw/` or `processed/` is versioned (see `.gitignore`).

## Processed format (output of step 2)

Each dataset `<name>` is two CSV files in `processed/`:

| File | One row per | Columns |
|---|---|---|
| `<name>_values.csv` | batch and time point | `batch_id`, `time`, one column per parameter |
| `<name>_batches.csv` | batch | `batch_id`, `quality`, extra info (recipe, faults, inverter, date...) |

All batches share the same time grid and have no missing values. `BatchData.load(folder, name)` reads them into the 3-D array used by the methods. A `<name>_cleaning_report.json` says what cleaning did.

## Simulated CHO fed-batch runs

Generated, no download (about 25 s per scenario):

```bash
python scripts/step1_makedata_cho.py      # raw lab exports -> data/raw/cho/<scenario>/
python scripts/step2_processdata_cho.py   # -> data/processed/cho_<scenario>_{values,batches}.csv
```

Each scenario folder contains:

| File | Content |
|---|---|
| `online_timeseries.csv` | sensors every 15 min: pH, DO, temperature, gas flows, base and feed added, weight, capacitance |
| `offline_samples.csv` | daily samples: VCD, viability, cell diameter, glucose, lactate, glutamine, glutamate, ammonium, osmolality, pH, pCO2, pO2, titer (from day 3) |
| `runs_metadata.csv` | one row per run: condition (true recipe), injected faults, sampled kinetic parameters |

Scenarios: `base` (3 recipes), `level_shift` (+ a recipe with the same shape at a lower level), `time_shift` (+ a recipe with delayed growth). 48 runs each, ~25% with one fault.

## Simple synthetic curves

Used by the unit tests:

```bash
python scripts/step1_makedata_synthetic.py
```

## Solar PV (Kaggle)

Dataset: "Solar Power Generation Data" (kaggle.com/datasets/anikannal/solar-power-generation-data).
Licence on Kaggle: "Data files © Original Authors" (no open licence).
The data is therefore NOT redistributed in this repository, neither raw nor processed:
download it from Kaggle to reproduce the results. Check the author name and year on the page when citing.

Download (needs a Kaggle account and `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d anikannal/solar-power-generation-data -p data/raw/solar --unzip
```

Expected files in `data/raw/solar/`:

| File | Content |
|---|---|
| `Plant_1_Generation_Data.csv` | per inverter (`SOURCE_KEY`), 15 min: `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` |
| `Plant_1_Weather_Sensor_Data.csv` | per plant, 15 min: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` |
| `Plant_2_...` | same for plant 2 |

Sanity checks:
- date format differs between the files: parse explicitly, never let pandas guess
- missing 15-min timestamps and duplicates
- `DC_POWER` vs `AC_POWER` ratio (unit problem?)
- inverters stuck at zero or frozen during daytime
- irradiation > 0 at night or power > 0 with zero irradiation
