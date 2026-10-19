"""Step 4 - Copy what the online app needs into app/data/ (this folder is versioned).

    python scripts/step4_export_app_data.py

The online app cannot run the steps before (no raw data, and the solar data cannot be
redistributed). It only needs:
    cho_base_values.csv, cho_base_batches.csv   simulated bioprocess runs (free to share)
    results_summary.csv                         mean scores per method and cut point, for the
                                                bioprocess dataset and the two solar plants
"""
import shutil

import pandas as pd

from earlycall.evaluation import summarize
from earlycall.paths import DATA_PROCESSED, REPORTS, ROOT

APP_DATA = ROOT / "app" / "data"
BIOPROCESS = "cho_base"
RESULT_DATASETS = [BIOPROCESS, "solar_plant1", "solar_plant2"]


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    for table in ["values", "batches"]:
        shutil.copy(DATA_PROCESSED / f"{BIOPROCESS}_{table}.csv", APP_DATA / f"{BIOPROCESS}_{table}.csv")

    results = pd.read_csv(REPORTS / "results.csv")
    summaries = []
    for name in RESULT_DATASETS:
        summary = summarize(results[results["dataset"] == name])
        summaries.append(summary.assign(dataset=name))
    pd.concat(summaries, ignore_index=True).to_csv(APP_DATA / "results_summary.csv", index=False)
    print(f"App data written to {APP_DATA}")


if __name__ == "__main__":
    main()
