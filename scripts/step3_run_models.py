"""Step 3 - Run and compare all methods on every processed dataset.

    python scripts/step3_run_models.py              # all datasets found in data/processed
    python scripts/step3_run_models.py solar_plant2 # only the ones named

Output: reports/results.csv (one row per dataset, method, fold, cut point;
        solar also has one AUC per injected fault type)
        reports/figures/auc_<dataset>.png
"""
import sys

import pandas as pd

from earlycall.batch_data import BatchData
from earlycall.evaluation import evaluate, summarize
from earlycall.methods import dataset_settings, default_methods
from earlycall.paths import DATA_PROCESSED, FIGURES, REPORTS
from earlycall.plotting import plot_metric_vs_time

# Correlation method only: trailing moving average over this many time points (0 = no smoothing).
# Solar: 1 point = 15 min, so 4 = one hour. None = use the dataset default (methods.dataset_settings).
SMOOTHING_WINDOW = 5


def available_datasets() -> list[str]:
    """Names of the processed datasets, e.g. 'solar_plant1' for solar_plant1_values.csv."""
    return sorted(p.name.removesuffix("_values.csv") for p in DATA_PROCESSED.glob("*_values.csv"))


def run_dataset(name: str) -> pd.DataFrame:
    data = BatchData.load(DATA_PROCESSED, name)
    settings = dataset_settings(name)                   # low threshold, smoothing, correlation threshold
    if SMOOTHING_WINDOW is not None:
        settings["smoothing_window"] = SMOOTHING_WINDOW
    print(f"{name}: {data.n_batches} batches, settings {settings} ...", flush=True)
    cause_column = "fault_type" if "fault_type" in data.meta.columns else None   # solar: injected faults
    results = evaluate(default_methods(**settings), data, cause_column=cause_column,
                       estimate_quality=settings["estimate_quality"])
    results.insert(0, "dataset", name)

    summary = summarize(results)
    print(summary.pivot(index="method", columns="percent", values="auc").round(2).to_string(), "\n")
    for column in [c for c in summary.columns if c.startswith("auc_")]:
        print(f"{column} (one fault type against healthy runs):")
        print(summary.pivot(index="method", columns="percent", values=column).round(2).to_string(), "\n")

    fig = plot_metric_vs_time(summary, metric="auc", title=f"{name}: how early can we tell?")
    fig.savefig(FIGURES / f"auc_{name}.png", dpi=150)
    return results


def main() -> None:
    names = sys.argv[1:] or available_datasets()
    if not names:
        print("No processed data found: run the step 1 and step 2 scripts first.")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = pd.concat([run_dataset(name) for name in names], ignore_index=True)
    results.to_csv(REPORTS / "results.csv", index=False)
    print(f"Results: {REPORTS / 'results.csv'}, figures: {FIGURES}")


if __name__ == "__main__":
    main()
