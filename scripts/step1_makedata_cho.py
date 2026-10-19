"""Step 1 - Simulate the CHO lab exports, one folder per scenario (about 25 s per scenario).

    python scripts/step1_makedata_cho.py

Output: data/raw/cho/<scenario>/{online_timeseries, offline_samples, runs_metadata}.csv
"""
from earlycall.data.cho_simulator import SCENARIOS, generate
from earlycall.paths import DATA_RAW


def main() -> None:
    for scenario in SCENARIOS:
        folder = DATA_RAW / "cho" / scenario
        generate(folder, n_runs=48, scenario=scenario, anomaly_rate=0.25, seed=42)
        print(f"{scenario}: written to {folder}")


if __name__ == "__main__":
    main()
