"""Step 1 (+2) - Simple synthetic curves, one dataset per scenario.

    python scripts/step1_makedata_synthetic.py

The curves are generated already aligned, so the only processing is filling short gaps
and dropping incomplete runs. Output: data/processed/synthetic_<scenario>_{values,batches}.csv
"""
import json

from earlycall.batch_data import BatchData
from earlycall.data.simulate import SCENARIOS, simulate_batches
from earlycall.paths import DATA_PROCESSED
from earlycall.preprocessing import clean_tables


def main() -> None:
    for scenario in SCENARIOS:
        raw = simulate_batches(scenario=scenario, seed=0)
        values, batches = raw.to_tables()
        # the simulator makes holes of up to 4 points (simulate.add_missing): fill them all
        values, batches, report = clean_tables(values, batches, raw.params, max_gap=4)
        data = BatchData.from_tables(values, batches)

        name = f"synthetic_{scenario}"
        data.save(DATA_PROCESSED, name)
        (DATA_PROCESSED / f"{name}_cleaning_report.json").write_text(json.dumps(report, indent=2))
        print(f"{name}: {data.n_batches} batches, {data.n_params} params, {data.n_time} time points")


if __name__ == "__main__":
    main()
