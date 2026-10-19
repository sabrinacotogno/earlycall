"""Step 2 - CHO lab exports -> aligned run-level tables, one dataset per scenario.

    python scripts/step2_processdata_cho.py

Output: data/processed/cho_<scenario>_{values,batches}.csv + a cleaning report.
"""
import json

from earlycall.data.cho import prepare_scenario
from earlycall.data.cho_simulator import SCENARIOS
from earlycall.paths import DATA_PROCESSED, DATA_RAW


def main() -> None:
    for scenario in SCENARIOS:
        data, report = prepare_scenario(DATA_RAW / "cho" / scenario)
        name = f"cho_{scenario}"
        data.save(DATA_PROCESSED, name)
        (DATA_PROCESSED / f"{name}_cleaning_report.json").write_text(json.dumps(report, indent=2))
        print(f"{name}: {data.n_batches} runs, {data.n_params} params, {data.n_time} time points")


if __name__ == "__main__":
    main()
