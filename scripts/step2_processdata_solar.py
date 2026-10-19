"""Step 2 - Kaggle solar files -> inverter-run tables with injected faults, one dataset per plant.

Quality = 1 healthy / 0 faulty (see earlycall/data/solar.py); input = relative_power.

    python scripts/step2_processdata_solar.py

Needs the Kaggle files in data/raw/solar/ (see data/README.md).
Output: data/processed/solar_plant<1|2>_{values,batches}.csv + a cleaning report.
"""
import json

from earlycall.data.solar import prepare_plant
from earlycall.paths import DATA_PROCESSED, DATA_RAW


def main() -> None:
    for plant in (1, 2):
        data, report = prepare_plant(DATA_RAW / "solar", plant)
        name = f"solar_plant{plant}"
        data.save(DATA_PROCESSED, name)
        (DATA_PROCESSED / f"{name}_cleaning_report.json").write_text(json.dumps(report, indent=2))
        print(f"{name}: {data.n_batches} runs, {data.n_params} params, {data.n_time} time points")


if __name__ == "__main__":
    main()
