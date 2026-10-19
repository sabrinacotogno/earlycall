"""EarlyCall online app.

    streamlit run app/streamlit_app.py

Reads only app/data/ (written by scripts/step4_export_app_data.py):
the CHO simulated dataset and the result tables. No raw data, no solar data.

Tabs: About, Tunnels, Monitor, Compare (bioprocess), Solar (results only).
This file only calls the package and plots: no logic of its own.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))          # the package, without installing it

import numpy as np
import pandas as pd
import streamlit as st

from earlycall.batch_data import BatchData
from earlycall.evaluation import cv_groups, grouped_splits
from earlycall.methods import BatchGrouper, dataset_settings
from earlycall.plotting import plot_scores, plot_tunnels

DATA = APP_DIR / "data"
DATASET = "cho_base"
DATASET_LABEL = "CHO simulated dataset"
METHOD_NAMES = {"base_rate": "base rate", "correlation": "BatchGrouper", "ridge": "ridge",
                "logistic": "logistic", "multiway_pls": "PLS"}
SCORES = {"auc": "Ranking (area under the ROC curve)",
          "recall": "Recall: low runs with an alarm",
          "false_alarm": "False alarms: good runs with an alarm"}
PARAM_LABELS = {"VCD_1e6_per_mL": "viable cells (1e6/mL)", "glucose_gL": "glucose (g/L)",
                "lactate_gL": "lactate (g/L)", "ammonium_mM": "ammonium (mM)",
                "capacitance_pF_cm": "capacitance (pF/cm)", "O2_flow_mLmin": "oxygen flow (mL/min)"}


# ------------------------------------------------------------------ data and model
@st.cache_data
def load_batches() -> BatchData:
    """The CHO simulated dataset, with batches renamed batch1, batch2, ..."""
    data = BatchData.load(DATA, DATASET)
    data.batch_ids = [f"batch{i}" for i in range(1, data.n_batches + 1)]
    return data


@st.cache_resource
def fit_model():
    """Train/test split as in the evaluation (first fold); BatchGrouper fitted on train only."""
    data = load_batches()
    train_idx, test_idx = grouped_splits(cv_groups(data))[0]
    train, test = data.subset(train_idx), data.subset(test_idx)
    settings = dataset_settings(DATASET)
    model = BatchGrouper(corr_threshold=settings["corr_threshold"], min_group_size=3,
                         smoothing_window=settings["smoothing_window"]).fit(train)
    return model, train, test


@st.cache_data
def load_results(dataset: str) -> pd.DataFrame:
    results = pd.read_csv(DATA / "results_summary.csv")
    results = results[results["dataset"] == dataset].copy()
    results["method"] = results["method"].map(METHOD_NAMES).fillna(results["method"])
    return results


def tunnel_table(model: BatchGrouper) -> pd.DataFrame:
    return pd.DataFrame({
        "tunnel": range(1, model.n_groups_ + 1),
        "batches": [len(m) for m in model.groups_],
        "titer min (g/L)": model.group_range_[:, 0].round(2),
        "titer max (g/L)": model.group_range_[:, 1].round(2),
        "mean titer (g/L)": model.group_mean_.round(2),
        "alarm": ["yes" if m < model.low_threshold_ else "no" for m in model.group_mean_],
    })


def score_view(dataset: str, key: str) -> None:
    """Score selector, then the plot (left) and the same numbers as a table (right)."""
    results = load_results(dataset)
    metric = st.radio("Score", list(SCORES), format_func=SCORES.get, horizontal=True, key=key)
    plot_column, table_column = st.columns([3, 2])
    plot_column.pyplot(plot_scores(results, metric, SCORES[metric]))
    table = results.pivot(index="percent", columns="method", values=metric).round(2)
    table.index = [f"{p}%" for p in table.index]
    table_column.dataframe(table)


# ------------------------------------------------------------------ tabs
def tab_about(model: BatchGrouper, train: BatchData, test: BatchData) -> None:
    st.markdown(
        "Repeated industrial runs (here, cell-culture batches of 14 days) vary a lot, and their "
        "quality is measured only at the end. **EarlyCall** gives an early call during the run.\n\n"
        "1. **Tunnels.** Past batches are grouped when their curves are correlated on every parameter. "
        "Each group is a tunnel: a reference profile, a band, and the final titers of its batches.\n"
        "2. **Monitor.** A running batch is matched to a tunnel using only the data seen so far. "
        "The tunnel's mean titer is the estimate; an alarm is raised if it is below the threshold.\n"
        "3. **Compare.** How early the call can be trusted, against standard models.\n\n"
        "Every call points back to real past batches, so it can be checked and explained."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training batches", train.n_batches)
    c2.metric("Tunnels found", model.n_groups_)
    c3.metric("Test batches", test.n_batches)
    c4.metric("Low titer below", f"{model.low_threshold_:.2f} g/L")
    st.caption(f"Data: {DATASET_LABEL} (a simplified cell-culture model written for this project). "
               "Low = bottom 30% of the training titers.")


def tab_tunnels(model: BatchGrouper, train: BatchData) -> None:
    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{model.n_groups_} tunnels** from {train.n_batches} training batches "
                    f"({int((model.labels_ < 0).sum())} batches fit no tunnel).")
        st.dataframe(tunnel_table(model), hide_index=True)
    with right:
        param = st.selectbox("Parameter", train.params, format_func=lambda p: PARAM_LABELS.get(p, p),
                             key="tunnel_param")
        p = train.params.index(param)
        st.pyplot(plot_tunnels(train.percent, model.reference_[:, p], model.band_std_[:, p], model.group_mean_,
                               model.low_threshold_, PARAM_LABELS.get(param, param)))
        st.caption("Green tunnel: mean titer above the threshold. Orange: below (alarm).")


def tab_monitor(model: BatchGrouper, test: BatchData) -> None:
    c1, c2 = st.columns([1, 3])
    batch = c1.selectbox("Test batch", test.batch_ids)
    t_index = c2.select_slider("Time into the batch", options=list(range(3, test.n_time + 1)),
                               value=test.n_time // 3, format_func=lambda t: f"{test.percent[t - 1]:.0f}%")
    b = test.batch_ids.index(batch)
    run = test.X[b:b + 1]
    group = int(model.assign(run, t_index)[0])
    estimate = float(model.predict_quality(run, t_index)[0])
    day = test.time[t_index - 1] / 24

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day", f"{day:.1f} of 14")
    m2.metric("Tunnel", "none" if group < 0 else str(group + 1))
    if group < 0:
        m3.metric("Estimated titer", "not enough information")
    else:
        alarm = estimate < model.low_threshold_
        m3.metric("Estimated titer", f"{estimate:.2f} g/L", "alarm" if alarm else "no alarm",
                  delta_color="inverse" if alarm else "normal")
    truly_low = bool(model.is_low(test.y[b:b + 1])[0])
    m4.metric("True final titer", f"{test.y[b]:.2f} g/L", "low" if truly_low else "ok",
              delta_color="inverse" if truly_low else "normal")

    left, right = st.columns([2, 3])
    with left:
        param = st.selectbox("Parameter", test.params, format_func=lambda p: PARAM_LABELS.get(p, p),
                             key="monitor_param")
        if group < 0:
            st.info("No tunnel fits this batch on every parameter yet: no estimate, no alarm.")
        else:
            low, high = model.group_range_[group]
            st.markdown(f"Tunnel {group + 1} holds **{len(model.groups_[group])} past batches**. "
                        f"Their final titer: {low:.2f} to {high:.2f} g/L.")
        st.caption("Strong colour: the tunnel of this batch. Dark line: the batch so far.")
    p = test.params.index(param)
    right.pyplot(plot_tunnels(test.percent, model.reference_[:, p], model.band_std_[:, p], model.group_mean_,
                              model.low_threshold_, PARAM_LABELS.get(param, param),
                              highlight=None if group < 0 else group, run=test.X[b, p], t_index=t_index))


def tab_compare() -> None:
    st.markdown("Each method estimates the final titer from the data seen so far. "
                "Alarm: estimate below the threshold. Scores on test batches, mean of 5 folds.")
    score_view(DATASET, key="compare_score")


def tab_solar() -> None:
    st.markdown(
        "The same idea on real solar inverter data (Kaggle). One run = one inverter from 06:00 to 18:30. "
        "Each inverter is compared with the other inverters at the same moment, which removes the weather. "
        "Known faults (failed string, stuck tracker) are injected into 30% of the runs. "
        "Each method gives the probability that a run is faulty; alarm when it is above the share of "
        "faulty runs in training.\n\n"
        "Only the scores are shown: the solar data cannot be redistributed."
    )
    plant = st.radio("Plant", ["solar_plant1", "solar_plant2"], horizontal=True,
                     format_func=lambda name: name.replace("solar_plant", "Plant "))
    score_view(plant, key="solar_score")


def main() -> None:
    st.set_page_config(page_title="EarlyCall", page_icon="📈", layout="wide")
    st.title("EarlyCall")
    st.markdown(f"Early monitoring of repeated industrial runs · {DATASET_LABEL}")
    model, train, test = fit_model()
    about, tunnels, monitor, compare, solar = st.tabs(["About", "Tunnels", "Monitor", "Compare", "Solar"])
    with about:
        tab_about(model, train, test)
    with tunnels:
        tab_tunnels(model, train)
    with monitor:
        tab_monitor(model, test)
    with compare:
        tab_compare()
    with solar:
        tab_solar()


if __name__ == "__main__":
    main()
