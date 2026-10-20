"""EarlyCall online app.

    streamlit run app/streamlit_app.py

Reads only app/data/ (written by scripts/step4_export_app_data.py):
the CHO simulated dataset and the result tables. No raw data, no solar data.

Pages: About (the tool, generic) | Bioproduction (main application) | Solar (use case, results only).
This file only calls the package and plots: no logic of its own.
"""
import base64
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))          # the package, without installing it

import pandas as pd
import streamlit as st

from earlycall.batch_data import BatchData
from earlycall.evaluation import cv_groups, grouped_splits
from earlycall.methods import BatchGrouper, dataset_settings
from earlycall.plotting import (plot_quality_distribution, plot_quality_ranges, plot_runs_by_quality,
                                plot_scores, plot_tunnels)

DATA = APP_DIR / "data"
DATASET = "cho_base"
METHOD_NAMES = {"base_rate": "base rate", "correlation": "BatchGrouper", "ridge": "ridge",
                "logistic": "logistic", "multiway_pls": "PLS"}
SCORES = {"auc": "Ranking (area under the ROC curve)",
          "recall": "Recall: low runs with an alarm",
          "false_alarm": "False alarms: good runs with an alarm"}
PARAM_LABELS = {"VCD_1e6_per_mL": "viable cells (1e6/mL)", "glucose_gL": "glucose (g/L)",
                "lactate_gL": "lactate (g/L)", "ammonium_mM": "ammonium (mM)",
                "capacitance_pF_cm": "capacitance (pF/cm)", "O2_flow_mLmin": "oxygen flow (mL/min)"}

TUNNEL_FIGURE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 320" role="img" aria-label="Three tunnels of past runs over time and a current run
 that follows the high-quality tunnel up to now" style="width:100%;max-width:860px;font-family:sans-serif">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7"
   orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9aa3ab"/></marker></defs>
  <path d="M62,245 C300,235 420,80 880,45 L880,95 C420,128 300,262 62,262 Z" fill="#d9822b" fill-opacity="0.30"/>
  <path d="M62,255 C300,250 460,190 880,190 L880,236 C460,238 300,270 62,272 Z" fill="#9aa3ab" fill-opacity="0.35"/>
  <path d="M62,250 C300,242 440,130 880,115 L880,165 C440,178 300,266 62,266 Z" fill="#0f7b6c" fill-opacity="0.35"/>
  <path d="M62,258 C200,253 340,205 540,168" fill="none" stroke="#14213d" stroke-width="5" stroke-linecap="round"/>
  <circle cx="540" cy="168" r="8" fill="#14213d"/>
  <line x1="540" y1="22" x2="540" y2="288" stroke="#14213d" stroke-width="1.5" stroke-dasharray="7 6"/>
  <path d="M62,290 V14" stroke="#9aa3ab" stroke-width="1.5" marker-end="url(#arrow)"/>
  <path d="M62,290 H888" stroke="#9aa3ab" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="550" y="34" font-size="15" fill="#14213d">now</text>
  <text x="250" y="268" font-size="15" fill="#14213d">current run</text>
  <text x="872" y="74" font-size="16" font-weight="600" fill="#b4541a" text-anchor="end">low-quality tunnel</text>
  <text x="872" y="145" font-size="16" font-weight="600" fill="#0b5a50" text-anchor="end">high-quality tunnel</text>
  <text x="872" y="218" font-size="16" font-weight="600" fill="#5b6470" text-anchor="end">medium-quality tunnel</text>
  <text x="884" y="310" font-size="13" fill="#5b6470" text-anchor="end">time</text>
  <text x="70" y="16" font-size="13" fill="#5b6470">process parameter</text>
</svg>
"""


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


def label(param: str) -> str:
    return PARAM_LABELS.get(param, param)


def tunnel_table(model: BatchGrouper) -> pd.DataFrame:
    return pd.DataFrame({
        "tunnel": range(1, model.n_groups_ + 1),
        "batches": [len(m) for m in model.groups_],
        "titer min": model.group_range_[:, 0].round(2),
        "titer max": model.group_range_[:, 1].round(2),
        "mean titer": model.group_mean_.round(2),
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


# ------------------------------------------------------------------ About
def page_about() -> None:
    st.markdown(
        "#### Know early how a run will end\n"
        "Many industrial processes work in **repeated runs**: a batch in a reactor, a production cycle, "
        "a working day of a machine. Runs that start alike can end very differently, and the outcome is "
        "often known only at the end, when it is too late to act. "
        "EarlyCall compares a run that is still going with the runs of the past, and gives an early call."
    )
    svg = base64.b64encode(TUNNEL_FIGURE.encode()).decode()
    st.markdown(f'<img src="data:image/svg+xml;base64,{svg}" style="width:100%;max-width:860px">',
                unsafe_allow_html=True)
    st.caption("Past runs with similar trajectories form tunnels, each with a known outcome. "
               "The current run is compared with the tunnels using only the data seen so far.")

    st.markdown("#### How it works")
    steps = [
        ("1 · Learn from past runs",
         "Two runs are similar when their curves are correlated on every parameter. Groups of runs that "
         "are all similar to each other form tunnels: a reference profile, a tolerance band and the "
         "outcomes of their runs."),
        ("2 · Match the current run",
         "At any moment, the current run is compared with each tunnel cut at the same time. It joins the "
         "tunnel it follows on every parameter, or none if no tunnel fits yet."),
        ("3 · Act early",
         "The tunnel gives an expected outcome and an alarm when it is poor. The best tunnel is also a "
         "target trajectory: a run can be steered back toward it."),
    ]
    for column, (title, text) in zip(st.columns(3), steps):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)

    st.markdown("#### Built to be trusted")
    st.markdown(
        "- **Traceable:** each call points to real past runs.\n"
        "- **Transparent:** no hidden weights; the threshold, the tunnels and their members can be inspected.\n"
        "- **Reproducible:** the same data always gives the same tunnels."
    )


# ------------------------------------------------------------------ Bioproduction
def tab_dataset(data: BatchData, model: BatchGrouper) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "**CHO simulated dataset.** Cell-culture batches (CHO cells producing an antibody) of 14 days, "
            "generated by a simplified mechanistic model written for this project:\n"
            "- cell growth and death, glucose, lactate, glutamine, ammonium and product (titer);\n"
            "- three recipes: temperature shift early, late or never;\n"
            "- realistic lab exports: online sensors every 15 minutes, daily samples, measurement noise, "
            "missing values and a few faults (logger gap, missed feed, probe dropout).\n\n"
            "**Outcome:** the final titer (g/L) on day 14. A batch is **low** when its titer is in the "
            "bottom 30% of the training batches."
        )
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Batches", data.n_batches)
        m2.metric("Parameters", data.n_params)
        m1.metric("Duration", "14 days")
        m2.metric("Low titer below", f"{model.low_threshold_:.2f} g/L")

    param = st.selectbox("Parameter", data.params, format_func=label, key="dataset_param")
    p = data.params.index(param)
    plot_column, histogram_column = st.columns([3, 2])
    plot_column.pyplot(plot_runs_by_quality(data.percent, data.X[:, p], data.y, label(param)))
    histogram_column.pyplot(plot_quality_distribution(data.y, model.low_threshold_))


def tab_tunnels(model: BatchGrouper, train: BatchData) -> None:
    st.markdown(f"**{model.n_groups_} tunnels** found among {train.n_batches} training batches "
                f"({int((model.labels_ < 0).sum())} batches fit no tunnel). "
                "Green: mean titer above the threshold. Orange: below, the tunnel raises an alarm.")
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(tunnel_table(model), hide_index=True)
        st.pyplot(plot_quality_ranges(model.group_range_, model.group_mean_, model.low_threshold_))
        st.caption("Final titer of each tunnel's batches (g/L): range and mean.")
    with right:
        param = st.selectbox("Parameter", train.params, format_func=label, key="tunnel_param")
        p = train.params.index(param)
        st.pyplot(plot_tunnels(train.percent, model.reference_[:, p], model.band_std_[:, p], model.group_mean_,
                               model.low_threshold_, label(param)))
        st.caption("Tunnel profiles: mean curve and band (2 standard deviations).")


def tab_monitor(model: BatchGrouper, test: BatchData) -> None:
    c1, c2 = st.columns([1, 3])
    batch = c1.selectbox("Test batch", test.batch_ids)
    t_index = c2.select_slider("Time into the batch", options=list(range(3, test.n_time + 1)),
                               value=test.n_time // 3, format_func=lambda t: f"{test.percent[t - 1]:.0f}%")
    b = test.batch_ids.index(batch)
    run = test.X[b:b + 1]
    group = int(model.assign(run, t_index)[0])
    estimate = float(model.predict_quality(run, t_index)[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day", f"{test.time[t_index - 1] / 24:.1f} of 14")
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
        param = st.selectbox("Parameter", test.params, format_func=label, key="monitor_param")
        if group < 0:
            st.info("No tunnel fits this batch on every parameter yet: no estimate, no alarm.")
        else:
            low, high = model.group_range_[group]
            st.markdown(f"Tunnel {group + 1} holds **{len(model.groups_[group])} past batches**. "
                        f"Their final titer: {low:.2f} to {high:.2f} g/L.")
        st.caption("Strong colour: the tunnel of this batch. Dark line: the batch so far.")
    p = test.params.index(param)
    right.pyplot(plot_tunnels(test.percent, model.reference_[:, p], model.band_std_[:, p], model.group_mean_,
                              model.low_threshold_, label(param),
                              highlight=None if group < 0 else group, run=test.X[b, p], t_index=t_index))


def tab_comparison() -> None:
    st.markdown("Each method estimates the final titer from the data seen so far; alarm when the estimate "
                "is below the threshold. BatchGrouper against two standard models (ridge regression on "
                "summary features, PLS on the whole trajectory) and a base rate that ignores the run. "
                "Scores on test batches, mean of 5 folds.")
    score_view(DATASET, key="bio_score")


def page_bioproduction() -> None:
    st.markdown("The main application: cell-culture batches, monitored from day 1 to day 14.")
    data = load_batches()
    model, train, test = fit_model()
    dataset, tunnels, monitor, comparison = st.tabs(["Dataset", "Tunnels", "Monitor", "Comparison"])
    with dataset:
        tab_dataset(data, model)
    with tunnels:
        tab_tunnels(model, train)
    with monitor:
        tab_monitor(model, test)
    with comparison:
        tab_comparison()


# ------------------------------------------------------------------ Solar
def page_solar() -> None:
    st.markdown(
        "A possible use case: solar inverters. One run = one inverter from 06:00 to 18:30. "
        "Each inverter is compared with the other inverters at the same moment, which removes the weather. "
        "Known faults (failed string, stuck tracker) are injected into 30% of the runs of real Kaggle data. "
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
    st.caption("Early warning for repeated industrial runs")
    about, bioproduction, solar = st.tabs(["About", "Bioproduction · main application", "Solar · use case"])
    with about:
        page_about()
    with bioproduction:
        page_bioproduction()
    with solar:
        page_solar()


if __name__ == "__main__":
    main()
