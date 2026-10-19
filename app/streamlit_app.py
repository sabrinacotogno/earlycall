"""EarlyCall demo app.

    streamlit run app/streamlit_app.py

Tab 1 Discover : groups found by BatchGrouper on the training runs
Tab 2 Monitor  : one test run at a chosen time: assigned group, estimated titer, alarm
Tab 3 Compare  : how early each method can tell (reads reports/results.csv)

Bioprocess: a group's estimate is the mean titer of its runs; alarm if it is below the
low threshold (30th percentile of training titers).
Solar: only Discover (its own view) and Compare. Quality is a yes/no injected fault, the input
is relative power (inverter power / median of the other inverters). A group is risky if its
share of faulty runs is above the share in the whole training set.

This file only calls the package and plots: no logic of its own.
"""
import numpy as np
import pandas as pd
import streamlit as st

from earlycall.batch_data import BatchData
from earlycall.evaluation import cv_groups, grouped_splits, summarize
from earlycall.methods import BatchGrouper, dataset_settings
from earlycall.paths import DATA_PROCESSED, REPORTS
from earlycall.plotting import plot_group_references, plot_metric_vs_time, plot_reference_band, plot_trajectories


def available_datasets() -> list[str]:
    return sorted(p.name.removesuffix("_values.csv") for p in DATA_PROCESSED.glob("*_values.csv"))


@st.cache_data
def load(name: str) -> BatchData:
    return BatchData.load(DATA_PROCESSED, name)


@st.cache_resource
def fit_on_first_fold(name: str, corr_threshold: float, min_group_size: int, smoothing_window: int):
    """Train/test split as in the evaluation (first fold), BatchGrouper fitted on train only."""
    data = load(name)
    train_idx, test_idx = grouped_splits(cv_groups(data))[0]
    train, test = data.subset(train_idx), data.subset(test_idx)
    model = BatchGrouper(corr_threshold=corr_threshold, min_group_size=min_group_size,
                         smoothing_window=smoothing_window,
                         low_threshold=dataset_settings(name)["low_threshold"]).fit(train)
    return model, train, test


def n_ungrouped(model: BatchGrouper) -> int:
    return int((model.labels_ < 0).sum())


def group_table(model: BatchGrouper) -> pd.DataFrame:
    """Bioprocess: one row per group, with its estimated titer and whether it raises an alarm."""
    return pd.DataFrame({
        "group": range(model.n_groups_),
        "size": [len(m) for m in model.groups_],
        "titer min": model.group_range_[:, 0].round(2),
        "titer max": model.group_range_[:, 1].round(2),
        "mean titer (estimate)": model.group_mean_.round(2),
        "alarm": ["yes" if m < model.low_threshold_ else "no" for m in model.group_mean_],
    })


def tab_discover(model: BatchGrouper, train: BatchData) -> None:
    st.write(f"The model found {model.n_groups_} groups among {train.n_batches} training runs "
             f"({n_ungrouped(model)} runs in no group). A run is low if its titer is below "
             f"{model.low_threshold_:.2f} g/L (30th percentile of the training runs). "
             "A group raises an alarm if its mean titer is below that threshold.")
    st.dataframe(group_table(model), hide_index=True)
    param = st.selectbox("Parameter", train.params, key="discover_param")
    st.pyplot(plot_trajectories(train, param, model.labels_))


def clock(hour: float) -> str:
    """6.25 -> '06:15'."""
    return f"{int(hour):02d}:{round(hour % 1 * 60):02d}"


def progress_text(name: str, test: BatchData, t_index: int) -> str:
    """One readable line: how far the batch is, and the time in the dataset's own units."""
    percent = test.percent[t_index - 1]
    now, start, end = test.time[t_index - 1], test.time[0], test.time[-1]
    if name.startswith("solar"):   # time = hour of the day
        return f"**{percent:.0f}% of the run done**: it is {clock(now)} (run period {clock(start)} to {clock(end)})"
    return f"**{percent:.0f}% of the run done**: {now:g} h out of {end:g} h (day {now / 24:.1f} of {end / 24:g})"


def tab_monitor(name: str, model: BatchGrouper, test: BatchData) -> None:
    run_id = st.selectbox("Test run", test.batch_ids)
    b = test.batch_ids.index(run_id)
    t_index = st.select_slider(
        "Time into the batch (% of batch end)",
        options=list(range(3, test.n_time + 1)),              # t_index = number of points seen
        value=test.n_time // 3,
        format_func=lambda t: f"{test.percent[t - 1]:.0f}%",
    )
    st.markdown(progress_text(name, test, t_index))

    X_run = test.X[b:b + 1]
    group = int(model.assign(X_run, t_index)[0])
    estimate = float(model.predict_quality(X_run, t_index)[0])
    truly_low = bool(model.is_low(test.y[b:b + 1])[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Assigned group", "none" if group < 0 else str(group))
    if group < 0:
        col2.metric("Estimated titer", "not enough information")
    else:
        alarm = estimate < model.low_threshold_
        col2.metric("Estimated titer", f"{estimate:.2f}", "alarm: low" if alarm else "no alarm",
                    delta_color="inverse" if alarm else "normal")
    col3.metric("True final titer", f"{test.y[b]:.2f}", "low" if truly_low else "ok",
                delta_color="inverse" if truly_low else "normal")

    if group >= 0:
        low, high = model.group_range_[group]
        st.write(f"Group {group}: its runs ended between {low:.2f} and {high:.2f} g/L.")
        param = st.selectbox("Parameter", test.params, key="monitor_param")
        p = test.params.index(param)
        st.pyplot(plot_reference_band(test.percent, model.reference_[group, p], model.band_std_[group, p],
                                      param, run=test.X[b, p], t_index=t_index))
    else:
        st.info("No group matches this run on every parameter: not enough information, no alarm.")


# ------------------------------------------------------------------ solar views
def tab_discover_solar(model: BatchGrouper, train: BatchData) -> None:
    st.markdown(
        f"The method found {model.n_groups_} groups among {train.n_batches} training runs "
        f"({n_ungrouped(model)} runs fit no group). In the whole training set, "
        f"{model.base_rate_:.0%} of the runs are faulty: a group is risky if its share is higher."
    )
    order = np.argsort(-model.group_risk_)                       # most faulty groups first
    table = pd.DataFrame({
        "group": order,
        "size": [len(model.groups_[g]) for g in order],
        "faulty runs": [f"{model.group_risk_[g]:.0%}" for g in order],
        "risky": ["yes" if model.group_risk_[g] > model.base_rate_ else "no" for g in order],
    })
    st.dataframe(table, hide_index=True, height=38 + 35 * 5)
    st.pyplot(plot_group_references(train.time, model.reference_[:, 0], model.group_risk_))


METHOD_NAMES = {"base_rate": "base rate (no information)", "correlation": "BatchGrouper",
                "ridge": "ridge regression", "logistic": "logistic regression",
                "multiway_pls": "multiway PLS"}
SCORE_NAMES = {"auc": "Ranking: area under the ROC curve (1 = perfect, 0.5 = coin flip)",
               "recall": "Recall: share of low runs that raise an alarm",
               "false_alarm": "False alarms: share of good runs that raise an alarm"}


def methods_caption(name: str) -> str:
    if name.startswith("solar"):
        return ("Each method gives the probability that the run is faulty. Alarm: probability above the "
                "share of faulty runs in the training set. Logistic regression and multiway PLS are "
                "trained at each cut time on the training runs cut at that time.")
    return ("Each method estimates the final titer. Alarm: estimate below the 30th percentile of the "
            "training titers. A run that matches no group: not enough information, no alarm.")


def tab_compare(name: str) -> None:
    results_file = REPORTS / "results.csv"
    if not results_file.exists():
        st.warning("Run `python scripts/step3_run_models.py` first.")
        return
    results = pd.read_csv(results_file)
    results = results[results["dataset"] == name]
    if results.empty:
        st.warning(f"No results for {name}: run `python scripts/step3_run_models.py {name}`.")
        return
    results["method"] = results["method"].map(METHOD_NAMES).fillna(results["method"])
    st.caption(methods_caption(name))
    metric = st.selectbox("Score", list(SCORE_NAMES), format_func=SCORE_NAMES.get)
    summary = summarize(results)
    st.pyplot(plot_metric_vs_time(summary, metric=metric, title=name))
    st.dataframe(summary.pivot(index="method", columns="percent", values=metric).round(2))


def main() -> None:
    st.set_page_config(page_title="EarlyCall", layout="wide")
    st.title("EarlyCall: \nearly monitoring of repeated runs")

    names = available_datasets()
    if not names:
        st.error("No processed data: run the step 1 and step 2 scripts first.")
        return

    with st.sidebar:
        name = st.selectbox("Dataset", names)
        defaults = dataset_settings(name)
        with st.expander("Advanced settings"):
            corr_threshold = st.slider("Correlation threshold", 0.5, 0.99, defaults["corr_threshold"], 0.01)
            min_group_size = st.slider("Minimum group size", 2, 10, 3)
            smoothing_window = st.slider("Smoothing window (points, 0 = none)", 0, 12,
                                         defaults["smoothing_window"])

    model, train, test = fit_on_first_fold(name, corr_threshold, min_group_size, smoothing_window)
    if name.startswith("solar"):
        discover, compare = st.tabs(["Discover", "Compare"])
        with discover:
            tab_discover_solar(model, train)
    else:
        discover, monitor, compare = st.tabs(["Discover", "Monitor", "Compare"])
        with discover:
            tab_discover(model, train)
        with monitor:
            tab_monitor(name, model, test)
    with compare:
        tab_compare(name)


if __name__ == "__main__":
    main()
