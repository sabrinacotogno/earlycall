"""
Claude (Claude Opus 5.5) generated synthetic dataset for CHO fed-batch bioreactor runs.

Synthetic CHO fed-batch bioreactor dataset generator (2 L bench-top scale, 14 days).

Simplified mechanistic model, NOT a validated process model:
ODE core (viable/dead cells, glucose, lactate, glutamine, glutamate, ammonium,
osmolality, IgG-like titer, volume) + simple on/off pH, DO and temperature control
+ measurement noise, realistic sampling schedule and injected, labelled faults.

It writes what a lab would export, so the loader (cho.py) has real cleaning to do:
    online_timeseries.csv   sensors every 15 min, one row per run and time
    offline_samples.csv     daily samples (VCD, metabolites, titer), with missing values
    runs_metadata.csv       one row per run: condition (ground truth), faults, parameters

Usage:
    python -m earlycall.data.cho_simulator --outdir data/raw/cho --n-runs 48 --scenario base
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DT = 1 / 12            # integration step: 5 min
LOG_EVERY = 3          # online logging every 15 min
T_END = 336.0          # h (14 days)
START = pd.Timestamp("2026-01-12 08:00")


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


NO_SHIFT = float("inf")

CONDITIONS = {
    # t_shift: temperature shift time (h), T_shift: target (degC), feed_frac: feed volume
    # fraction per day, xmax: VCD ceiling (1e6 cells/mL), vcd0: inoculation density
    # Main recipes. Mean final titer over 48 runs, seed 42: A ~4.5, B ~3.4, C ~1.9 g/L
    "A_lateTS":  dict(t_shift=144.0,    T_shift=33.0, feed_frac=0.040, xmax=52.0, vcd0=0.50),
    "B_earlyTS": dict(t_shift=72.0,     T_shift=33.0, feed_frac=0.040, xmax=52.0, vcd0=0.50),
    "C_noTS":    dict(t_shift=NO_SHIFT, T_shift=36.8, feed_frac=0.035, xmax=48.0, vcd0=0.50),
    # Stress variants of A, built to challenge the methods (titer ~3.6 and ~3.7)
    "A_lowXmax": dict(t_shift=144.0,    T_shift=33.0, feed_frac=0.040, xmax=32.0, vcd0=0.50),  # same shape, lower level
    "A_lowSeed": dict(t_shift=144.0,    T_shift=33.0, feed_frac=0.040, xmax=52.0, vcd0=0.20),  # growth delayed in time
}

SCENARIOS = {
    "base":        ["A_lateTS", "B_earlyTS", "C_noTS"],
    "level_shift": ["A_lateTS", "B_earlyTS", "C_noTS", "A_lowXmax"],
    "time_shift":  ["A_lateTS", "B_earlyTS", "C_noTS", "A_lowSeed"],
}

ANOMALIES = ["logger_gap", "ph_probe_drift", "capacitance_dropout", "temp_excursion", "missed_feed"]


def make_design(n_runs: int, conditions: list[str], anomaly_rate: float,
                rng: np.random.Generator) -> list[tuple[str, str, list[str]]]:
    """Return [(run_id, condition, anomalies)] with conditions cycled evenly
    and at most one fault per run, drawn with probability anomaly_rate."""
    design = []
    for k in range(n_runs):
        cond = conditions[k % len(conditions)]
        faults = [str(rng.choice(ANOMALIES))] if rng.random() < anomaly_rate else []
        design.append((f"R{k + 1:03d}", cond, faults))
    return design


def simulate(run_id, cond_name, anomalies, seed, run_index):
    rng = np.random.default_rng(seed)
    C = CONDITIONS[cond_name]

    # ---- batch-to-batch parameter variability -------------------------------------
    P = dict(
        mu_max=rng.lognormal(np.log(0.038), 0.05),
        qg0=0.012 * rng.lognormal(0, 0.07),        # glucose uptake, g/L/h per 1e6 cells/mL
        qgln0=0.0042 * rng.lognormal(0, 0.08),      # glutamine uptake, mM/h per 1e6 cells/mL
        qP0=1.0 * rng.lognormal(0, 0.08),           # specific productivity, pg/cell/h
        ylac=0.85 * rng.lognormal(0, 0.10),         # lactate yield on glucose, g/g
        lac_shift=rng.uniform(0.3, 1.1),            # strength of late lactate consumption
        kd_scale=rng.lognormal(0, 0.15),
        xmax=C["xmax"] * rng.normal(1, 0.06),
        vcd0=rng.normal(C["vcd0"], 0.05 * C["vcd0"]),
    )

    n = int(round(T_END / DT)) + 1
    n_log = (n - 1) // LOG_EVERY + 1

    # ---- initial state ------------------------------------------------------------
    Xv = P["vcd0"]; Xd = Xv * 0.02 / 0.98
    Glc, Gln, Glu, Lac, Amm, Osm, Ti = 6.0, 4.0, 0.8, 0.15, 0.4, 305.0, 0.0
    V = 2.0                                   # L
    pH = 7.00; pCO2 = 45.0
    Tsp_now = 36.8; T_true = 36.8; alk_pool = 0.0
    DO_dev = 0.0; ou_mu = 0.0; ou_o2 = 0.0
    base_on = False; co2_on = False
    base_cum = 0.0; feed_cum = 0.0
    O2_prev = 0.0

    # ---- anomaly definitions ------------------------------------------------------
    an = {}
    if "ph_probe_drift" in anomalies:
        an["ph_drift_start"] = 150.0; an["ph_drift_rate"] = 0.0009      # pH units/h
    if "temp_excursion" in anomalies:
        an["tex_t0"] = 200.0; an["tex_dur"] = 6.0; an["tex_dT"] = 3.0
    if "logger_gap" in anomalies:
        an["gap"] = (rng.uniform(150, 250), 3.5)
    if "capacitance_dropout" in anomalies:
        an["cap_gap"] = (rng.uniform(100, 200), 6.0)
    missed_feed_day = 7 if "missed_feed" in anomalies else None

    # ---- storage ------------------------------------------------------------------
    cols = ["time_h", "pH", "DO_pct", "temp_C", "agitation_rpm", "air_flow_mLmin", "O2_flow_mLmin",
            "CO2_flow_mLmin", "base_cum_mL", "feed_cum_mL", "weight_kg", "capacitance_pF_cm"]
    online = np.full((n_log, len(cols)), np.nan)
    truth_samples = []

    def cap_signal(Xv_, d_):
        return 0.6 + 5.8 * Xv_ * (d_ / 16.3) ** 3

    def diameter(t_, T_, Osm_):
        frac = np.clip((36.8 - T_) / 3.8, 0, 1.3)
        return 16.3 + 0.7 * frac - 0.012 * t_ / 24 + 0.003 * (Osm_ - 320)

    sample_steps = {int(round(24 * k / DT)): k for k in range(0, 15)}
    feed_steps = {int(round((24 * k + 0.5) / DT)): k for k in range(3, 14)}

    for i in range(n):
        t = i * DT
        # ------------- discrete events ------------------------------------------------
        if i in sample_steps:
            d = diameter(t, T_true, Osm)
            truth_samples.append(dict(day=sample_steps[i], time_h=t, Xv=Xv, Xd=Xd, d=d, Glc=Glc, Lac=Lac,
                                      Gln=Gln, Glu=Glu, Amm=Amm, Osm=Osm, pH=pH, pCO2=pCO2,
                                      Ti=Ti, DO=40.0 + DO_dev))
            V -= 0.006                                            # 6 mL sample volume
        if i in feed_steps and feed_steps[i] != missed_feed_day:
            Vf = C["feed_frac"] * V
            Cf = dict(Glc=50.0, Gln=25.0, Glu=10.0, Lac=0.0, Amm=0.0, Osm=700.0)
            w = Vf / (V + Vf)
            Glc = Glc * (1 - w) + Cf["Glc"] * w; Gln = Gln * (1 - w) + Cf["Gln"] * w
            Glu = Glu * (1 - w) + Cf["Glu"] * w; Lac *= (1 - w); Amm *= (1 - w)
            Osm = Osm * (1 - w) + Cf["Osm"] * w
            Xv *= (1 - w); Xd *= (1 - w); Ti *= (1 - w)
            V += Vf; feed_cum += Vf * 1000
            pH -= 0.02; DO_dev -= rng.uniform(2.5, 5.5)
            if Glc < 4.0:                                         # supplemental glucose (400 g/L stock)
                Vg = V * (5.5 - Glc) / (400 - 5.5); w = Vg / (V + Vg)
                Glc = Glc * (1 - w) + 400 * w; Gln *= (1 - w); Glu *= (1 - w); Lac *= (1 - w)
                Amm *= (1 - w); Osm = Osm * (1 - w) + 2500 * w
                Xv *= (1 - w); Xd *= (1 - w); Ti *= (1 - w)
                V += Vg; feed_cum += Vg * 1000

        # ------------- temperature ----------------------------------------------------
        if t < C["t_shift"]:
            Tsp_now = 36.8
        else:
            Tsp_now = max(C["T_shift"], 36.8 - (t - C["t_shift"]) * (36.8 - C["T_shift"]) / 3.0)
        T_true += (Tsp_now - T_true) * DT / 0.4
        T_kin = T_true
        T_disp = T_true
        if "tex_t0" in an and an["tex_t0"] <= t <= an["tex_t0"] + an["tex_dur"]:
            bump = an["tex_dT"] * np.sin(np.pi * (t - an["tex_t0"]) / an["tex_dur"])
            T_kin = T_true + bump; T_disp = T_kin

        # ------------- kinetics -------------------------------------------------------
        ou_mu += (-ou_mu * DT / 12) + 0.02 * np.sqrt(DT / 12) * rng.standard_normal()
        fT_mu = min(np.exp(0.25 * (T_kin - 36.8)), 1.15)
        f_glc = Glc / (0.3 + Glc)
        nit = Gln + 0.5 * Glu
        f_nit = nit / (0.4 + nit)
        f_lac = 1 / (1 + Lac / 9.0)
        f_amm = 1 / (1 + (Amm / 15.0) ** 2)
        f_osm = 1 / (1 + (max(Osm - 380, 0) / 150.0) ** 2)
        f_dens = max(0.0, 1 - Xv / P["xmax"])
        f_lag = sig((t - 6) / 2.5)
        mu_pot = P["mu_max"] * fT_mu
        mu = mu_pot * f_glc * f_nit * f_lac * f_amm * f_osm * f_dens * f_lag * (1 + ou_mu)
        r = np.clip(mu / (mu_pot + 1e-9), 0, 1)

        fT_low = 0.6 + 0.4 * sig((T_kin - 34.0) / 0.8)              # death modulation with temperature
        kd = P["kd_scale"] * fT_low * (0.0004 + 0.0010 * (Lac / 4.0) ** 2 + 0.0012 * (Amm / 8.0) ** 2
                                        + 0.010 * sig((0.25 - Glc) / 0.05)
                                        + 0.0035 * max(Osm - 450, 0) / 100 + 0.000004 * t
                                        + 0.0025 * sig((Gln + Glu - 0.15) / -0.05))
        if T_kin > 37.5:
            kd += 0.012 * (T_kin - 37.5)
        k_lys = 0.006

        fT_g = np.exp(0.12 * (T_kin - 36.8))
        qGlc = P["qg0"] * fT_g * Glc / (0.4 + Glc) * (0.5 + 0.5 * r)
        ylac = P["ylac"] * (Glc / (2.0 + Glc)) * (0.65 + 0.35 * sig((T_kin - 34.5) / 0.7))
        lac_prod = ylac * qGlc * Xv
        lac_cons = (P["lac_shift"] * 0.011 * Xv * Lac / (2.0 + Lac) * sig((t - C["t_shift"] - 20) / 15)
                    if Glc > 0.2 else 0.0)
        qGln = P["qgln0"] * np.exp(0.10 * (T_kin - 36.8)) * Gln / (0.3 + Gln) * (0.5 + 0.5 * r)
        k_deg = 0.0012
        dGln = -qGln * Xv - k_deg * Gln
        dGlu = 0.5 * qGln * Xv - 0.35 * P["qgln0"] * Xv * Glu / (0.5 + Glu) * (0.5 + 0.5 * r)
        dAmm = 0.75 * qGln * Xv + k_deg * Gln
        qP = P["qP0"] * (1 + 0.22 * np.clip(36.8 - T_kin, 0, 4)) * (0.55 + 0.45 * r) * (0.75 + 0.25 * f_glc)
        dTi = qP * Xv * 1e-3 * (1.0 if Glc > 0.1 else 0.6)
        qO2 = 0.45 * np.exp(0.06 * (T_kin - 36.8)) * (0.6 + 0.4 * r)
        OUR = qO2 * Xv                                              # mmol/L/h

        # ------------- pH / gas control --------------------------------------------
        drift = 0.0
        if "ph_drift_start" in an and t > an["ph_drift_start"]:
            drift = an["ph_drift_rate"] * (t - an["ph_drift_start"])
        pH_meas_ctrl = pH + drift + 0.003 * rng.standard_normal()
        if base_on and pH_meas_ctrl >= 6.99:
            base_on = False
        elif (not base_on) and pH_meas_ctrl < 6.95:
            base_on = True
        if co2_on and pH_meas_ctrl <= 7.01:
            co2_on = False
        elif (not co2_on) and pH_meas_ctrl > 7.03:
            co2_on = True
        Fb = 4.0e-3 if base_on else 0.0                            # L/h of 1 M Na2CO3 (2 eq/L)
        acid_load = 11.1 * (lac_prod - lac_cons) + 0.008 * Xv - 1.6 * dAmm
        alk_pool += (Fb * 2000.0 / V - alk_pool) * DT / 0.15        # mixing delay -> overshoot
        alk_in = alk_pool                                           # mmol/L/h
        dpH = (-acid_load + alk_in) / 30.0 - (0.06 if co2_on else 0.0)
        pH += dpH * DT
        base_cum += Fb * DT * 1000

        # ------------- pCO2, osmolality ---------------------------------------------
        pCO2 += ((38 + 1.9 * Xv - pCO2) / 3.0 + (12.0 if co2_on else 0.0)) * DT
        dil = Fb / V
        dOsm = (11.1 * lac_prod - 11.1 * lac_cons - 5.55 * qGlc * Xv + 1.0 * dAmm
                + (2000.0 - Osm) * dil)

        # ------------- mass balances (Euler) ----------------------------------------
        Xv += ((mu - kd) * Xv - Xv * dil) * DT
        Xd += (kd * Xv - k_lys * Xd - Xd * dil) * DT
        Glc += (-qGlc * Xv - Glc * dil) * DT
        Lac += (lac_prod - lac_cons - Lac * dil) * DT
        Gln += (dGln - Gln * dil) * DT
        Glu += (dGlu - Glu * dil) * DT
        Amm += (dAmm - Amm * dil) * DT
        Ti += (dTi - Ti * dil) * DT
        Osm += dOsm * DT
        V += Fb * DT - 0.0005 * DT                                  # base in, evaporation out
        Xv = max(Xv, 1e-4); Glc = max(Glc, 0.0); Gln = max(Gln, 0.0); Glu = max(Glu, 0.0)

        # ------------- DO / gas flows -------------------------------------------------
        DO_dev += (-DO_dev * DT / 0.3) + 1.5 * np.sqrt(DT / 0.3) * rng.standard_normal()
        ou_o2 += (-ou_o2 * DT / 0.5) + 0.10 * np.sqrt(DT / 0.5) * rng.standard_normal()
        air = 15.0
        O2_need = OUR * V * 22.4 / 60.0                            # mL/min
        O2_flow = max(0.0, (O2_need - 0.21 * air * 0.5) / 0.5) * np.exp(ou_o2)
        agit = 110.0 if t < 72 else 120.0
        if t < 10 or O2_flow < 0.05:
            O2_flow = 0.0
        CO2_flow = 10.0 if co2_on else 0.0

        # ------------- logging --------------------------------------------------------
        if i % LOG_EVERY == 0:
            j = i // LOG_EVERY
            d = diameter(t, T_true, Osm)
            online[j] = [
                t,
                pH + drift + 0.004 * rng.standard_normal(),
                40.0 + DO_dev + 0.15 * rng.standard_normal(),
                T_disp + 0.03 * rng.standard_normal() + 0.04 * np.sin(2 * np.pi * t / 0.75),
                agit + 0.3 * rng.standard_normal(),
                air + 0.1 * rng.standard_normal(),
                O2_flow, CO2_flow, base_cum, feed_cum,
                V * 1.005 + 0.002 * rng.standard_normal(),
                cap_signal(Xv, d) + 0.12 * rng.standard_normal() + 0.01 * Xv * rng.standard_normal(),
            ]

    # ---- assemble online dataframe + faults --------------------------------------
    on = pd.DataFrame(online, columns=cols)
    for c in ("pH", "DO_pct", "temp_C", "capacitance_pF_cm"):          # rare spikes
        idx = rng.random(len(on)) < 0.0006
        scale = {"pH": 0.08, "DO_pct": 9.0, "temp_C": 0.6, "capacitance_pF_cm": 4.0}[c]
        on.loc[idx, c] += rng.choice([-1, 1], idx.sum()) * scale
    if "gap" in an:
        g0, gd = an["gap"]
        on.loc[(on.time_h >= g0) & (on.time_h < g0 + gd), cols[1:]] = np.nan
    if "cap_gap" in an:
        g0, gd = an["cap_gap"]
        on.loc[(on.time_h >= g0) & (on.time_h < g0 + gd), "capacitance_pF_cm"] = np.nan
    on.insert(0, "run_id", run_id)
    on.insert(1, "condition", cond_name)
    on.insert(2, "timestamp", START + pd.Timedelta(days=2 * run_index) + pd.to_timedelta(on.time_h, unit="h"))

    # ---- offline analytics with measurement error --------------------------------
    rows = []
    for s in truth_samples:
        def ln(x, cv): return x * rng.lognormal(0, cv)
        row = dict(run_id=run_id, condition=cond_name, day=s["day"], time_h=s["time_h"],
                   VCD_1e6_per_mL=ln(s["Xv"], 0.06),
                   viability_pct=min(100.0, 100 * s["Xv"] / (s["Xv"] + s["Xd"]) + rng.normal(0, 1.2)),
                   cell_diameter_um=s["d"] + rng.normal(0, 0.12),
                   glucose_gL=max(0, ln(s["Glc"], 0.03) + rng.normal(0, 0.04)),
                   lactate_gL=max(0, ln(s["Lac"], 0.03) + rng.normal(0, 0.03)),
                   glutamine_mM=max(0, ln(s["Gln"], 0.04) + rng.normal(0, 0.05)),
                   glutamate_mM=max(0, ln(s["Glu"], 0.04) + rng.normal(0, 0.05)),
                   ammonium_mM=max(0, ln(s["Amm"], 0.05) + rng.normal(0, 0.08)),
                   osmolality_mOsm_kg=s["Osm"] + rng.normal(0, 3.0),
                   pH_offline=s["pH"] + rng.normal(0, 0.02),
                   pCO2_mmHg=max(10, s["pCO2"] + rng.normal(0, 3.0)),
                   pO2_mmHg=1.6 * s["DO"] + rng.normal(0, 4.0),
                   titer_gL=(ln(s["Ti"], 0.05) if s["day"] >= 3 else np.nan))
        for k in ("glucose_gL", "lactate_gL", "glutamine_mM", "glutamate_mM", "ammonium_mM", "titer_gL"):
            if s["day"] > 0 and rng.random() < 0.02:
                row[k] = np.nan                                        # occasional missing analyte
        rows.append(row)
    off = pd.DataFrame(rows)

    meta = dict(run_id=run_id, condition=cond_name, seed=seed, temp_shift_h=(None if C["t_shift"] == NO_SHIFT else C["t_shift"]),
                temp_shift_C=C["T_shift"], feed_frac_per_day=C["feed_frac"],
                inoculation_VCD=round(P["vcd0"], 3), mu_max_per_h=round(P["mu_max"], 4),
                qP_scale=round(P["qP0"], 3), lactate_shift_strength=round(P["lac_shift"], 3),
                anomalies=";".join(anomalies) if anomalies else "none",
                anomaly_details=str({k: (tuple(round(x, 1) for x in v) if isinstance(v, tuple) else v)
                                     for k, v in an.items()} | ({"missed_feed_day": missed_feed_day}
                                                              if missed_feed_day else {})))
    return on, off, meta


ONLINE_ROUNDING = {"pH": 3, "DO_pct": 2, "temp_C": 2, "agitation_rpm": 1, "air_flow_mLmin": 2,
                   "O2_flow_mLmin": 2, "CO2_flow_mLmin": 1, "base_cum_mL": 2, "feed_cum_mL": 2,
                   "weight_kg": 4, "capacitance_pF_cm": 2, "time_h": 2}


def generate(outdir: str | Path, n_runs: int = 48, scenario: str = "base",
             anomaly_rate: float = 0.25, seed: int = 42) -> None:
    """Simulate n_runs runs for one scenario and write the three CSV files to outdir."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}, got {scenario!r}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    design = make_design(n_runs, SCENARIOS[scenario], anomaly_rate, np.random.default_rng(seed))

    ons, offs, metas = [], [], []
    for k, (run_id, cond, faults) in enumerate(design):
        on, off, meta = simulate(run_id, cond, faults, seed + 1 + k, k)
        ons.append(on); offs.append(off); metas.append(meta)

    pd.concat(ons).round(ONLINE_ROUNDING).to_csv(out / "online_timeseries.csv", index=False)
    pd.concat(offs).round(3).to_csv(out / "offline_samples.csv", index=False)
    pd.DataFrame(metas).to_csv(out / "runs_metadata.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="data/raw/cho")
    parser.add_argument("--n-runs", type=int, default=48)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="base")
    parser.add_argument("--anomaly-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.outdir, args.n_runs, args.scenario, args.anomaly_rate, args.seed)
    print(f"{args.n_runs} runs ({args.scenario}) written to {args.outdir}")


if __name__ == "__main__":
    main()
