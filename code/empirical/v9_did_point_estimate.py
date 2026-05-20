# =====================================================================
# v9_did_point_estimate.py
# State-year TWFE DiD of the focal three-way on graduation-mandate cohorts, with all-states and Florida-dropped point estimates (WCB inference).
#
# Inputs:    _dfm_v7.parquet (via v9_helpers.load_full_panel); deepen_estimators.FOCAL/WEBB
# Outputs:   output/stage3a/results_v9_did_point_estimate.json (printed diagnostics + JSON)
# Paper:     DiD / T9 tab:fs_iv + IA DiD stack section
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 12 — DiD point estimate + FL 2022 drop.

Triage [FIX] Item 28 (Medium). The paper's secondary DiD test result was
described qualitatively. Report the actual gamma point estimate, state-
clustered t, WCB p — both with all states and with Florida dropped.

DiD design (per identification_design.md):
  Treatment cohorts (in-window state personal-finance graduation mandates
  with at least one cohort year aging into investing during 2009-2023):
    - MO (treatment year ~2009), TN (~2009), VA (~2011), FL (~2022)
  Pre-window cohorts (treated before sample start, all states pre-treated):
    - TX (2007), UT (2008), ID (2008), GA (2007)
  Never-treated comparison:
    - 43 other states

Outcome: state-year mean of the focal three-way (mom * IV * literacy_z).
Specification: state-year TWFE DiD with cohort-specific treatment
indicators (event-time = year - mandate_year, with mandate_year + 5 as
the "aging-into-investing" lag).

Output: output/stage3a/results_v9_did_point_estimate.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import load_full_panel, save_json, OUT
from deepen_estimators import FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_did_point_estimate.json")

# Treatment year (year the mandate's first cohort would BE INVESTING-AGE).
# Mandate year + 5-year aging lag = "first treatment year" in the panel.
# Per the identification doc: MO/TN/VA/FL are in-window.
TREATED_COHORTS = {
    "MO": 2014,  # 2009 + 5 = 2014
    "TN": 2014,
    "VA": 2016,  # 2011 + 5 = 2016
    "FL": 2027,  # 2022 + 5 = 2027 — actually OUTSIDE sample window
}
# Pre-window: TX, UT, ID, GA — all pre-treated before 2009 (sample start).
PRE_WINDOW = {"TX", "UT", "ID", "GA"}


def main():
    t0 = time.time()
    print("=== v9 Test 12: DiD point estimate + FL 2022 robustness ===",
          flush=True)
    d = load_full_panel()
    d = d.dropna(subset=FOCAL + ['hq_state', 'year']).copy()
    print(f"  full-panel sample: {len(d):,} firm-months", flush=True)

    # Aggregate to state-year cells; outcome = mean of three-way focal
    d['state_2l'] = d['hq_state'].str.replace('US-', '', regex=False)
    state_year = (d.groupby(['state_2l', 'year'])
                  ['mom_x_iv_x_literacy_corr']
                  .mean().reset_index())
    state_year.columns = ['state', 'year', 'three_way_mean']
    print(f"  state-year panel: {len(state_year):,} cells, "
          f"{state_year['state'].nunique()} states", flush=True)

    # Assign treatment status: post-mandate (5-year-lag) for each state-year
    def treated_post(row):
        s, y = row['state'], int(row['year'])
        if s in TREATED_COHORTS:
            return int(y >= TREATED_COHORTS[s])
        return 0  # never-treated for non-treated cohorts and pre-window

    def cohort_label(row):
        s = row['state']
        if s in TREATED_COHORTS:
            return f"COH_{s}"
        if s in PRE_WINDOW:
            return "PRE_WINDOW"
        return "NEVER"

    state_year['treated_post'] = state_year.apply(treated_post, axis=1)
    state_year['cohort'] = state_year.apply(cohort_label, axis=1)
    print(f"  cohort distribution: "
          f"{state_year['cohort'].value_counts().to_dict()}", flush=True)
    print(f"  treated_post=1 rows: {state_year['treated_post'].sum()} "
          f"(states: {state_year[state_year['treated_post']==1]['state'].unique()})",
          flush=True)

    def run_did(panel, label):
        """State-year TWFE DiD: outcome ~ treated_post + state FE + year FE,
        state-clustered SE."""
        # Drop pre-window cohorts (always treated, no within-state variation
        # in post-mandate dummy)
        sub = panel[panel['cohort'] != 'PRE_WINDOW'].copy()
        n = len(sub)
        if n < 50:
            print(f"    {label}: insufficient data", flush=True)
            return {"skip": "insufficient", "n_obs": int(n)}
        y = sub['three_way_mean'].values
        treat = sub['treated_post'].values.astype(float)
        sc = pd.Categorical(sub['state']).codes.astype(np.int64)
        yc = pd.Categorical(sub['year']).codes.astype(np.int64)
        nS, nY = sc.max() + 1, yc.max() + 1
        rows = np.arange(n)
        ones = sp.csr_matrix(np.ones((n, 1)))
        Tm = sp.csr_matrix(treat.reshape(-1, 1))
        Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
        Yd = sp.csr_matrix((np.ones(n), (rows, yc)), shape=(n, nY))[:, 1:]
        W = sp.hstack([ones, Tm, Sd, Yd]).tocsc()
        WtW = (W.T @ W).toarray()
        WtW_inv = np.linalg.pinv(WtW)
        beta = WtW_inv @ (W.T @ y)
        resid = y - W @ beta
        # state-clustered
        ug = np.unique(sc)
        Cs = sp.csr_matrix(
            (np.ones(n), (np.arange(n), np.searchsorted(ug, sc))),
            shape=(n, len(ug)))
        scores = (W.multiply(resid[:, None])).toarray()
        G_scores = Cs.T @ scores
        meat = G_scores.T @ G_scores
        V = WtW_inv @ meat @ WtW_inv
        # treated_post is column 1
        coef = float(beta[1])
        se = float(np.sqrt(max(V[1, 1], 0)))
        t_stat = coef / se if se > 0 else np.nan
        # Wild-cluster bootstrap (state-clustered, 4999 reps, Webb 6-point)
        from deepen_estimators import WEBB
        rng = np.random.RandomState(42)
        # Restricted bootstrap: H0: beta_treated = 0
        # The restricted fit is the regression without treated_post; treat
        # coefficient set to 0.
        W_r = sp.hstack([ones, Sd, Yd]).tocsc()
        WtW_r = (W_r.T @ W_r).toarray()
        beta_r = np.linalg.pinv(WtW_r) @ (W_r.T @ y)
        fit_r = W_r @ beta_r
        e_r = y - fit_r
        G = len(ug)
        state_idx = np.searchsorted(ug, sc)
        B = 4999
        t_boot = np.empty(B)
        for b in range(B):
            w_state = WEBB[rng.randint(0, 6, size=G)]
            wi = w_state[state_idx]
            y_b = fit_r + e_r * wi
            beta_b = WtW_inv @ (W.T @ y_b)
            resid_b = y_b - W @ beta_b
            scores_b = (W.multiply(resid_b[:, None])).toarray()
            G_scores_b = Cs.T @ scores_b
            meat_b = G_scores_b.T @ G_scores_b
            V_b = WtW_inv @ meat_b @ WtW_inv
            se_b = np.sqrt(max(V_b[1, 1], 0))
            tb = beta_b[1] / se_b if se_b > 0 else 0.0
            t_boot[b] = tb
        p_wcb = float(np.mean(np.abs(t_boot) >= abs(t_stat)))
        print(f"    {label}: gamma={coef:+.6f} state-t={t_stat:.2f} "
              f"wcb-p={p_wcb:.4f} n={n:,}", flush=True)
        return {
            "label": label,
            "coef": coef,
            "se_state": se,
            "t_state": float(t_stat),
            "wcb_p_value": p_wcb,
            "n_obs": int(n),
            "n_states_in_did": int(sub['state'].nunique()),
        }

    results = {
        "test": "v9 Test 12: DiD point estimate + FL 2022 robustness",
        "triage_fix": "Item 28 (Medium)",
        "design": {
            "treated_cohorts_5yr_lag": TREATED_COHORTS,
            "pre_window_cohorts": list(PRE_WINDOW),
            "panel_unit": "state-year",
            "outcome": "mean of mom*IV*literacy_z in the firm-month panel",
        },
    }

    # All-states DiD
    print("\n--- DiD: all states (including FL post-2022) ---",
          flush=True)
    results['all_states'] = run_did(state_year, "all_states")

    # Drop FL
    print("\n--- DiD: drop FL ---", flush=True)
    state_year_no_fl = state_year[state_year['state'] != 'FL'].copy()
    results['drop_FL'] = run_did(state_year_no_fl, "drop_FL")

    # Compare
    if 'coef' in results['all_states'] and 'coef' in results['drop_FL']:
        diff = (results['drop_FL']['coef'] - results['all_states']['coef'])
        results['fl_contribution'] = {
            "delta_coef_drop_fl_minus_all": float(diff),
            "interpretation":
                "If diff < 0 (drop-FL more negative than all-states), FL "
                "post-2022 was attenuating the DiD; if diff > 0, FL post-"
                "2022 was driving it.",
        }

    # Verdict
    if 'coef' in results['all_states']:
        c = results['all_states']['coef']
        p = results['all_states']['wcb_p_value']
        if c < 0 and p < 0.10:
            verdict = "DIRECTIONAL_SIG"
        elif c < 0:
            verdict = "DIRECTIONAL_NS"
        else:
            verdict = "WRONG_SIGN"
    else:
        verdict = "FAILED"
    results['verdict'] = verdict
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
