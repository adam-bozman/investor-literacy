# =====================================================================
# v7_precision_fm.py
# Precision-weighted (inverse-variance) Fama-MacBeth on the monthly three-way
# cross-sectional slopes, compared against the pooled TWFE estimate.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_precision_fm.json;
#            output/stage3a/monthly_slopes_v7.csv (no LaTeX tables)
# Paper:     Internet Appendix full inference battery section
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Q2 (referee structured Q2; triager row 26): precision-weighted Fama-
MacBeth.

The paper's Fama-MacBeth coefficient is -0.006 with t=-0.85 (Newey-West);
the pooled TWFE is -0.0117 with t=-2.15 under state clustering. The referee
asks: compute the precision-weighted FM (inverse-variance-weighted mean of
the 180 cross-sectional slopes) — if it matches TWFE, that strengthens the
TWFE reading.

PROCEDURE
---------
For each month m, run the cross-sectional regression
    r_{i,m} = a_m + b_m * mom*IV*literacy_z + (6 lower-order focal as controls)
              + state FE_m + epsilon_{i,m}
Save (b_m, SE_m). The standard FM weights all b_m equally; the precision-
weighted FM weights each b_m by 1/SE_m^2.

Two SE conventions for the cross-sectional regression:
  (i)  OLS heteroskedasticity-robust SE (HC1)
  (ii) state-clustered SE within the month (CR1 within-month)

We report both.

Output: results_v7_precision_fm.json + a new monthly_slopes_v7.csv.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from deepen_estimators import FOCAL

np.random.seed(42)

DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_precision_fm.json")
OUT_SLOPES = os.path.join(ROOT, "output/stage3a/monthly_slopes_v7.csv")


def cross_section_one_month(d_month):
    """Return (b_m, SE_m_HC1, SE_m_state, N, n_states)."""
    n = len(d_month)
    if n < 50 or d_month['hq_state'].nunique() < 5:
        return None
    focal = d_month[FOCAL].values.astype(float)
    lo6 = focal[:, :6]
    three = focal[:, 6]
    y = d_month['ret'].values.astype(float)
    sc = pd.Categorical(d_month['hq_state']).codes.astype(np.int64)
    nS = sc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6 = sp.csr_matrix(lo6)
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    W = sp.hstack([ones, W6, Sd]).tocsc()
    WtW = (W.T @ W).toarray()
    try:
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        return None

    def partial(v):
        return v - W @ (WtW_inv @ (W.T @ v))

    three_t = partial(three)
    yt = partial(y)
    Sxx = float(three_t @ three_t)
    if Sxx <= 0:
        return None
    b = float(three_t @ yt) / Sxx
    e = yt - b * three_t
    score = three_t * e
    # HC1
    meat_hc1 = float((score ** 2).sum())
    se_hc1 = float(np.sqrt(meat_hc1) / Sxx)
    # cluster on hq_state within the month
    df_state = pd.DataFrame({'sc': sc, 'score': score})
    ss_state = (df_state.groupby('sc')['score'].sum() ** 2).sum()
    se_state = float(np.sqrt(ss_state) / Sxx)
    return b, se_hc1, se_state, n, int(d_month['hq_state'].nunique())


def main():
    print("=== Q2 precision-weighted Fama-MacBeth (v7) ===", flush=True)
    dfm = pd.read_parquet(DFM_CACHE)
    dfm = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"estimation sample: {len(dfm):,} firm-months, "
          f"{dfm['ym'].nunique()} months", flush=True)

    rows = []
    months = sorted(dfm['ym'].unique())
    for k, m in enumerate(months):
        d_month = dfm[dfm['ym'] == m]
        res = cross_section_one_month(d_month)
        if res is None:
            continue
        b, se_hc1, se_state, nm, ns = res
        rows.append({'ym': m, 'slope': b, 'se_hc1': se_hc1,
                     'se_state': se_state, 'n': nm, 'n_states': ns})
        if (k + 1) % 30 == 0:
            print(f"  processed {k+1}/{len(months)} months "
                  f"(latest: {m}, b={b:+.5f}, se_state={se_state:.4f})",
                  flush=True)
    slopes = pd.DataFrame(rows)
    slopes.to_csv(OUT_SLOPES, index=False)
    print(f"\nsaved {OUT_SLOPES} ({len(slopes)} months)", flush=True)

    n = len(slopes)
    # standard FM (equal-weighted)
    fm_eq = float(slopes['slope'].mean())
    se_eq_simple = float(slopes['slope'].std(ddof=1) / np.sqrt(n))
    t_eq_simple = fm_eq / se_eq_simple
    # Newey-West with lag 4 (the v3 paper convention)
    nw_lag = 4
    sl = slopes['slope'].values
    s_dev = sl - fm_eq
    var_nw = float((s_dev ** 2).sum()) / n
    for L in range(1, nw_lag + 1):
        w = 1.0 - L / (nw_lag + 1)
        cov = float((s_dev[:-L] * s_dev[L:]).sum()) / n
        var_nw += 2 * w * cov
    se_nw = float(np.sqrt(var_nw / n))
    t_nw = fm_eq / se_nw

    # precision-weighted: w_m = 1 / SE_m^2
    # (use state-clustered SE within month — closest to TWFE convention)
    w_state = 1.0 / (slopes['se_state'] ** 2)
    fm_pw_state = float((slopes['slope'] * w_state).sum() / w_state.sum())
    # SE of precision-weighted estimator under inverse-variance assumption:
    # var(b_pw) = 1 / sum(w_m). But since b_m's are not literally independent
    # across months, also report a Newey-West-corrected SE on the weighted
    # slopes (where the weighted contribution is z_m = w_m * (b_m - fm_pw) /
    # sum(w_m); the t-stat is fm_pw / sqrt(sum(w_m * (b_m-fm_pw)^2) / (n-1) /
    # sum(w_m)^2) is a Hartung-Knapp-style adjustment).
    se_pw_inv = float(np.sqrt(1.0 / w_state.sum()))
    t_pw_inv = fm_pw_state / se_pw_inv
    # Knapp-Hartung adjustment: more conservative when slopes are heterogeneous
    Q = float((w_state * (slopes['slope'] - fm_pw_state) ** 2).sum())
    se_pw_kh = float(np.sqrt(Q / (n - 1) / w_state.sum()))
    t_pw_kh = fm_pw_state / se_pw_kh

    # also: HC1-precision-weighted
    w_hc1 = 1.0 / (slopes['se_hc1'] ** 2)
    fm_pw_hc1 = float((slopes['slope'] * w_hc1).sum() / w_hc1.sum())
    se_pw_inv_hc1 = float(np.sqrt(1.0 / w_hc1.sum()))
    t_pw_inv_hc1 = fm_pw_hc1 / se_pw_inv_hc1
    Q_hc1 = float((w_hc1 * (slopes['slope'] - fm_pw_hc1) ** 2).sum())
    se_pw_kh_hc1 = float(np.sqrt(Q_hc1 / (n - 1) / w_hc1.sum()))
    t_pw_kh_hc1 = fm_pw_hc1 / se_pw_kh_hc1

    print(f"\n--- equal-weighted FM ---", flush=True)
    print(f"  mean slope = {fm_eq:+.6f}, n_months = {n}", flush=True)
    print(f"  simple SE = {se_eq_simple:.6f}, t = {t_eq_simple:.3f}",
          flush=True)
    print(f"  Newey-West lag4 SE = {se_nw:.6f}, t = {t_nw:.3f}",
          flush=True)

    print(f"\n--- precision-weighted FM (state-clustered SE^-2 weights) ---",
          flush=True)
    print(f"  weighted mean slope = {fm_pw_state:+.6f}", flush=True)
    print(f"  inverse-variance SE = {se_pw_inv:.6f}, t = {t_pw_inv:.3f}",
          flush=True)
    print(f"  Knapp-Hartung SE     = {se_pw_kh:.6f}, t = {t_pw_kh:.3f}",
          flush=True)

    print(f"\n--- precision-weighted FM (HC1 SE^-2 weights) ---", flush=True)
    print(f"  weighted mean slope = {fm_pw_hc1:+.6f}", flush=True)
    print(f"  inverse-variance SE = {se_pw_inv_hc1:.6f}, "
          f"t = {t_pw_inv_hc1:.3f}", flush=True)
    print(f"  Knapp-Hartung SE     = {se_pw_kh_hc1:.6f}, "
          f"t = {t_pw_kh_hc1:.3f}", flush=True)

    results = {
        'task': 'Q2 (referee structured Q2; triager row 26): precision-'
                'weighted Fama-MacBeth. Inverse-variance-weighted mean of '
                'the cross-sectional slopes. Tests whether precision '
                'weighting brings FM into line with TWFE.',
        'reference_twfe': {'gamma': -0.011727, 't_state_cl': -2.17,
                           'wcb_p': 0.0775},
        'n_months': n,
        'equal_weighted': {
            'mean_slope': fm_eq,
            'se_simple': se_eq_simple,
            't_simple': t_eq_simple,
            'se_newey_west_lag4': se_nw,
            't_newey_west_lag4': t_nw,
        },
        'precision_weighted_state_cluster_within_month': {
            'weighted_mean_slope': fm_pw_state,
            'se_inverse_variance': se_pw_inv,
            't_inverse_variance': t_pw_inv,
            'se_knapp_hartung': se_pw_kh,
            't_knapp_hartung': t_pw_kh,
        },
        'precision_weighted_hc1': {
            'weighted_mean_slope': fm_pw_hc1,
            'se_inverse_variance': se_pw_inv_hc1,
            't_inverse_variance': t_pw_inv_hc1,
            'se_knapp_hartung': se_pw_kh_hc1,
            't_knapp_hartung': t_pw_kh_hc1,
        },
        'convergence_with_twfe': {
            'twfe_gamma': -0.011727,
            'pw_state_gamma': fm_pw_state,
            'pw_hc1_gamma': fm_pw_hc1,
            'pw_state_vs_twfe_gap_pct': round(
                100.0 * (fm_pw_state - (-0.011727)) / abs(-0.011727), 1),
            'pw_hc1_vs_twfe_gap_pct': round(
                100.0 * (fm_pw_hc1 - (-0.011727)) / abs(-0.011727), 1),
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
