# =====================================================================
# v7_twfe_eiv_lag36.py
# TWFE version of the lag-36-month-literacy errors-in-variables correction:
# reduced-form TWFE and IV-TWFE (lag-36 instruments for the literacy interactions),
# parallel to the headline with state-clustered SE and wild-cluster bootstrap.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_twfe_eiv.json (no tables written)
# Paper:     Internet Appendix classical-EIV section + Main Table T9 tab:fs_iv
#            region (first-stage / IV diagnostics)
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Q3 (referee structured Q3; triager row 27): TWFE version of the
lag-36-month-literacy EIV instrument correction. IA Table B currently reports
the Fama-MacBeth version (t=-0.23); the referee asks for the TWFE version
parallel to the headline.

PROCEDURE
---------
Two specifications:
  (A) REDUCED-FORM TWFE: substitute literacy_z with literacy_z(t-36). The
      three-way is mom * IV * literacy_z(t-36), and similarly all lower-
      order interactions. This is the natural reduced-form TWFE analogue
      of the FM EIV-instrumented regression: under classical EIV, lit_z(t-36)
      is correlated with the true literacy signal and uncorrelated with the
      current-month measurement error in lit_z. The TWFE point estimate is
      the reduced-form, which is informative about the sign and magnitude of
      the underlying relationship.

  (B) IV-TWFE: instrument current literacy_z (and its 3 interaction terms)
      with literacy_z(t-36) (and its lagged interaction terms). Stage 1
      fits each of {literacy_z, mom*literacy_z, iv*literacy_z, three_way} on
      its lag-36 counterpart plus state+month FE. Stage 2 fits returns on the
      fitted three-way (+ 3 fitted lower-order) and the 3 non-instrumented
      lower-order (mom, iv, mom*iv) + state+month FE. This is the literal
      IV-TWFE.

The TWFE version is parallel to the headline (state+month FE, state-clustered
CR1, state-level wild-cluster bootstrap on the three-way coefficient).

Output: results_v7_twfe_eiv.json
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

from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               FOCAL, _cluster_matrix)

np.random.seed(42)
DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_twfe_eiv.json")

B_WCB = 4999


def build_lag36_literacy(dfm):
    """Build literacy_z_lag36 = literacy_z at firm-month (t-36). Within firm,
    shift by 36 months. Returns dfm with the new column."""
    dfm = dfm.sort_values(['permno', 'date']).copy()
    # group by permno, take literacy_z at t-36 (36-month-lag)
    dfm['literacy_score_lag36'] = (
        dfm.groupby('permno')['literacy_score_corrected'].shift(36))
    # also rebuild the lagged interactions
    dfm['mom_x_lit_lag36'] = (
        dfm['mom_12_2'] * dfm['literacy_score_lag36'])
    dfm['iv_x_lit_lag36'] = (
        dfm['iv'] * dfm['literacy_score_lag36'])
    dfm['mom_x_iv_x_lit_lag36'] = (
        dfm['mom_x_iv'] * dfm['literacy_score_lag36'])
    return dfm


def reduced_form_twfe(d):
    """Reduced-form TWFE: substitute lit_z with lit_z(t-36)."""
    # Build FOCAL_lag36
    d = d.copy()
    d['_focal_lit'] = d['literacy_score_lag36']
    d['_focal_mxl'] = d['mom_x_lit_lag36']
    d['_focal_ixl'] = d['iv_x_lit_lag36']
    d['_focal_mxixl'] = d['mom_x_iv_x_lit_lag36']
    rename_map = {
        'literacy_score_corrected': '_focal_lit',
        'mom_x_literacy_corr': '_focal_mxl',
        'iv_x_literacy_corr': '_focal_ixl',
        'mom_x_iv_x_literacy_corr': '_focal_mxixl',
    }
    d_view = d.copy()
    d_view['literacy_score_corrected'] = d['_focal_lit']
    d_view['mom_x_literacy_corr'] = d['_focal_mxl']
    d_view['iv_x_literacy_corr'] = d['_focal_ixl']
    d_view['mom_x_iv_x_literacy_corr'] = d['_focal_mxixl']
    r = twfe_three_way(d_view)
    wcb = wild_cluster_bootstrap_state(d_view, B=B_WCB, seed=42)
    return r, wcb


def iv_twfe(d):
    """IV-TWFE: instrument {lit, m*lit, i*lit, mxi*lit} with their lag-36
    counterparts. Stage 1: project each instrumented var on its lag plus
    state+month FE + (mom, iv, mom_x_iv). Stage 2: fit ret on fitted four
    + (mom, iv, mom_x_iv) + state+month FE. Three-way coef is the IV
    estimate."""
    n = len(d)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))

    # Non-instrumented exogenous: mom, iv, mom_x_iv
    exog = d[['mom_12_2', 'iv', 'mom_x_iv']].values.astype(float)
    EXOG = sp.csr_matrix(exog)

    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]

    # Instruments: lit_lag36, mom_x_lit_lag36, iv_x_lit_lag36, mxixl_lag36
    z = d[['literacy_score_lag36', 'mom_x_lit_lag36',
           'iv_x_lit_lag36', 'mom_x_iv_x_lit_lag36']].values.astype(float)
    Z = sp.csr_matrix(z)

    # Stage 1 controls: intercept + EXOG + state FE + month FE + instruments
    W1 = sp.hstack([ones, EXOG, Sd, Md, Z]).tocsc()
    W1tW1 = (W1.T @ W1).toarray()
    W1tW1_inv = np.linalg.inv(W1tW1)

    # We want to project each of the 4 endog vars
    endog = d[['literacy_score_corrected', 'mom_x_literacy_corr',
               'iv_x_literacy_corr',
               'mom_x_iv_x_literacy_corr']].values.astype(float)
    # Fitted via OLS of each endog col on W1 - but W1 INCLUDES Z; we want the
    # fitted values to be the projection on (intercept + EXOG + FE + Z),
    # which IS W1. So:
    fitted = W1 @ (W1tW1_inv @ (W1.T @ endog))
    # Stage 2: y on (fitted endog + EXOG + FE), no intercept overlap
    # We use FWL: partial out (intercept + EXOG + FE) from y and from fitted
    W2 = sp.hstack([ones, EXOG, Sd, Md]).tocsc()
    W2tW2 = (W2.T @ W2).toarray()
    W2tW2_inv = np.linalg.inv(W2tW2)

    def partial2(v):
        return v - W2 @ (W2tW2_inv @ (W2.T @ v))

    y = d['ret'].values.astype(float)
    yt = partial2(y)
    fitted_t = partial2(fitted)   # n x 4
    XtX = fitted_t.T @ fitted_t
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (fitted_t.T @ yt)
    # Three-way coef is beta[3]
    e = yt - fitted_t @ beta
    # IV-state-clustered SE
    Cs, states = _cluster_matrix(sc)
    Cm, _ = _cluster_matrix(mc)
    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    Ci, _ = _cluster_matrix(inter)
    scores = fitted_t * e[:, None]   # n x 4

    def meat(C):
        S = np.asarray(C.T @ scores)
        return S.T @ S
    V_state = XtX_inv @ meat(Cs) @ XtX_inv
    V_cgm = XtX_inv @ (meat(Cs) + meat(Cm) - meat(Ci)) @ XtX_inv

    three_coef = float(beta[3])
    se_state = float(np.sqrt(V_state[3, 3]))
    cgm_psd = bool(V_cgm[3, 3] > 0)
    se_cgm = float(np.sqrt(V_cgm[3, 3])) if cgm_psd else None
    t_state = three_coef / se_state if se_state > 0 else np.nan

    # IV first-stage F (joint F on the four lag-36 instruments in stage 1
    # regression of the THREE-WAY endog on FE+EXOG+instruments)
    # Stage-1 reg of (mom_x_iv_x_literacy_corr) on W_minus_Z + Z
    target = d['mom_x_iv_x_literacy_corr'].values.astype(float)
    Wnoz = sp.hstack([ones, EXOG, Sd, Md]).tocsc()
    WnozTWnoz = (Wnoz.T @ Wnoz).toarray()
    Wnoz_inv = np.linalg.inv(WnozTWnoz)

    def partial_noZ(v):
        return v - Wnoz @ (Wnoz_inv @ (Wnoz.T @ v))

    Z_t = np.column_stack([partial_noZ(z[:, k]) for k in range(4)])
    target_t = partial_noZ(target)
    Z_tZ_t = Z_t.T @ Z_t
    try:
        pi = np.linalg.solve(Z_tZ_t, Z_t.T @ target_t)
        u = target_t - Z_t @ pi
        rss = float((u ** 2).sum())
        ess = float((target_t ** 2).sum()) - rss
        df1 = 4
        df2 = n - 4 - Wnoz.shape[1]
        F = (ess / df1) / (rss / df2) if rss > 0 else np.nan
    except np.linalg.LinAlgError:
        F = np.nan

    return {
        'gamma_three_way_iv': three_coef,
        'se_state_clustered_CR1': se_state,
        't_state_clustered_CR1': float(t_state),
        'se_cgm_two_way': se_cgm,
        't_cgm_two_way': three_coef / se_cgm if se_cgm else None,
        'first_stage_F_jointly_4_instruments': float(F),
        'first_stage_F_threshold_warning': bool(F < 10),
        'n_obs': int(n),
        'n_state_clusters': int(len(states)),
        'spec': 'IV-TWFE: instrument {lit_z, mom*lit, iv*lit, mom*iv*lit} '
                'with their lag-36 counterparts; stage-2 ret ~ fitted four + '
                'mom + iv + mom_x_iv + state + month FE.',
    }


def main():
    print("=== Q3 TWFE EIV lag-36 (v7) ===", flush=True)
    dfm = pd.read_parquet(DFM_CACHE)
    dfm = build_lag36_literacy(dfm)
    print(f"merged panel after lag-36 build: {len(dfm):,} firm-months",
          flush=True)
    valid = dfm.dropna(subset=FOCAL + ['ret', 'hq_state',
                                       'literacy_score_lag36',
                                       'mom_x_lit_lag36',
                                       'iv_x_lit_lag36',
                                       'mom_x_iv_x_lit_lag36']).copy()
    print(f"sample with non-null lag-36 literacy: {len(valid):,}",
          flush=True)

    # ----- (A) reduced-form TWFE -----
    print("\n--- (A) reduced-form TWFE (lag-36 literacy directly) ---",
          flush=True)
    r_rf, wcb_rf = reduced_form_twfe(valid)
    print(f"  gamma_RF = {r_rf['coef']:+.6f} | CGM t={r_rf['t']:.2f} "
          f"({r_rf['se_kind']}) | state-cl t={r_rf['t_state']:.2f} | "
          f"wcb p={wcb_rf['p_value']:.4f}", flush=True)

    # ----- (B) IV-TWFE -----
    print("\n--- (B) IV-TWFE (lag-36 instruments for endog literacy "
          "interactions) ---", flush=True)
    r_iv = iv_twfe(valid)
    print(f"  gamma_IV = {r_iv['gamma_three_way_iv']:+.6f} | "
          f"state-cl t={r_iv['t_state_clustered_CR1']:.2f} | "
          f"CGM t={r_iv['t_cgm_two_way']} | "
          f"first-stage F (4 inst) = "
          f"{r_iv['first_stage_F_jointly_4_instruments']:.1f}",
          flush=True)

    results = {
        'task': 'Q3 (referee structured Q3; triager row 27): TWFE version '
                'of the lag-36 EIV instrument. IA Table B has FM version; '
                'this provides the TWFE analogue parallel to the headline.',
        'reference': {
            'headline_twfe_full_panel': {'gamma': -0.011727, 't_state': -2.17,
                                          'wcb_p': 0.0775},
            'ia_table_B_fm_eiv_lag36': {'gamma': None, 't': -0.23,
                                         'note': 'FM Newey-West'},
        },
        'sample': {
            'panel_firm_months': int(len(dfm)),
            'valid_with_lag36_firm_months': int(len(valid)),
            'valid_pct': round(100.0 * len(valid) / len(dfm), 1),
        },
        'reduced_form_twfe_lag36': {
            'gamma': r_rf['coef'],
            'se_cgm_two_way': r_rf['se'] if r_rf['se_kind'] == 'cgm_two_way'
            else None,
            't_cgm_two_way': r_rf['t']
            if r_rf['se_kind'] == 'cgm_two_way' else None,
            'se_state_clustered_CR1': r_rf['se_state'],
            't_state_clustered_CR1': r_rf['t_state'],
            'wcb_p_value': wcb_rf['p_value'],
            'wcb_B': wcb_rf['B'],
            'wcb_ci_studentized': wcb_rf['ci_studentized'],
            'n_obs': int(r_rf['n_obs']),
            'spec': 'TWFE with literacy_z(t-36) and all literacy interactions'
                    ' built on lag-36 literacy; same FE + clustering as '
                    'headline.',
        },
        'iv_twfe_lag36': r_iv,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
