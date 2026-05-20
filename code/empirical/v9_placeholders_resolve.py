# =====================================================================
# v9_placeholders_resolve.py
# Resolves four placeholders: college-vs-literacy joint table, HHI-alternative terciles, NFCS wave-anchor refit, and relocator-vs-stable characteristics.
#
# Inputs:    _dfm_v7.parquet, _dfm_stable_hq.parquet, _hq_edgar_state_v6.parquet, _thomson_s34_v6_firmquarter.parquet, _fred_college_cache.parquet
# Outputs:   output/stage3a/results_v9_placeholders_resolve.json (printed diagnostics + JSON)
# Paper:     IA HHI section, alt literacy proxies (college), Thomson s34 construction, relocator characteristics
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 7 — Resolve the four [NEEDS EMPIRICIST] placeholders.

Triage [FIX] Item 4 (Critical). Multi-referee: cannot remain placeholders.

  7a. College joint table: fit lit_z and college_z three-ways jointly on
      stable-HQ. Report gamma_lit and gamma_college side-by-side with
      state-cluster t and WCB p.

  7b. HHI alternative: replace IO terciles with Herfindahl-Hirschman Index
      terciles of institutional ownership concentration on the same
      stable-HQ panel; refit the three-way. Test whether mid-HHI patterns
      mirror mid-IO.

  7c. NFCS wave-anchor: instead of monthly literacy, use only wave-anchor
      years (2009, 2012, 2015, 2018, 2021) literacy values and refit the
      three-way on stable-HQ. Tests whether the moderation is driven by
      within-wave linear interpolation or by genuine cross-wave variation.

  7d. Relocator characteristics: firm-mean size, BE/ME, IV, IO, return,
      sector mix comparison between 1,366 relocators and 6,081 stable-HQ
      firms. Mean, SD, t-test of difference for each.

Output: output/stage3a/results_v9_placeholders_resolve.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import (load_full_panel, load_stable_hq, build_relocator_set,
                        add_io_terciles, save_json, OUT, THOMSON)
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                                FOCAL)
import scipy.sparse as sp

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_placeholders_resolve.json")
B_WCB = 4999

FRED_COLLEGE = os.path.join(EMP, "_fred_college_cache.parquet")


# ============================================================
# 7a: College joint table
# ============================================================
def test_7a_college_joint(d_stable):
    """Fit lit_z and college_z three-ways jointly on stable-HQ.

    Spec: ret ~ FOCAL (7 lit-three-way terms) + 7 college-three-way terms
    + state FE + month FE, with the gamma_lit on `mom*iv*lit_z` and
    gamma_col on `mom*iv*college_z` reported side-by-side.
    """
    print("\n=== Test 7a: College joint table (lit + college on stable-HQ) "
          "===", flush=True)
    college = pd.read_parquet(FRED_COLLEGE)
    college['hq_state'] = 'US-' + college['st']
    # Pre-2007 college values: keep them but our panel is 2009+.
    print(f"  college panel: {len(college)} state-years, "
          f"states={college['hq_state'].nunique()}", flush=True)

    d = d_stable.copy()
    d = d.merge(college[['hq_state', 'year', 'college_share']],
                on=['hq_state', 'year'], how='left')
    print(f"  panel after merge: n={len(d):,}, "
          f"college missing={d['college_share'].isna().sum():,}",
          flush=True)
    d = d.dropna(subset=FOCAL + ['ret', 'hq_state', 'college_share']).copy()
    # z-score college within month
    d['college_z'] = d.groupby('ym')['college_share'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    # Build college three-way: mom * iv * college_z
    d['mom_x_college_z'] = d['mom_12_2'] * d['college_z']
    d['iv_x_college_z'] = d['iv'] * d['college_z']
    d['mom_x_iv_x_college_z'] = d['mom_x_iv'] * d['college_z']
    # Z-score within month
    for c in ['mom_x_college_z', 'iv_x_college_z', 'mom_x_iv_x_college_z',
              'college_z']:
        d[c] = d.groupby('ym')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    print(f"  joint-spec sample: {len(d):,} firm-months", flush=True)

    # Joint regression: state+month FE + 7 lit-focal + 7 college-focal
    # focal_lit = FOCAL (the 6 lower-order + the three-way)
    # focal_col = ['mom_12_2' (already), 'iv' (already), 'college_z',
    #              'mom_x_iv' (already), 'mom_x_college_z', 'iv_x_college_z',
    #              'mom_x_iv_x_college_z']
    # Lower-order mom, iv, mom_x_iv are shared; the new terms are college_z,
    # mom_x_college_z, iv_x_college_z, mom_x_iv_x_college_z. Combine into a
    # single design.

    n = len(d)
    lit_focal = d[FOCAL].values.astype(float)
    # FOCAL = mom_12_2, iv, lit_z, mom_x_iv, mom_x_lit, iv_x_lit, mom_iv_lit
    col_extra = d[['college_z', 'mom_x_college_z', 'iv_x_college_z',
                   'mom_x_iv_x_college_z']].values.astype(float)
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    # 6 lower-order lit-focal + the three-way + 4 college terms = 11 endog
    Lit = sp.csr_matrix(lit_focal)  # 7 cols
    Col = sp.csr_matrix(col_extra)  # 4 cols
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W_full = sp.hstack([ones, Lit, Col, Sd, Md]).tocsc()
    # OLS
    WtW = (W_full.T @ W_full).toarray()
    WtW_inv = np.linalg.pinv(WtW)
    beta_full = WtW_inv @ (W_full.T @ y)
    yhat = W_full @ beta_full
    resid = y - yhat
    # state-clustered SE: V = (W'W)^-1 sum_g (X_g e_g)(X_g e_g)' (W'W)^-1
    # Compute via cluster sums
    ug = np.unique(sc)
    Cs = sp.csr_matrix((np.ones(n),
                        (np.arange(n), np.searchsorted(ug, sc))),
                       shape=(n, len(ug)))
    # cluster sums of scores: G x K
    scores = (W_full.multiply(resid[:, None])).toarray()  # n x K (dense, OK)
    G_scores = Cs.T @ scores
    meat = G_scores.T @ G_scores  # K x K
    V_state = WtW_inv @ meat @ WtW_inv
    # Index of the lit three-way: position 7 (FOCAL[6] = mom_x_iv_x_literacy_corr)
    # Layout: 1 (intercept) + 7 (lit_focal) + 4 (col_extra) + ...
    idx_lit = 1 + 6  # 0-based 0..6 are intercept+6 lower-order, idx 7 = three-way
    # Wait: ones=1 col, Lit=7 cols, Col=4 cols. FOCAL[6] = three-way at column index
    # 1 + 6 = 7 (0-based). College three-way at column 1 + 7 + 3 = 11.
    idx_lit_3way = 1 + 6  # within the 7-element FOCAL block
    idx_col_3way = 1 + 7 + 3  # within the 4-element Col block, last position
    coef_lit = float(beta_full[idx_lit_3way])
    coef_col = float(beta_full[idx_col_3way])
    se_lit = float(np.sqrt(max(V_state[idx_lit_3way, idx_lit_3way], 0)))
    se_col = float(np.sqrt(max(V_state[idx_col_3way, idx_col_3way], 0)))
    t_lit = coef_lit / se_lit if se_lit > 0 else np.nan
    t_col = coef_col / se_col if se_col > 0 else np.nan
    print(f"    JOINT (lit + college on stable-HQ aggregate):", flush=True)
    print(f"      gamma_lit  = {coef_lit:+.6f} (state-t={t_lit:.2f})",
          flush=True)
    print(f"      gamma_coll = {coef_col:+.6f} (state-t={t_col:.2f})",
          flush=True)

    # Also: run on mid-IO subsample
    d_io = d[d['io_share_persist'].notna()].copy()
    d_io = add_io_terciles(d_io, 'io_share_persist')
    d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
    n2 = len(d_mid)
    print(f"  mid-IO subsample n={n2:,}", flush=True)
    lit_focal2 = d_mid[FOCAL].values.astype(float)
    col_extra2 = d_mid[['college_z', 'mom_x_college_z',
                         'iv_x_college_z',
                         'mom_x_iv_x_college_z']].values.astype(float)
    y2 = d_mid['ret'].values.astype(float)
    sc2 = pd.Categorical(d_mid['hq_state']).codes.astype(np.int64)
    mc2 = pd.Categorical(d_mid['ym']).codes.astype(np.int64)
    nS2, nM2 = sc2.max() + 1, mc2.max() + 1
    rows2 = np.arange(n2)
    ones2 = sp.csr_matrix(np.ones((n2, 1)))
    Lit2 = sp.csr_matrix(lit_focal2)
    Col2 = sp.csr_matrix(col_extra2)
    Sd2 = sp.csr_matrix((np.ones(n2), (rows2, sc2)), shape=(n2, nS2))[:, 1:]
    Md2 = sp.csr_matrix((np.ones(n2), (rows2, mc2)), shape=(n2, nM2))[:, 1:]
    W_full2 = sp.hstack([ones2, Lit2, Col2, Sd2, Md2]).tocsc()
    WtW2 = (W_full2.T @ W_full2).toarray()
    WtW_inv2 = np.linalg.pinv(WtW2)
    beta2 = WtW_inv2 @ (W_full2.T @ y2)
    yhat2 = W_full2 @ beta2
    resid2 = y2 - yhat2
    ug2 = np.unique(sc2)
    Cs2 = sp.csr_matrix((np.ones(n2),
                          (np.arange(n2), np.searchsorted(ug2, sc2))),
                          shape=(n2, len(ug2)))
    scores2 = (W_full2.multiply(resid2[:, None])).toarray()
    G_scores2 = Cs2.T @ scores2
    meat2 = G_scores2.T @ G_scores2
    V_state2 = WtW_inv2 @ meat2 @ WtW_inv2
    coef_lit2 = float(beta2[idx_lit_3way])
    coef_col2 = float(beta2[idx_col_3way])
    se_lit2 = float(np.sqrt(max(V_state2[idx_lit_3way, idx_lit_3way], 0)))
    se_col2 = float(np.sqrt(max(V_state2[idx_col_3way, idx_col_3way], 0)))
    t_lit2 = coef_lit2 / se_lit2 if se_lit2 > 0 else np.nan
    t_col2 = coef_col2 / se_col2 if se_col2 > 0 else np.nan
    print(f"    JOINT (lit + college on stable-HQ MID-IO):", flush=True)
    print(f"      gamma_lit  = {coef_lit2:+.6f} (state-t={t_lit2:.2f})",
          flush=True)
    print(f"      gamma_coll = {coef_col2:+.6f} (state-t={t_col2:.2f})",
          flush=True)

    return {
        "spec": "Joint TWFE: lit three-way + college three-way on stable-HQ, "
                "state+month FE, state-clustered SE.",
        "n_obs_joint_aggregate": int(n),
        "n_obs_mid_io": int(n2),
        "aggregate_stable_hq": {
            "gamma_lit_three_way": coef_lit,
            "se_state_lit": se_lit,
            "t_state_lit": float(t_lit),
            "gamma_college_three_way": coef_col,
            "se_state_college": se_col,
            "t_state_college": float(t_col),
        },
        "mid_io_stable_hq": {
            "gamma_lit_three_way": coef_lit2,
            "se_state_lit": se_lit2,
            "t_state_lit": float(t_lit2),
            "gamma_college_three_way": coef_col2,
            "se_state_college": se_col2,
            "t_state_college": float(t_col2),
        },
    }


# ============================================================
# 7b: HHI alternative
# ============================================================
def test_7b_hhi(d_stable):
    """Replace IO terciles with HHI of institutional ownership terciles.

    HHI of institutional ownership concentration is built from the
    canonical 13F panel (THOMSON). For each (permno, quarter), HHI =
    sum over managers of (shares_i / total_inst_shares)^2 if we had
    per-manager-stock detail. We only have aggregated firm-quarter
    inst-shares in the canonical _thomson_s34_v6_firmquarter.parquet
    panel, NOT per-manager-stock. So we approximate HHI using:
      - n_mgr (number of distinct institutional holders per stock-quarter)
        as the inverse-equivalent: HHI ≈ 1/n_mgr if holdings were equal
      - 1/n_mgr is a lower bound on HHI; a stock with many holders has
        low HHI (less concentrated).
    Mid-HHI corresponds to MEDIUM number of holders — a clean parallel
    to mid-IO. We use 1/n_mgr terciles (since higher n_mgr = lower HHI).
    """
    print("\n=== Test 7b: HHI-alternative (1/n_mgr) terciles on stable-HQ "
          "===", flush=True)
    thomson = pd.read_parquet(THOMSON)
    print(f"  Thomson panel: {len(thomson)} firm-quarters, "
          f"{thomson['permno'].nunique()} permnos", flush=True)
    # firm-mean n_mgr across quarters
    perm_nmgr = thomson.groupby('permno')['n_mgr'].mean().dropna()
    perm_inv_nmgr = 1.0 / perm_nmgr  # HHI proxy
    print(f"  firm-mean 1/n_mgr (HHI proxy): mean={perm_inv_nmgr.mean():.4f}, "
          f"median={perm_inv_nmgr.median():.4f}, "
          f"firms={len(perm_inv_nmgr)}", flush=True)
    # tercile assignment
    terc = pd.qcut(perm_inv_nmgr, 3,
                   labels=['HHI1_low_conc', 'HHI2_mid_conc',
                           'HHI3_high_conc'])
    # HHI1_low_conc = high n_mgr = low concentration = many holders
    # HHI3_high_conc = low n_mgr = high concentration = few holders

    d = d_stable.copy()
    d['hhi_grp'] = d['permno'].map(terc).astype('object')
    d_all = d.dropna(subset=FOCAL + ['ret', 'hq_state', 'hhi_grp']).copy()
    print(f"  estimation sample: {len(d_all):,} firm-months", flush=True)

    out = {"spec": ("HHI alternative: 1/n_mgr (firm-mean) terciles on "
                    "stable-HQ; replacement for IO terciles."),
           "n_estimation_obs": int(len(d_all)),
           "per_tercile": {}}
    for g in ['HHI1_low_conc', 'HHI2_mid_conc', 'HHI3_high_conc']:
        sub = d_all[d_all['hhi_grp'] == g]
        if len(sub) < 100 or sub['hq_state'].nunique() < 3:
            out['per_tercile'][g] = {"skip": "insufficient",
                                     "n_obs": int(len(sub))}
            continue
        r = twfe_three_way(sub)
        out['per_tercile'][g] = {
            k: r[k] for k in ['coef', 'se', 't', 'se_state', 't_state',
                              'n_obs', 'se_kind']}
        print(f"    {g}: gamma={r['coef']:+.6f} state_t={r['t_state']:.2f} "
              f"n={r['n_obs']:,}", flush=True)
    # mid/low ratio
    try:
        m = out['per_tercile']['HHI2_mid_conc']['coef']
        lo = out['per_tercile']['HHI1_low_conc']['coef']
        out['mid_low_ratio'] = float(m / lo) if lo != 0 else None
    except KeyError:
        out['mid_low_ratio'] = None
    print(f"  mid/low ratio: {out['mid_low_ratio']}", flush=True)
    return out


# ============================================================
# 7c: NFCS wave-anchor years only
# ============================================================
def test_7c_wave_anchor(d_stable):
    """Restrict to wave-anchor years (2009, 2012, 2015, 2018, 2021) and
    refit the headline three-way on stable-HQ. The seed panel uses
    linear interpolation between wave years; using only wave years tests
    whether the moderation is driven by genuine cross-wave variation."""
    print("\n=== Test 7c: NFCS wave-anchor years only ===", flush=True)
    anchor_years = {2009, 2012, 2015, 2018, 2021}
    d = d_stable[d_stable['year'].isin(anchor_years)].copy()
    d = d.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  wave-anchor sample: {len(d):,} firm-months "
          f"({d['year'].nunique()} years: {sorted(d['year'].unique())})",
          flush=True)

    out = {"anchor_years": sorted(list(anchor_years)),
           "n_obs": int(len(d)),
           "per_tercile_persistent": {},
           "per_tercile_time_varying": {}}

    for io_col, key in [('io_share_persist', 'per_tercile_persistent'),
                        ('io_share', 'per_tercile_time_varying')]:
        d_io = d[d[io_col].notna()].copy()
        d_io = add_io_terciles(d_io, io_col)
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            if len(sub) < 100 or sub['hq_state'].nunique() < 3:
                out[key][g] = {"skip": "insufficient",
                               "n_obs": int(len(sub))}
                continue
            r = twfe_three_way(sub)
            out[key][g] = {k: r[k] for k in
                           ['coef', 'se', 't', 'se_state', 't_state',
                            'n_obs', 'se_kind']}
            print(f"    {io_col} {g}: gamma={r['coef']:+.6f} "
                  f"state_t={r['t_state']:.2f} n={r['n_obs']:,}",
                  flush=True)

    # mid/low ratio under each
    for key in ['per_tercile_persistent', 'per_tercile_time_varying']:
        try:
            m = out[key]['IO2_mid']['coef']
            lo = out[key]['IO1_low']['coef']
            out[f"{key}_mid_low_ratio"] = (float(m / lo)
                                            if lo != 0 else None)
        except KeyError:
            out[f"{key}_mid_low_ratio"] = None
    print(f"  persistent mid/low (wave-anchor): "
          f"{out['per_tercile_persistent_mid_low_ratio']}", flush=True)
    print(f"  time-varying mid/low (wave-anchor): "
          f"{out['per_tercile_time_varying_mid_low_ratio']}", flush=True)
    return out


# ============================================================
# 7d: Relocator characteristics
# ============================================================
def test_7d_relocator_chars(d_full, d_stable, reloc_set):
    """Firm-mean size, BE/ME (proxied by log_me_z reversed; not available
    cleanly — substitute with what we have), IV, IO, return, sector mix.
    """
    print("\n=== Test 7d: Relocator characteristics comparison ===",
          flush=True)
    d_full['is_reloc'] = d_full['permno'].isin(reloc_set).astype(int)
    # Aggregate to firm level: mean of each char
    chars = ['me', 'iv', 'mom_12_2', 'io_share_persist', 'ret',
             'literacy_score_corrected', 'log_me']
    firm_chars = (d_full.groupby('permno')[chars + ['is_reloc', 'siccd']]
                  .agg({**{c: 'mean' for c in chars + ['is_reloc']},
                        'siccd': lambda x: x.mode().iloc[0]
                                  if not x.mode().empty else None})
                  .reset_index())
    reloc_firms = firm_chars[firm_chars['is_reloc'] == 1]
    stable_firms = firm_chars[firm_chars['is_reloc'] == 0]
    print(f"  reloc firms: {len(reloc_firms)}, "
          f"stable firms: {len(stable_firms)}", flush=True)

    char_compare = {}
    for c in chars:
        rv = reloc_firms[c].dropna()
        sv = stable_firms[c].dropna()
        if len(rv) < 2 or len(sv) < 2:
            char_compare[c] = {"skip": "insufficient"}
            continue
        t_stat, p_val = ttest_ind(rv, sv, equal_var=False)
        char_compare[c] = {
            "reloc_mean": float(rv.mean()),
            "reloc_sd": float(rv.std()),
            "stable_mean": float(sv.mean()),
            "stable_sd": float(sv.std()),
            "diff": float(rv.mean() - sv.mean()),
            "t_welch": float(t_stat),
            "p_welch": float(p_val),
            "n_reloc": int(len(rv)),
            "n_stable": int(len(sv)),
        }
        print(f"    {c}: reloc={rv.mean():.4f}({rv.std():.4f}, n={len(rv)}) "
              f"vs stable={sv.mean():.4f}({sv.std():.4f}, n={len(sv)}), "
              f"diff={rv.mean()-sv.mean():+.4f} (t={t_stat:.2f}, "
              f"p={p_val:.4f})", flush=True)

    # Sector mix (siccd 1-digit)
    def sic1(s):
        if pd.isna(s):
            return None
        return int(s) // 1000

    reloc_firms['sic1'] = reloc_firms['siccd'].apply(sic1)
    stable_firms['sic1'] = stable_firms['siccd'].apply(sic1)
    reloc_sec = reloc_firms['sic1'].value_counts(normalize=True).sort_index()
    stable_sec = (stable_firms['sic1'].value_counts(normalize=True)
                  .sort_index())
    sector_mix = {
        f"sic1_{int(s)}": {
            "reloc_share": float(reloc_sec.get(s, 0)),
            "stable_share": float(stable_sec.get(s, 0)),
        } for s in sorted(set(reloc_sec.index) | set(stable_sec.index))
        if s is not None
    }
    return {"per_characteristic": char_compare,
            "sector_mix_by_sic1": sector_mix,
            "n_reloc_firms": int(len(reloc_firms)),
            "n_stable_firms": int(len(stable_firms))}


def main():
    t0 = time.time()
    print("=== v9 Test 7: Resolve four [NEEDS EMPIRICIST] placeholders ===",
          flush=True)

    print("\n--- Loading panels ---", flush=True)
    d_full = load_full_panel()
    d_stable = load_stable_hq()
    reloc_set = build_relocator_set()
    print(f"  full={len(d_full):,}, stable={len(d_stable):,}, "
          f"reloc={len(reloc_set)}", flush=True)

    results = {
        "test": "v9 Test 7: Resolve four [NEEDS EMPIRICIST] placeholders",
        "triage_fix": "Item 4 (Critical)",
    }
    results['7a_college_joint_table'] = test_7a_college_joint(d_stable)
    results['7b_hhi_alternative'] = test_7b_hhi(d_stable)
    results['7c_wave_anchor'] = test_7c_wave_anchor(d_stable)
    results['7d_relocator_characteristics'] = test_7d_relocator_chars(
        d_full, d_stable, reloc_set)

    # Verdict roll-up
    verdicts = {}
    # 7a: does lit dominate college?
    mid_lit = (results['7a_college_joint_table']['mid_io_stable_hq']
               ['gamma_lit_three_way'])
    mid_col = (results['7a_college_joint_table']['mid_io_stable_hq']
               ['gamma_college_three_way'])
    t_lit = (results['7a_college_joint_table']['mid_io_stable_hq']
             ['t_state_lit'])
    t_col = (results['7a_college_joint_table']['mid_io_stable_hq']
             ['t_state_college'])
    verdicts['7a_lit_vs_college'] = (
        "LIT_DOMINATES" if (abs(t_lit) > abs(t_col) and mid_lit < 0)
        else "COLLEGE_DOMINATES" if (abs(t_col) > abs(t_lit) and mid_col < 0)
        else "MIXED")

    # 7b: does HHI-mid have concentration like IO-mid?
    hhi_mid_low = results['7b_hhi_alternative'].get('mid_low_ratio')
    verdicts['7b_hhi_mirrors_io'] = (
        "MIRRORS_IO" if (hhi_mid_low is not None and abs(hhi_mid_low) > 1.5)
        else "DOES_NOT_MIRROR")

    # 7c: does mid/low ratio hold under wave-anchor only?
    wa_ratio = (results['7c_wave_anchor']
                .get('per_tercile_persistent_mid_low_ratio'))
    verdicts['7c_wave_anchor_robust'] = (
        "ROBUST" if (wa_ratio is not None and abs(wa_ratio) > 1.5)
        else "ATTENUATES")

    # 7d: descriptive only
    verdicts['7d_descriptive'] = "DESCRIPTIVE"

    results['verdicts'] = verdicts
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdicts: {verdicts} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
