# =====================================================================
# v7_refined_reloc_and_iv_decomposition.py
# Re-runs the IO sign-flip on a refined (between-state-only) stable-HQ subsample
# (Task 1) and decomposes the within-retail wrong-sign literacy gradient by IV
# decile on the broad stable-HQ sample (Task 2).
#
# Inputs:    _dfm_pit_hq_v7_centerpiece.parquet (cached PIT firm-month panel);
#            _hq_edgar_state_v6.parquet (SEC EDGAR 10-K-header HQ history);
#            _dfm_stable_hq.parquet (broad stable-HQ cache from the centerpiece).
# Outputs:   output/stage3a/results_v7_refined_reloc.json;
#            output/stage3a/results_v7_within_retail_iv_decomposition.json (no tables)
# Paper:     Internet Appendix refined-relocator section + within-retail IV-decile
#            section
# Run order: see code/00_master.py
# =====================================================================

"""v7 refined-relocator + IV-decile decomposition (operator-mandated diagnostic).

CONTEXT:
  v7 stable-HQ centerpiece (broad relocator definition: reloc = hdr_changed
  OR sec_disagree -> 1,366 firms dropped) collapsed the IO sign flip
  (persistent DIFF=-0.0079 wcb p=0.650; time-varying -0.0058 wcb p=0.755) and
  surfaced a wrong-sign within-retail literacy gradient (persistent DIFF
  low-lit minus high-lit = +0.0033 wcb p=0.024).

  The operator's hypothesis: the v6 broad relocator flag is too inclusive
  because flag_sec_disagree compares the LAST 10-K header to the static
  Compustat snapshot, dropping 524 firms whose 10-K headers were internally
  consistent (no between-state header change in-sample) but disagree with the
  static Compustat HQ field — these firms may not actually be in-sample
  relocators in any meaningful sense. A refined "between-state-only"
  definition restricts the relocator flag to firms whose 10-K-header STATE
  field actually changed between two distinct US-state codes during the
  in-sample filings.

  EMPIRICAL CHECK on the cached EDGAR file _hq_edgar_state_v6.parquet
  (tmp_inspect2.py): the `hdr_changed` boolean column equals exactly the
  between-state flag (n_distinct_states >= 2). Zero firms are flagged
  hdr_changed=True with only one distinct state. The broadness in
  flag_sec_disagree therefore comes ENTIRELY from cross-source (10-K-header
  vs Compustat static) disagreements, not from within-state header tweaks.

  REFINED relocator (this script) = hdr_between_state flag alone
    = (n_10k >= 2) AND (n_distinct_states_in_10K_headers >= 2)
    = 954 firms (vs v6 broad 1,366).
  We drop the 524 sec_disagree-only firms because their 10-K headers were
  internally stable in-sample (they did not relocate during the sample
  window per the EDGAR record).

TASK 1: Re-run the IO sign flip on the refined stable-HQ subsample
  (drop only the 954 refined relocators).

  DECISION RULE (operator-mandated, applied literally):
    RECOVERY  <=>  |DIFF| > 0.015  AND  wcb p < 0.30  (for either measure)
    Else NO RECOVERY.

TASK 2: IV-decile decomposition of the within-retail wrong-sign on the
  BROAD stable-HQ sample (the v7_stable_hq_centerpiece sample, 1,366
  dropped). Within low-IO, compute (low-lit minus high-lit) three-way DIFF
  in each IV decile separately, to locate the wrong-sign in IV-space.

Output:
  output/stage3a/results_v7_refined_reloc.json
  output/stage3a/results_v7_within_retail_iv_decomposition.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)

from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               stacked_io_difference,
                               stacked_state_group_difference, FOCAL)
from deepen_thomson_io_panel_v6 import LAG_MONTHS

np.random.seed(42)

HQ_EDGAR = os.path.join(EMP, "_hq_edgar_state_v6.parquet")
DFM_PIT_CACHE = os.path.join(EMP, "_dfm_pit_hq_v7_centerpiece.parquet")
DFM_STABLE_BROAD_CACHE = os.path.join(EMP, "_dfm_stable_hq.parquet")  # v7 broad
OUT_JSON_T1 = os.path.join(ROOT, "output", "stage3a",
                           "results_v7_refined_reloc.json")
OUT_JSON_T2 = os.path.join(ROOT, "output", "stage3a",
                           "results_v7_within_retail_iv_decomposition.json")

B_WCB = 4999


def states_distinct(s):
    if not isinstance(s, str):
        return 0
    try:
        seq = json.loads(s)
    except Exception:
        return 0
    return len({rec[1] for rec in seq if rec and rec[1]})


def last_hdr_state(s):
    if not isinstance(s, str):
        return None
    try:
        seq = json.loads(s)
    except Exception:
        return None
    return seq[-1][1] if seq else None


def build_flags(hq, panel_hq):
    """Compute both the v6 BROAD and the refined BETWEEN-STATE-ONLY flags
    and return both relocator permno sets + diagnostics."""
    panel_hq = panel_hq.copy()
    panel_hq['hq_state_2l'] = panel_hq['hq_state'].str.replace(
        'US-', '', regex=False)
    m = panel_hq.merge(hq, on='permno', how='left')
    m['n_10k'] = m['n_10k'].fillna(0).astype(int)
    m['hdr_state_last'] = m['hdr_states'].apply(last_hdr_state)
    m['n_distinct_states'] = m['hdr_states'].apply(states_distinct)
    m['flag_hdr_changed'] = (m['hdr_changed'] == True)
    m['flag_sec_disagree'] = (
        (m['n_10k'] >= 2)
        & m['hdr_state_last'].notna()
        & m['hq_state_2l'].notna()
        & (m['hdr_state_last'] != m['hq_state_2l'])
    )
    m['flag_hdr_between_state'] = (m['n_10k'] >= 2) & (m['n_distinct_states'] >= 2)
    m['reloc_v6_broad'] = m['flag_hdr_changed'] | m['flag_sec_disagree']
    m['reloc_refined'] = m['flag_hdr_between_state']

    diag = {
        "panel_firms": int(len(m)),
        "flag_covered_firms_n10k_ge1": int((m['n_10k'] >= 1).sum()),
        "flag_hdr_changed_count": int(m['flag_hdr_changed'].sum()),
        "flag_sec_disagree_count": int(m['flag_sec_disagree'].sum()),
        "flag_hdr_between_state_count": int(m['flag_hdr_between_state'].sum()),
        "v6_broad_union_reloc": int(m['reloc_v6_broad'].sum()),
        "refined_between_state_only_reloc": int(m['reloc_refined'].sum()),
        "delta_kept_under_refined": int(
            (m['reloc_v6_broad'] & ~m['reloc_refined']).sum()),
        "empirical_check_hdr_changed_eq_between_state": int(
            (m['flag_hdr_changed'] == m['flag_hdr_between_state']).all()),
        "definition_v6_broad": "reloc = hdr_changed OR sec_disagree",
        "definition_refined": (
            "reloc = (n_10k >= 2) AND (n_distinct_states_in_10K_headers >= 2)"
            " — i.e., the 10-K-header STATE field actually changed between "
            "two distinct US-state codes during in-sample filings. Excludes "
            "the 524 sec_disagree-only firms whose in-sample 10-K headers "
            "were internally consistent (so they did not relocate during the "
            "sample window per the EDGAR record).")
    }
    print("\n=== relocator flag diagnostics ===", flush=True)
    for k, v in diag.items():
        print(f"  {k}: {v}", flush=True)
    reloc_broad = set(m.loc[m['reloc_v6_broad'], 'permno'].astype(int))
    reloc_refined = set(m.loc[m['reloc_refined'], 'permno'].astype(int))
    return reloc_broad, reloc_refined, diag


def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


def task1_refined_io_sign_flip(dfm_refined, reloc_diag):
    """Task 1: 13F IO sign flip on the refined (between-state-only) stable-HQ
    subsample. Static HQ literacy, persistent + time-varying."""
    print("\n\n========================================================", flush=True)
    print("=== TASK 1: IO sign flip on REFINED stable-HQ subsample", flush=True)
    print("========================================================", flush=True)
    d_all = dfm_refined.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos, "
          f"{d_all['hq_state'].nunique()} HQ states", flush=True)

    out = {}
    for measure, io_col, label in [
        ('persistent', 'io_share_persist',
         'persistent (between-firm) IO measure (time-mean io_share)'),
        ('time_varying', 'io_share',
         f'time-varying IO measure (most-recent-quarter, lagged {LAG_MONTHS} months)'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io, perm_io, _ = add_io_terciles(d_io, io_col)
        terc_means = d_io.groupby('io_grp')[io_col].mean().to_dict()
        n_p = d_io['permno'].nunique()
        n_o = len(d_io)
        print(f"  N={n_o:,} firm-months, {n_p} permnos", flush=True)
        print(f"  tercile means: "
              f"{ {k: round(v, 3) for k, v in terc_means.items()} }",
              flush=True)
        pertile = {}
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            r = twfe_three_way(sub)
            pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                            't_state', 'n_obs', 'se_kind']}
            print(f"    {g}: gamma_static={r['coef']:+.6f} "
                  f"cgm_t={r['t']:.2f} state_t={r['t_state']:.2f} "
                  f"n={r['n_obs']:,}", flush=True)

        d_stack = d_io[d_io['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
        d_stack['io_grp_bin'] = np.where(d_stack['io_grp'] == 'IO1_low',
                                         'low', 'high')
        print(f"  running stacked difference (B={B_WCB}) ...", flush=True)
        t_start = time.time()
        sd = stacked_io_difference(d_stack, B=B_WCB, seed=42)
        print(f"    elapsed {time.time()-t_start:.1f}s", flush=True)
        print(f"    DIFF (low-high) = {sd['difference_coef']:+.6f} | "
              f"state-cl SE {sd['se_state_clustered']:.6f} "
              f"t={sd['t_state_clustered']:.2f} | "
              f"wcb p={sd['wcb_p_value']:.4f}", flush=True)
        out[measure] = {
            "io_measure": label,
            "tercile_means": {k: float(v) for k, v in terc_means.items()},
            "per_tercile_threeway_static": pertile,
            "stacked_difference_static": sd,
            "n_permnos": int(n_p),
            "n_firm_months": int(n_o),
        }

    return d_all, out


def recovery_verdict(out):
    """Apply the operator's RECOVERY rule:
        RECOVERY <=> |DIFF|>0.015 AND wcb p<0.30 for EITHER measure.
    Report per-measure and overall."""
    verdicts = {}
    overall_recovery = False
    for measure in ['persistent', 'time_varying']:
        sd = out[measure]['stacked_difference_static']
        diff = sd['difference_coef']
        p = sd['wcb_p_value']
        mag = abs(diff)
        rec = bool((mag > 0.015) and (p < 0.30))
        verdicts[measure] = {
            "DIFF": diff,
            "abs_DIFF": mag,
            "wcb_p": p,
            "criterion_abs_diff_gt_0p015": bool(mag > 0.015),
            "criterion_wcb_p_lt_0p30": bool(p < 0.30),
            "recovery": rec,
        }
        if rec:
            overall_recovery = True
    verdicts['overall_recovery'] = bool(overall_recovery)
    verdicts['rule'] = ("RECOVERY <=> |DIFF| > 0.015 AND wcb p < 0.30 for "
                        "EITHER the persistent or time-varying measure. "
                        "Otherwise NO RECOVERY.")
    return verdicts


def task2_iv_decile_lit_gradient(dfm_broad, results):
    """Task 2: Within low-IO tercile of the BROAD stable-HQ sample (the
    1,366-dropped v7 stable-HQ centerpiece sample), characterize the
    (low-lit minus high-lit) three-way DIFF by IV decile.

    Procedure for each IV decile d (0..9), within low-IO:
      - Split firms (within that IV decile) into high-lit-state vs
        low-lit-state via state-level median split on
        literacy_score_corrected mean within low-IO.
      - Use stacked_state_group_difference to estimate (low-lit minus high-lit)
        difference in the three-way coefficient, with state-clustered SE +
        WCB p.
    """
    print("\n\n========================================================", flush=True)
    print("=== TASK 2: IV-decile decomposition of within-retail wrong-sign", flush=True)
    print("========================================================", flush=True)
    d_all = dfm_broad.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    out = {}

    for measure, io_col, label in [
        ('persistent', 'io_share_persist', 'persistent IO measure'),
        ('time_varying', 'io_share', 'time-varying IO measure'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io, _, _ = add_io_terciles(d_io, io_col)
        d_lowio = d_io[d_io['io_grp'] == 'IO1_low'].copy()
        print(f"  low-IO sample: {len(d_lowio):,} firm-months, "
              f"{d_lowio['permno'].nunique()} permnos", flush=True)

        # IV deciles within low-IO (firm-level qcut on time-mean iv).
        firm_iv = d_lowio.groupby('permno')['iv'].mean()
        firm_iv_dec = pd.qcut(firm_iv, 10, labels=list(range(10)),
                              duplicates='drop')
        d_lowio['iv_dec'] = d_lowio['permno'].map(firm_iv_dec)

        # State-level mean literacy WITHIN low-IO subsample for the lit split.
        state_lit = (d_lowio.groupby('hq_state')
                     ['literacy_score_corrected'].mean())
        lit_median = state_lit.median()
        hi_lit_states = set(state_lit[state_lit >= lit_median].index)
        lo_lit_states = set(state_lit[state_lit < lit_median].index)
        d_lowio['lit_grp'] = np.where(
            d_lowio['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')

        decile_results = []
        for dec in sorted(d_lowio['iv_dec'].dropna().unique()):
            sub = d_lowio[d_lowio['iv_dec'] == dec].copy()
            n_st = sub['hq_state'].nunique()
            n_p = sub['permno'].nunique()
            n_obs = len(sub)
            if n_obs < 2000 or n_st < 10:
                print(f"  IV dec {dec}: SKIP (N={n_obs:,}, states={n_st})",
                      flush=True)
                continue
            # Per-group three-ways in this decile.
            sub_hl = sub[sub['lit_grp'] == 'high_lit']
            sub_ll = sub[sub['lit_grp'] == 'low_lit']
            r_hl = twfe_three_way(sub_hl) if len(sub_hl) > 1000 else None
            r_ll = twfe_three_way(sub_ll) if len(sub_ll) > 1000 else None
            try:
                sd_lit = stacked_state_group_difference(
                    sub, 'lit_grp', 'low_lit', B=B_WCB, seed=42)
            except Exception as e:
                print(f"  IV dec {dec}: stacked diff failed ({e}), skipping",
                      flush=True)
                continue
            mean_iv = float(sub['iv'].mean())
            decile_results.append({
                "iv_decile": int(dec),
                "mean_iv_z": mean_iv,
                "n_obs": int(n_obs),
                "n_states": int(n_st),
                "n_permnos": int(n_p),
                "n_high_lit_obs": int(len(sub_hl)),
                "n_low_lit_obs": int(len(sub_ll)),
                "gamma_high_lit": (float(r_hl['coef'])
                                   if r_hl is not None else None),
                "t_state_high_lit": (float(r_hl['t_state'])
                                     if r_hl is not None else None),
                "gamma_low_lit": (float(r_ll['coef'])
                                  if r_ll is not None else None),
                "t_state_low_lit": (float(r_ll['t_state'])
                                    if r_ll is not None else None),
                "DIFF_lowlit_minus_highlit": float(sd_lit['difference_coef']),
                "se_state_clustered": float(sd_lit['se_state_clustered']),
                "t_state_clustered": float(sd_lit['t_state_clustered']),
                "wcb_p_value": float(sd_lit['wcb_p_value']),
                "B_wcb": int(sd_lit['B']),
                "n_state_clusters_in_diff": int(sd_lit['n_state_clusters']),
            })
            print(f"  IV dec {dec} (mean iv_z={mean_iv:+.2f}, N={n_obs:,}): "
                  f"DIFF={sd_lit['difference_coef']:+.6f} "
                  f"state-t={sd_lit['t_state_clustered']:.2f} "
                  f"wcb-p={sd_lit['wcb_p_value']:.4f}", flush=True)
            if r_hl is not None and r_ll is not None:
                print(f"    high-lit gamma={r_hl['coef']:+.6f} "
                      f"(state-t={r_hl['t_state']:.2f}, n={r_hl['n_obs']:,})",
                      flush=True)
                print(f"    low-lit  gamma={r_ll['coef']:+.6f} "
                      f"(state-t={r_ll['t_state']:.2f}, n={r_ll['n_obs']:,})",
                      flush=True)

        # Sample-weighted aggregate of DIFF to compare with the v7 stable-HQ
        # centerpiece headline (+0.0033 persistent).
        if decile_results:
            tot_n = sum(r['n_obs'] for r in decile_results)
            wavg_diff = sum(r['DIFF_lowlit_minus_highlit'] * r['n_obs']
                            for r in decile_results) / tot_n
            sum_pos = sum(1 for r in decile_results
                          if r['DIFF_lowlit_minus_highlit'] > 0)
            sum_neg = sum(1 for r in decile_results
                          if r['DIFF_lowlit_minus_highlit'] < 0)
            biggest_pos = max(decile_results,
                              key=lambda r: r['DIFF_lowlit_minus_highlit'])
            biggest_neg = min(decile_results,
                              key=lambda r: r['DIFF_lowlit_minus_highlit'])
            summary = {
                "n_deciles_estimated": len(decile_results),
                "n_deciles_positive_DIFF": sum_pos,
                "n_deciles_negative_DIFF": sum_neg,
                "obs_weighted_avg_DIFF": float(wavg_diff),
                "max_positive_DIFF_decile": int(biggest_pos['iv_decile']),
                "max_positive_DIFF_value": float(
                    biggest_pos['DIFF_lowlit_minus_highlit']),
                "max_negative_DIFF_decile": int(biggest_neg['iv_decile']),
                "max_negative_DIFF_value": float(
                    biggest_neg['DIFF_lowlit_minus_highlit']),
            }
            print(f"\n  summary: obs-weighted DIFF = {wavg_diff:+.6f} "
                  f"({sum_pos} positive deciles, {sum_neg} negative)",
                  flush=True)
        else:
            summary = {"n_deciles_estimated": 0}

        out[measure] = {
            "io_measure": label,
            "n_hi_lit_states": len(hi_lit_states),
            "n_lo_lit_states": len(lo_lit_states),
            "lit_split_median_threshold": float(lit_median),
            "per_iv_decile": decile_results,
            "summary": summary,
        }

    results['task2_iv_decile_decomposition'] = out
    return out


def main():
    t0 = time.time()
    print("=== loading inputs ===", flush=True)
    hq = pd.read_parquet(HQ_EDGAR)
    print(f"  EDGAR HQ-state pull: {len(hq):,} firms", flush=True)

    dfm = pd.read_parquet(DFM_PIT_CACHE)
    print(f"  PIT panel (full): {len(dfm):,} firm-months, "
          f"{dfm['permno'].nunique():,} permnos", flush=True)

    # Build relocator sets (broad + refined).
    panel_hq = (dfm.groupby('permno')
                .agg(hq_state=('hq_state', 'first'),
                     n_months=('date', 'size'))
                .reset_index())
    reloc_broad, reloc_refined, reloc_diag = build_flags(hq, panel_hq)

    dfm_refined = dfm[~dfm['permno'].isin(reloc_refined)].copy()
    print(f"\n  REFINED stable-HQ panel: {len(dfm_refined):,} firm-months "
          f"(dropped {len(dfm) - len(dfm_refined):,}), "
          f"{dfm_refined['permno'].nunique():,} permnos "
          f"(dropped {dfm['permno'].nunique() - dfm_refined['permno'].nunique()})",
          flush=True)

    # TASK 1 — IO sign flip on refined sample.
    d_all_refined, t1 = task1_refined_io_sign_flip(dfm_refined, reloc_diag)
    verdicts = recovery_verdict(t1)
    print("\n=== Recovery verdict per operator's rule ===", flush=True)
    for measure in ['persistent', 'time_varying']:
        v = verdicts[measure]
        print(f"  {measure}: DIFF={v['DIFF']:+.6f}, |DIFF|={v['abs_DIFF']:.6f} "
              f"(>{0.015}? {v['criterion_abs_diff_gt_0p015']}), "
              f"wcb p={v['wcb_p']:.4f} (<{0.30}? "
              f"{v['criterion_wcb_p_lt_0p30']}) -> recovery={v['recovery']}",
              flush=True)
    overall = "RECOVERY" if verdicts['overall_recovery'] else "NO RECOVERY"
    print(f"  OVERALL: {overall}", flush=True)

    results_t1 = {
        "task": ("Task 1: 13F IO sign flip on REFINED stable-HQ subsample. "
                 "Refined relocator = between-state 10-K-header change only "
                 "(excludes sec_disagree-only firms whose in-sample 10-K "
                 "headers were internally stable). Static HQ literacy."),
        "relocation_flag_diagnostics": reloc_diag,
        "refined_stable_hq_sample": {
            "firm_months": int(len(dfm_refined)),
            "permnos": int(dfm_refined['permno'].nunique()),
            "static_estimation_sample": int(len(d_all_refined)),
        },
        "io_sign_flip_refined": t1,
        "recovery_verdict": verdicts,
        "reference_v6_full_panel": {
            "io_persistent_DIFF": -0.0211, "io_persistent_wcb_p": 0.144,
            "io_time_varying_DIFF": -0.0181, "io_time_varying_wcb_p": 0.264,
        },
        "reference_v7_stable_hq_broad": {
            "io_persistent_DIFF": -0.0079, "io_persistent_wcb_p": 0.650,
            "io_time_varying_DIFF": -0.0058, "io_time_varying_wcb_p": 0.755,
        },
        "meta": {
            "B_wcb": B_WCB, "lag_months": LAG_MONTHS, "seed": 42,
            "inputs": {"hq_edgar": HQ_EDGAR, "dfm_pit_cache": DFM_PIT_CACHE},
        },
    }
    with open(OUT_JSON_T1, 'w') as f:
        json.dump(results_t1, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON_T1} ===", flush=True)

    # TASK 2 — IV-decile decomposition on broad stable-HQ.
    if os.path.exists(DFM_STABLE_BROAD_CACHE):
        dfm_broad = pd.read_parquet(DFM_STABLE_BROAD_CACHE)
        print(f"\n  loaded BROAD stable-HQ cached panel: "
              f"{len(dfm_broad):,} firm-months, "
              f"{dfm_broad['permno'].nunique():,} permnos", flush=True)
    else:
        dfm_broad = dfm[~dfm['permno'].isin(reloc_broad)].copy()
        print(f"\n  built BROAD stable-HQ panel inline: "
              f"{len(dfm_broad):,} firm-months", flush=True)

    results_t2 = {
        "task": ("Task 2: IV-decile decomposition of the within-retail "
                 "(low-IO) wrong-sign literacy gradient on the v7 BROAD "
                 "stable-HQ sample (1,366 firms dropped, 6,081 permnos, "
                 "519,090 firm-months). Within each NYSE-IV decile (0..9, "
                 "qcut on firm-level mean IV within low-IO), compute the "
                 "stacked (low-lit minus high-lit) three-way DIFF with "
                 "state-clustered SE and wild-cluster bootstrap p (B=4999, "
                 "Webb 6-pt). The lit split is the same as the v7 stable-HQ "
                 "centerpiece (median state-mean literacy within low-IO)."),
        "broad_stable_hq_sample": {
            "firm_months": int(len(dfm_broad)),
            "permnos": int(dfm_broad['permno'].nunique()),
        },
        "reference_v7_stable_hq_centerpiece_within_retail": {
            "persistent_DIFF_lowlit_minus_highlit": +0.0033,
            "persistent_wcb_p": 0.024,
            "note": "within-retail wrong sign (positive when seed predicts "
                    "negative or zero) on the BROAD stable-HQ sample, "
                    "persistent IO.",
        },
    }
    task2_iv_decile_lit_gradient(dfm_broad, results_t2)
    results_t2['meta'] = {
        "B_wcb": B_WCB, "lag_months": LAG_MONTHS, "seed": 42,
    }
    with open(OUT_JSON_T2, 'w') as f:
        json.dump(results_t2, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON_T2} ===", flush=True)

    print(f"\n=== TOTAL TIME: {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == '__main__':
    main()
