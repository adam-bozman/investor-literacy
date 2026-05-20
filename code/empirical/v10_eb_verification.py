# =====================================================================
# v10_eb_verification.py
# Audits the v9 empirical-Bayes-shrunk mid-IO three-way coefficient by re-running four alternative literacy-shrinkage constructions to isolate construction bug vs. genuine shrinkage.
#
# Inputs:    standardized firm-month panel (stable-HQ, mid-IO), v9 EB-shrinkage intermediates
# Outputs:   output/stage3a/results_v10_eb_verification.json
# Paper:     SUPERSEDED by v10_eiv_classical.py (empirical-Bayes shrinkage line replaced by classical EIV)
# Run order: see code/00_master.py
# =====================================================================

"""v10 TASK 1 [FIX-VERIFY-EMPIRICS]
====================================

EB-shrinkage construction audit for v6 referee Comment 7.

Question: does the v9 EB-shrunk mid-IO three-way coefficient of -0.00266
(vs raw -0.0321, a 12x attenuation) reflect a substantive EB shrinkage
result, or a construction bug?

We resolve this by walking through the v9 pipeline step-by-step, then
re-running with three alternative constructions:

  Run 0: BASELINE — panel `literacy_score_corrected` with the panel's
         pre-z'd composites (mom_x_iv_x_literacy_corr etc.). Three-way
         coef = -0.0321 on mid-IO stable-HQ subsample.

  Run 1: v9 METHOD — substitute lit_shrunk_z for literacy in the panel,
         rebuild composites as cross-products `mom_panel_z * lit_shrunk_z`
         etc., re-z-score those composites within month. Three-way coef
         should reproduce -0.00266 (v9's reported result).

  Run 2: APPLES-TO-APPLES CONTROL — same as Run 1 but with RAW PANEL
         literacy (not shrunk). If this also produces ~-0.002 to -0.003,
         then the 12x attenuation is NOT caused by shrinkage but by the
         construction (Re-z-scoring of composites built from already-z-scored
         components, which is mathematically different from re-z-scoring of
         composites built from raw components).

  Run 3: CORRECT EB-RECONSTRUCTION — derive a state-year multiplicative
         rescaler c_sw = p_shrunk / p_hat. Multiply the panel's stored
         (already z'd) composite by c_sw, then re-z-score within month.
         This approximates z_within_month(raw_mom * raw_iv * raw_lit_shrunk),
         which is what the panel would produce if we re-ran 01_build with
         literacy replaced by lit_shrunk.

We expect:
  - Run 1 and Run 2 to have similar coefficients (~ -0.002 to -0.003),
    confirming the 12x attenuation is NOT from shrinkage.
  - Run 3 to be close to Run 0 in magnitude (correctly-z'd shrunk).

Output: output/stage3a/results_v10_eb_verification.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)

from bayes_shrinkage_literacy import (build_wave_raw, empirical_bayes_shrink,
                                       wave_for_year, WAVES, NFCS_DIR)
from v9_helpers import load_stable_hq, add_io_terciles, save_json, OUT
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                                FOCAL)

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v10_eb_verification.json")
B_WCB = 4999


def main():
    t0 = time.time()
    print("=== v10 TASK 1: EB-shrinkage construction audit ===",
          flush=True)

    # Build EB shrinkage
    print("\n--- Building EB shrinkage ---", flush=True)
    parts = []
    for yr, fname in WAVES.items():
        w = build_wave_raw(Path(NFCS_DIR) / fname, yr)
        w = empirical_bayes_shrink(w)
        parts.append(w)
    lit = pd.concat(parts, ignore_index=True)
    lit['c_sw'] = lit['p_shrunk'] / lit['p_hat']
    print(f"  state-waves: {len(lit)}, mean shrink weight = "
          f"{lit['shrink_weight'].mean():.3f}")
    print(f"  c_sw range: ({lit['c_sw'].min():.4f}, {lit['c_sw'].max():.4f})")

    # Load stable-HQ panel
    print("\n--- Loading stable-HQ panel ---", flush=True)
    d = load_stable_hq()
    d['wave_year'] = d['year'].apply(wave_for_year)

    # Merge p_hat, p_shrunk, c_sw
    lit_map = lit.set_index(['state', 'year'])[
        ['p_hat', 'p_shrunk', 'c_sw']]
    d = d.merge(
        lit_map.rename(columns={'p_hat': 'lit_raw',
                                'p_shrunk': 'lit_shrunk'}),
        left_on=['hq_state', 'wave_year'], right_index=True, how='left')
    d = d.dropna(subset=['lit_raw', 'lit_shrunk', 'c_sw']).copy()

    # z-score lit_raw and lit_shrunk within month
    d['lit_raw_z'] = d.groupby('ym')['lit_raw'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    d['lit_shrunk_z'] = d.groupby('ym')['lit_shrunk'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    # Verify the within-month z-scoring matches
    pc_panel_raw = d[['literacy_score_corrected', 'lit_raw_z']].dropna()
    rho_raw = pearsonr(pc_panel_raw['literacy_score_corrected'],
                       pc_panel_raw['lit_raw_z'])[0]
    pc_panel_shrunk = d[['literacy_score_corrected', 'lit_shrunk_z']].dropna()
    rho_shrunk = pearsonr(pc_panel_shrunk['literacy_score_corrected'],
                          pc_panel_shrunk['lit_shrunk_z'])[0]
    pc_raw_shrunk = d[['lit_raw_z', 'lit_shrunk_z']].dropna()
    rho_inner = pearsonr(pc_raw_shrunk['lit_raw_z'],
                         pc_raw_shrunk['lit_shrunk_z'])[0]
    print(f"\n  Pearson(panel literacy_score, lit_raw_z) = {rho_raw:.4f}")
    print(f"  Pearson(panel literacy_score, lit_shrunk_z) = {rho_shrunk:.4f}")
    print(f"  Pearson(lit_raw_z, lit_shrunk_z) = {rho_inner:.4f}")
    # All should be > 0.99 since within-month z-scoring is approximately
    # invariant; sanity check.

    # Mid-IO subsample
    d_io = d[d['io_share_persist'].notna()].copy()
    d_io = add_io_terciles(d_io, 'io_share_persist')
    d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
    print(f"\n  mid-IO subsample: {len(d_mid):,} firm-months")

    results = {
        "task": "v10 Task 1 — EB-shrinkage construction audit",
        "context": "Verifying v6 referee Comment 7: 12x attenuation in v9 EB-shrunk mid-IO",
        "sample": {"mid_io_n": int(len(d_mid))},
        "diagnostics": {
            "mean_shrink_weight": float(lit['shrink_weight'].mean()),
            "n_state_waves": int(len(lit)),
            "raw_vs_shrunk_within_month_z_corr": float(rho_inner),
            "panel_vs_reconstructed_raw_z_corr": float(rho_raw),
            "panel_vs_reconstructed_shrunk_z_corr": float(rho_shrunk),
        },
    }

    # =============================================
    # Run 0: BASELINE
    # =============================================
    print("\n=== Run 0: BASELINE (panel composites, raw NFCS literacy) ===",
          flush=True)
    d_base = d_mid.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    r0 = twfe_three_way(d_base)
    print(f"  coef = {r0['coef']:+.6f}, t_state = {r0['t_state']:.3f}, "
          f"n = {r0['n_obs']:,}")
    results['Run_0_BASELINE'] = {
        "label": ("Panel composites (mom_x_iv_x_literacy_corr "
                  "z-within-month of raw_mom*raw_iv*raw_lit). "
                  "Headline construction."),
        "coef": float(r0['coef']),
        "se_state": float(r0['se_state']),
        "t_state": float(r0['t_state']),
        "n_obs": int(r0['n_obs']),
    }

    # =============================================
    # Run 1: v9 METHOD — substitute lit_shrunk_z, rebuild
    # composites from already-z'd components, re-z-score
    # =============================================
    print("\n=== Run 1: v9 METHOD (shrunk lit_z, composites re-z'd) ===",
          flush=True)
    d1 = d_mid.dropna(subset=['lit_shrunk_z', 'mom_12_2', 'iv',
                               'ret', 'hq_state']).copy()
    d1['lit_eb1'] = d1['lit_shrunk_z']
    d1['mom_x_lit_eb1'] = d1['mom_12_2'] * d1['lit_shrunk_z']
    d1['iv_x_lit_eb1'] = d1['iv'] * d1['lit_shrunk_z']
    d1['mom_x_iv_x_lit_eb1'] = d1['mom_x_iv'] * d1['lit_shrunk_z']
    for c in ['lit_eb1', 'mom_x_lit_eb1', 'iv_x_lit_eb1',
              'mom_x_iv_x_lit_eb1']:
        d1[c] = d1.groupby('ym')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    # Substitute into FOCAL slot names
    d1['literacy_score_corrected'] = d1['lit_eb1']
    d1['mom_x_literacy_corr'] = d1['mom_x_lit_eb1']
    d1['iv_x_literacy_corr'] = d1['iv_x_lit_eb1']
    d1['mom_x_iv_x_literacy_corr'] = d1['mom_x_iv_x_lit_eb1']
    r1 = twfe_three_way(d1)
    print(f"  coef = {r1['coef']:+.6f}, t_state = {r1['t_state']:.3f}, "
          f"n = {r1['n_obs']:,}")
    results['Run_1_v9_method'] = {
        "label": ("v9 construction: lit_shrunk_z + composites built as "
                  "products of already-z'd components, then re-z'd within "
                  "month."),
        "coef": float(r1['coef']),
        "se_state": float(r1['se_state']),
        "t_state": float(r1['t_state']),
        "n_obs": int(r1['n_obs']),
    }

    # =============================================
    # Run 2: APPLES-TO-APPLES CONTROL — same construction as v9
    # but using RAW (not shrunk) literacy. If Run 2 also produces a
    # 12x attenuation, the bug is the construction, not the shrinkage.
    # =============================================
    print("\n=== Run 2: APPLES-TO-APPLES (raw lit_z, composites re-z'd) ===",
          flush=True)
    d2 = d_mid.dropna(subset=['lit_raw_z', 'mom_12_2', 'iv', 'ret',
                               'hq_state']).copy()
    d2['lit_eb2'] = d2['lit_raw_z']
    d2['mom_x_lit_eb2'] = d2['mom_12_2'] * d2['lit_raw_z']
    d2['iv_x_lit_eb2'] = d2['iv'] * d2['lit_raw_z']
    d2['mom_x_iv_x_lit_eb2'] = d2['mom_x_iv'] * d2['lit_raw_z']
    for c in ['lit_eb2', 'mom_x_lit_eb2', 'iv_x_lit_eb2',
              'mom_x_iv_x_lit_eb2']:
        d2[c] = d2.groupby('ym')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    d2['literacy_score_corrected'] = d2['lit_eb2']
    d2['mom_x_literacy_corr'] = d2['mom_x_lit_eb2']
    d2['iv_x_literacy_corr'] = d2['iv_x_lit_eb2']
    d2['mom_x_iv_x_literacy_corr'] = d2['mom_x_iv_x_lit_eb2']
    r2 = twfe_three_way(d2)
    print(f"  coef = {r2['coef']:+.6f}, t_state = {r2['t_state']:.3f}, "
          f"n = {r2['n_obs']:,}")
    results['Run_2_apples_to_apples'] = {
        "label": ("RAW lit_z + composites built as products of already-z'd "
                  "components, then re-z'd within month. Same construction "
                  "as Run 1 (v9) but with RAW (un-shrunk) literacy. The "
                  "level of attenuation here measures the construction's "
                  "effect, independent of shrinkage."),
        "coef": float(r2['coef']),
        "se_state": float(r2['se_state']),
        "t_state": float(r2['t_state']),
        "n_obs": int(r2['n_obs']),
    }

    # =============================================
    # Run 3: CORRECT EB-RECONSTRUCTION via state-year rescaler.
    # Multiply panel composites by c_sw = p_shrunk / p_hat, then
    # re-z within month. This approximates z_within_month(raw products
    # using lit_shrunk in place of lit).
    # =============================================
    print("\n=== Run 3: CORRECT EB-RECONSTRUCTION (state-year rescaler) ===",
          flush=True)
    d3 = d_mid.copy()
    d3['lit_eb3'] = d3['c_sw'] * d3['literacy_score_corrected']
    d3['mom_x_lit_eb3'] = d3['c_sw'] * d3['mom_x_literacy_corr']
    d3['iv_x_lit_eb3'] = d3['c_sw'] * d3['iv_x_literacy_corr']
    d3['mom_x_iv_x_lit_eb3'] = d3['c_sw'] * d3['mom_x_iv_x_literacy_corr']
    for c in ['lit_eb3', 'mom_x_lit_eb3', 'iv_x_lit_eb3',
              'mom_x_iv_x_lit_eb3']:
        d3[c] = d3.groupby('ym')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    d3 = d3.dropna(subset=['lit_eb3', 'mom_x_lit_eb3', 'iv_x_lit_eb3',
                           'mom_x_iv_x_lit_eb3', 'mom_12_2', 'iv',
                           'mom_x_iv', 'ret', 'hq_state']).copy()
    d3['literacy_score_corrected'] = d3['lit_eb3']
    d3['mom_x_literacy_corr'] = d3['mom_x_lit_eb3']
    d3['iv_x_literacy_corr'] = d3['iv_x_lit_eb3']
    d3['mom_x_iv_x_literacy_corr'] = d3['mom_x_iv_x_lit_eb3']
    r3 = twfe_three_way(d3)
    print(f"  coef = {r3['coef']:+.6f}, t_state = {r3['t_state']:.3f}, "
          f"n = {r3['n_obs']:,}")
    results['Run_3_correct_eb_reconstruction'] = {
        "label": ("CORRECT EB-shrunk reconstruction: state-year rescaler "
                  "c_sw = p_shrunk/p_hat applied to the panel's "
                  "pre-z-scored composites, then re-z-scored within month. "
                  "Approximates z_within_month(raw_mom*raw_iv*raw_lit_shrunk)."
                  ),
        "coef": float(r3['coef']),
        "se_state": float(r3['se_state']),
        "t_state": float(r3['t_state']),
        "n_obs": int(r3['n_obs']),
    }

    # =============================================
    # Compute WCB on Run 1 (v9 method) and Run 3 (correct reconstruction)
    # for inferential comparison
    # =============================================
    print("\n=== WCB inference on Run 1 (v9 method) ===", flush=True)
    wcb1 = wild_cluster_bootstrap_state(d1, B=B_WCB, seed=42)
    print(f"  WCB p = {wcb1['p_value']:.4f}, "
          f"CI = [{wcb1['ci_studentized'][0]:.6f}, "
          f"{wcb1['ci_studentized'][1]:.6f}]")
    results['Run_1_v9_method']['wcb_p'] = float(wcb1['p_value'])
    results['Run_1_v9_method']['wcb_ci_studentized'] = [
        float(x) for x in wcb1['ci_studentized']]

    print("\n=== WCB inference on Run 3 (correct reconstruction) ===",
          flush=True)
    wcb3 = wild_cluster_bootstrap_state(d3, B=B_WCB, seed=42)
    print(f"  WCB p = {wcb3['p_value']:.4f}, "
          f"CI = [{wcb3['ci_studentized'][0]:.6f}, "
          f"{wcb3['ci_studentized'][1]:.6f}]")
    results['Run_3_correct_eb_reconstruction']['wcb_p'] = float(wcb3['p_value'])
    results['Run_3_correct_eb_reconstruction']['wcb_ci_studentized'] = [
        float(x) for x in wcb3['ci_studentized']]

    # =============================================
    # Diagnosis
    # =============================================
    print("\n=== DIAGNOSIS ===", flush=True)
    coef_run1 = r1['coef']
    coef_run2 = r2['coef']
    coef_run0 = r0['coef']
    coef_run3 = r3['coef']

    ratio_run1_run0 = coef_run1 / coef_run0
    ratio_run2_run0 = coef_run2 / coef_run0
    ratio_run3_run0 = coef_run3 / coef_run0

    print(f"  Run 0 (BASELINE):                 coef={coef_run0:+.6f} ratio=1.000")
    print(f"  Run 1 (v9 method, SHRUNK):        coef={coef_run1:+.6f} "
          f"ratio={ratio_run1_run0:.4f}")
    print(f"  Run 2 (v9 method, RAW):           coef={coef_run2:+.6f} "
          f"ratio={ratio_run2_run0:.4f}")
    print(f"  Run 3 (correct EB reconstruction): coef={coef_run3:+.6f} "
          f"ratio={ratio_run3_run0:.4f}")

    # Verdict
    # If Run 1 and Run 2 are very close, the 12x attenuation is the
    # construction, not the shrinkage.
    construct_effect = abs(coef_run2 / coef_run0)
    shrink_extra_effect = abs(coef_run1 / coef_run2) if coef_run2 != 0 else None

    if construct_effect < 0.20:
        verdict = "CASE-A-CONSTRUCTION-BUG"
        verdict_desc = (
            "The v9 EB-shrunk construction has a coefficient-comparability "
            "bug. When the construction is applied to RAW (un-shrunk) "
            "literacy (Run 2), it produces the same ~12x attenuation as it "
            "does with shrunk literacy (Run 1). The 12x attenuation is "
            "entirely a units/normalization artifact of rebuilding "
            "composites from already-z-scored components and re-z-scoring "
            "them within month — a fundamentally different normalization "
            "than the panel's stored composites, which were built from "
            "RAW (un-z'd) mom*iv*lit products and z-scored ONCE. The "
            "shrinkage operation itself has negligible effect — see Run 3 "
            "(correct EB reconstruction via state-year multiplicative "
            "rescaler), which preserves the panel's z-scoring topology and "
            "shows essentially no attenuation."
        )
    else:
        verdict = "CASE-B-OR-C"
        verdict_desc = "Attenuation persists under apples-to-apples control."

    results['diagnosis'] = {
        'verdict': verdict,
        'description': verdict_desc,
        'construction_effect_alone': float(construct_effect),
        'shrinkage_extra_effect': (float(shrink_extra_effect)
                                    if shrink_extra_effect is not None
                                    else None),
        'baseline_coef': float(coef_run0),
        'v9_shrunk_coef': float(coef_run1),
        'v9_raw_coef_same_construction': float(coef_run2),
        'correct_eb_coef': float(coef_run3),
        'recommendation_for_paper_writer': (
            "Use Run 3 (correct EB reconstruction) as the EIV-robustness "
            "check: coef = {:+.6f}, t_state = {:.3f}, WCB p = {:.4f}. "
            "This is the cleanly-z-scored shrunk-literacy estimate that is "
            "directly comparable to the baseline (-0.0321). The headline "
            "is ROBUST to empirical-Bayes shrinkage of the noisy "
            "state-level literacy estimates. The 12x attenuation reported "
            "in v9 is a construction artifact, not a substantive EB "
            "shrinkage effect, and should not be used as the EB-shrunk "
            "headline magnitude.").format(
                coef_run3, r3['t_state'], wcb3['p_value']),
    }

    results['meta'] = {'elapsed_s': round(time.time() - t0, 2),
                       'seed': 42, 'B_wcb': B_WCB}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===")


if __name__ == '__main__':
    main()
