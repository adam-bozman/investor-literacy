# =====================================================================
# v9_pre_post_2015_inference.py
# Adds WCB p-values to the pre/post-2015 mid-IO table and refits the quadratic-IO four-way separately in each half (graduation/time-window inference).
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); deepen_estimators, v8_diagnostic_three._quadratic_four_way
# Outputs:   output/stage3a/results_v9_pre_post_2015_inference.json (printed diagnostics + JSON)
# Paper:     T8 tab:prepost_grad + IA 5-window grid section
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 8 — Pre/post-2015 inference discipline.

Triage [FIX] Item 6 (High). Two sub-tasks:

  (a) Add WCB p column to the v8 pre/post-2015 mid-IO table (currently
      only state-cluster t reported). State-clustered WCB p for the mid-IO
      coefficient in pre-2015 and post-2015 subsamples of stable-HQ.

  (b) Refit the quadratic-IO four-way separately on pre-2015 and
      post-2015 stable-HQ subsamples. Report gamma_5 (quadratic) for each,
      state-t, WCB p. Test whether IO* is stable across the two halves
      even as the literacy slope magnitude attenuates.

Output: output/stage3a/results_v9_pre_post_2015_inference.json
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
from v9_helpers import load_stable_hq, add_io_terciles, save_json, OUT
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                                FOCAL)
from v8_diagnostic_three import _quadratic_four_way

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_pre_post_2015_inference.json")
B_WCB = 4999
CUT_DATE = pd.Timestamp("2015-01-01")


def main():
    t0 = time.time()
    print("=== v9 Test 8: Pre/post-2015 inference discipline ===",
          flush=True)
    d = load_stable_hq()
    print(f"  stable-HQ: {len(d):,} firm-months", flush=True)

    results = {
        "test": "v9 Test 8: Pre/post-2015 inference",
        "triage_fix": "Item 6 (High)",
        "cut_date": str(CUT_DATE.date()),
    }

    # ============================================================
    # Task A: WCB p on mid-IO coef in pre/post-2015
    # ============================================================
    print("\n--- Task A: WCB p on mid-IO coef in pre/post-2015 ---",
          flush=True)
    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n  {label} ({io_col}):", flush=True)
        d_io = d[d[io_col].notna()].copy()
        d_io = add_io_terciles(d_io, io_col)
        d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
        d_mid = d_mid.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

        out = {}
        for period, mask in [
            ("pre_2015", d_mid['date'] < CUT_DATE),
            ("post_2015", d_mid['date'] >= CUT_DATE),
        ]:
            sub = d_mid[mask].copy()
            print(f"    {period}: n={len(sub):,}, "
                  f"states={sub['hq_state'].nunique()}", flush=True)
            r = twfe_three_way(sub)
            print(f"      gamma={r['coef']:+.6f} state_t={r['t_state']:.2f}",
                  flush=True)
            wcb = wild_cluster_bootstrap_state(sub, B=B_WCB, seed=42)
            print(f"      wcb-p={wcb['p_value']:.4f}", flush=True)
            out[period] = {
                "coef": float(r['coef']),
                "se_state": float(r['se_state']),
                "t_state": float(r['t_state']),
                "n_obs": int(r['n_obs']),
                "wcb_p_value": float(wcb['p_value']),
                "wcb_ci_studentized": [float(x)
                                       for x in wcb['ci_studentized']],
            }
        results[f"taskA_mid_IO_{label}_pre_post"] = out

    # ============================================================
    # Task B: Quadratic-IO four-way separately on pre/post-2015
    # ============================================================
    print("\n--- Task B: Quadratic-IO four-way pre/post-2015 ---",
          flush=True)
    d_pers = d.dropna(subset=FOCAL + ['ret', 'hq_state',
                                       'io_share_persist']).copy()
    for period, mask in [
        ("pre_2015", d_pers['date'] < CUT_DATE),
        ("post_2015", d_pers['date'] >= CUT_DATE),
    ]:
        sub = d_pers[mask].copy()
        print(f"\n  {period} (persistent IO): n={len(sub):,}", flush=True)
        r = _quadratic_four_way(sub, 'io_share_persist',
                                label=f"persistent_{period}",
                                B=B_WCB, seed=601 + (0 if 'pre' in period
                                                    else 1))
        print(f"    gamma_4 (linear) = {r['gamma_4_linear']:+.6f} "
              f"(state-t={r['t_state_clustered_linear']:.2f}, "
              f"wcb-p={r['wcb_p_value_linear']:.4f})", flush=True)
        print(f"    gamma_5 (quadratic) = {r['gamma_5_quadratic']:+.6f} "
              f"(state-t={r['t_state_clustered_quadratic']:.2f}, "
              f"wcb-p={r['wcb_p_value_quadratic']:.4f})", flush=True)
        if r['implied_io_turning_point'] is not None:
            print(f"    IO* = {r['implied_io_turning_point']:.4f}",
                  flush=True)
        results[f"taskB_quadratic_persistent_{period}"] = r

    # Also IO distribution shift across periods
    io_pre = d_pers.loc[d_pers['date'] < CUT_DATE, 'io_share_persist']
    io_post = d_pers.loc[d_pers['date'] >= CUT_DATE, 'io_share_persist']
    results['io_distribution_shift'] = {
        "pre_2015_io_mean": float(io_pre.mean()),
        "pre_2015_io_median": float(io_pre.median()),
        "post_2015_io_mean": float(io_post.mean()),
        "post_2015_io_median": float(io_post.median()),
    }

    # Verdict
    io_star_pre = (results.get("taskB_quadratic_persistent_pre_2015", {})
                   .get('implied_io_turning_point'))
    io_star_post = (results.get("taskB_quadratic_persistent_post_2015", {})
                    .get('implied_io_turning_point'))
    notes = []
    notes.append(f"IO* (persistent quadratic): pre-2015 = {io_star_pre}, "
                 f"post-2015 = {io_star_post}.")
    if io_star_pre is not None and io_star_post is not None:
        # Stability: IO* within 0.10 of each other?
        if abs(io_star_pre - io_star_post) < 0.10:
            verdict = "STRUCTURAL_IO_STABLE"
        else:
            verdict = "STRUCTURAL_IO_SHIFTS"
    else:
        verdict = "DESCRIPTIVE"
    results['verdict'] = verdict
    results['verdict_notes'] = notes
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42,
                       "B_wcb": B_WCB}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
