# =====================================================================
# v9_quadratic_io_stable_hq.py
# Re-estimates the quadratic-IO four-way on the stable-HQ subsample (resolving full-panel vs stable-HQ cross-sample mixing) and recovers IO*.
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); deepen_estimators, v8_diagnostic_three._quadratic_four_way
# Outputs:   output/stage3a/results_v9_quadratic_io_stable_hq.json (printed diagnostics + JSON)
# Paper:     T7 tab:form_grid (quadratic) + IA 20-bin quadratic section
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 4 — Quadratic-IO four-way on stable-HQ subsample.

Triage [FIX] Item 3 (Critical). The structured referee identified that
prior Table 3 estimated the quadratic-IO four-way on the FULL panel, while
the headline Table 2 was on the stable-HQ subsample (519k firm-months).
This is cross-sample mixing. Re-estimate the quadratic-IO four-way on the
stable-HQ subsample under static-snapshot literacy.

Report gamma_3 (linear), gamma_4 (linear x IO interaction not strictly here
- we follow v8 _quadratic_four_way convention), gamma_4 = `mom*IV*lit*IO`
linear, gamma_5 = `mom*IV*lit*IO^2`, both with state-clustered t and WCB p.
Compute IO* = -gamma_4 / (2*gamma_5). Compare to full-panel IO* ~= 0.30.

Output: output/stage3a/results_v9_quadratic_io_stable_hq.json
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
from v9_helpers import load_stable_hq, save_json, OUT
from v8_diagnostic_three import _quadratic_four_way
from deepen_estimators import FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_quadratic_io_stable_hq.json")
B_WCB = 4999


def main():
    t0 = time.time()
    print("=== v9 Test 4: Quadratic-IO four-way on stable-HQ ===",
          flush=True)
    d = load_stable_hq()
    d = d.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  stable-HQ estimation sample: {len(d):,} firm-months, "
          f"{d['permno'].nunique()} permnos, "
          f"{d['hq_state'].nunique()} states", flush=True)

    results = {
        "test": "v9 Test 4: Quadratic-IO four-way on stable-HQ subsample",
        "triage_fix": "Item 3 (Critical)",
        "sample": {
            "n_firm_months": int(len(d)),
            "n_firms": int(d['permno'].nunique()),
            "n_states": int(d['hq_state'].nunique()),
        },
        "reference_full_panel_v8": {
            "io_star_full_panel_persistent": "~0.30 (from v8 Table 3)",
            "note": "v8 result was on the full v7 panel (623,896 firm-months); "
                    "v9 re-estimates on the stable-HQ subsample (519,090) — "
                    "the headline population.",
        },
    }

    for io_col, label, seed in [('io_share_persist', 'persistent_IO', 401),
                                ('io_share', 'time_varying_IO', 402)]:
        print(f"\n--- {label} ({io_col}) ---", flush=True)
        d_io = d[d[io_col].notna()].copy()
        print(f"  n with IO: {len(d_io):,}", flush=True)
        r = _quadratic_four_way(d_io, io_col, label=label,
                                B=B_WCB, seed=seed)
        print(f"  gamma_4 (linear) = {r['gamma_4_linear']:+.6f} "
              f"(state-t={r['t_state_clustered_linear']:.2f}, "
              f"wcb-p={r['wcb_p_value_linear']:.4f})", flush=True)
        print(f"  gamma_5 (quadratic) = {r['gamma_5_quadratic']:+.6f} "
              f"(state-t={r['t_state_clustered_quadratic']:.2f}, "
              f"wcb-p={r['wcb_p_value_quadratic']:.4f})", flush=True)
        if r['implied_io_turning_point'] is not None:
            print(f"  IO* = -gamma_4/(2*gamma_5) = "
                  f"{r['implied_io_turning_point']:.4f}", flush=True)
        results[label] = r

    # Verdict: does the quadratic mid-IO peak survive on stable-HQ?
    p_quad_pers = results['persistent_IO']['wcb_p_value_quadratic']
    g5_pers = results['persistent_IO']['gamma_5_quadratic']
    io_star_pers = results['persistent_IO']['implied_io_turning_point']

    # Slope of three-way wrt IO = g4 + 2*g5*IO. Mid-IO trough (most-negative
    # three-way at interior IO) requires g4 < 0 AND g5 > 0 (slope curve opens
    # upward, with interior minimum where it crosses zero).
    g4_pers = results['persistent_IO']['gamma_4_linear']
    trough = g4_pers < 0 and g5_pers > 0
    if trough and p_quad_pers < 0.05:
        verdict = "CONFIRMS_5PCT"
    elif trough and p_quad_pers < 0.10:
        verdict = "CONFIRMS_10PCT"
    elif trough:
        verdict = "WEAKENS_DIRECTION_OK_NOT_SIG"
    else:
        verdict = "WEAKENS_SIGN_FLIP"
    results['verdict'] = verdict
    results['verdict_notes'] = [
        f"Persistent gamma_5 = {g5_pers:.6f}, wcb-p = {p_quad_pers:.4f}, "
        f"IO* = {io_star_pers}.",
        "Stable-HQ test addresses the cross-sample-mixing concern raised in "
        "structured referee C3.",
    ]
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42,
                       "B_wcb": B_WCB}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
