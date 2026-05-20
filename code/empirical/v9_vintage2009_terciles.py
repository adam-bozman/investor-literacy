# =====================================================================
# v9_vintage2009_terciles.py
# Re-fits the headline three-way using IO terciles fixed at their 2009-vintage assignment (robustness against secular IO drift), with tercile-drift crosstab.
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); deepen_estimators.twfe_three_way
# Outputs:   output/stage3a/results_v9_vintage2009_terciles.json (printed diagnostics + JSON)
# Paper:     IA robustness — 2009-vintage / fixed terciles
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 3 — 2009-vintage terciles falsification.

Triage [FIX] Item 6c (Critical). Re-fit the three-way on the stable-HQ
subsample but assign IO terciles ONLY by their 2009-Q1 (first-six-month
2009 mean) IO value, held fixed across the panel rather than time-varying.

If the secular-IO-shift reading is correct (IO has drifted up over time,
moving firms across terciles), the IO* / mid-IO concentration pattern
should DRIFT under fixed 2009-vintage terciles: firms classified mid-IO in
2009 may now be high-IO; mid-IO concentration could either move or
attenuate, but the location should drift.

Output: output/stage3a/results_v9_vintage2009_terciles.json
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
from v9_helpers import (load_stable_hq, add_io_terciles,
                        add_io_terciles_2009, save_json, OUT)
from deepen_estimators import twfe_three_way, FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_vintage2009_terciles.json")


def headline_by_tercile(d, label, tercile_col):
    """Headline three-way per tercile, using a given tercile column."""
    d_all = d.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    out = {"label": label, "tercile_col": tercile_col,
           "estimation_sample_n": int(len(d_all)), "tercile_results": {}}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = d_all[d_all[tercile_col] == g]
        if len(sub) < 100 or sub['hq_state'].nunique() < 3:
            out['tercile_results'][g] = {"skip": "insufficient",
                                          "n_obs": int(len(sub))}
            continue
        r = twfe_three_way(sub)
        out['tercile_results'][g] = {
            k: r[k] for k in ['coef', 'se', 't', 'se_state', 't_state',
                              'n_obs', 'se_kind']}
        print(f"    {label} {g}: gamma={r['coef']:+.6f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}", flush=True)
    # Mid/low ratio
    try:
        mid = out['tercile_results']['IO2_mid'].get('coef')
        low = out['tercile_results']['IO1_low'].get('coef')
        out['mid_low_ratio'] = float(mid / low) if low and mid else None
    except Exception:
        out['mid_low_ratio'] = None
    return out


def crosstab_tercile_drift(d, current_col, vintage_col):
    """How many firms in each (vintage tercile, current tercile) cell?
    Diagnoses how much firms have drifted across terciles."""
    perm = (d[[current_col, vintage_col, 'permno']]
            .drop_duplicates(subset='permno'))
    return pd.crosstab(perm[vintage_col], perm[current_col]).to_dict()


def main():
    t0 = time.time()
    print("=== v9 Test 3: 2009-vintage terciles falsification ===",
          flush=True)
    d_stable = load_stable_hq()
    print(f"  stable-HQ: {len(d_stable):,} firm-months", flush=True)

    results = {
        "test": "v9 Test 3: 2009-vintage terciles on stable-HQ",
        "triage_fix": "Item 6c (Critical)",
        "sample": {
            "stable_hq_firm_months": int(len(d_stable)),
            "stable_hq_firms": int(d_stable['permno'].nunique()),
        },
    }

    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n--- {label} ({io_col}) ---", flush=True)
        d_io = d_stable[d_stable[io_col].notna()].copy()
        # Add both time-varying (default) and 2009-vintage terciles
        d_io = add_io_terciles(d_io, io_col, 'io_grp_tv')
        d_io = add_io_terciles_2009(d_io, io_col, 'io_grp_2009')
        n_with_2009 = d_io['io_grp_2009'].notna().sum()
        print(f"  firm-months with 2009 vintage assignment: "
              f"{n_with_2009:,}/{len(d_io):,}", flush=True)
        n_with_tv = d_io['io_grp_tv'].notna().sum()
        print(f"  firm-months with time-varying assignment: "
              f"{n_with_tv:,}", flush=True)

        # Headline by both classification schemes
        print("\n  --- TIME-VARYING tercile classification (canonical) ---",
              flush=True)
        hl_tv = headline_by_tercile(d_io, "stable_HQ_TV", "io_grp_tv")
        print("\n  --- 2009-VINTAGE FIXED tercile classification ---",
              flush=True)
        hl_2009 = headline_by_tercile(d_io, "stable_HQ_2009vintage",
                                      "io_grp_2009")

        # Drift table: how many firms moved across terciles?
        drift = crosstab_tercile_drift(d_io, 'io_grp_tv', 'io_grp_2009')
        results[label] = {
            "time_varying_terciles": hl_tv,
            "vintage_2009_terciles": hl_2009,
            "tercile_drift_perm_count_2009_to_tv": drift,
            "n_firms_with_2009_assignment": int(
                d_io.dropna(subset=['io_grp_2009'])['permno'].nunique()),
        }
        print(f"\n  Mid/low ratio under time-varying: {hl_tv['mid_low_ratio']}",
              flush=True)
        print(f"  Mid/low ratio under 2009-vintage:  "
              f"{hl_2009['mid_low_ratio']}", flush=True)
        print(f"  Tercile-drift diagonal vs off-diagonal (2009->TV): "
              f"{drift}", flush=True)

    # ===========================================================
    # Verdict
    # ===========================================================
    # If under 2009-vintage terciles the mid-IO concentration ATTENUATES
    # substantially (mid/low ratio drops toward 1.0), that supports the
    # secular-IO-shift reading. If it stays similar, the structural reading
    # holds.
    tv_pers_ratio = results['persistent_IO']['time_varying_terciles']['mid_low_ratio']
    v_pers_ratio = results['persistent_IO']['vintage_2009_terciles']['mid_low_ratio']
    notes = []
    notes.append(f"Persistent IO: time-varying mid/low ratio "
                 f"= {tv_pers_ratio}, vintage-2009 mid/low ratio = "
                 f"{v_pers_ratio}.")
    if tv_pers_ratio is not None and v_pers_ratio is not None:
        if abs(v_pers_ratio - 1.0) < 0.5 * abs(tv_pers_ratio - 1.0):
            verdict = "WEAKENS_SECULAR_SHIFT"
        elif abs(v_pers_ratio - tv_pers_ratio) / abs(tv_pers_ratio) < 0.3:
            verdict = "CONFIRMS_STRUCTURAL"
        else:
            verdict = "MIXED"
    else:
        verdict = "DESCRIPTIVE"
    results['verdict'] = verdict
    results['verdict_notes'] = notes
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
