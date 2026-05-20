# =====================================================================
# v9_variance_channel_diagnostic.py
# Tests whether mid-IO concentration is a coefficient vs precision phenomenon by comparing within-tercile focal-triple variance (Bartlett/Levene).
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); deepen_estimators.FOCAL
# Outputs:   output/stage3a/results_v9_variance_channel_diagnostic.json (printed diagnostics + JSON)
# Paper:     IA mechanism / variance-channel diagnostic
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 10 — Variance-channel-vs-coefficient-channel diagnostic.

Triage [FIX] Item 17 (High). Within each IO tercile of stable-HQ, compute
the within-tercile variance of (lit_z * IV * mom). If mid-IO has LOWER
variance, the mid-IO concentration is a coefficient phenomenon (real
estimate is more negative). If mid-IO has HIGHER design-matrix variance,
the mid-IO concentration could be a precision phenomenon (more variation
to identify the slope, so SE is smaller and significance is mechanical).

Report:
  - Var(triple) by tercile (and SD)
  - Mean and variance of each focal component (mom, IV, lit_z) by tercile
  - Bartlett's test of equal variance across the three terciles
  - Levene's test (more robust to non-normality) of equal variance

Output: output/stage3a/results_v9_variance_channel_diagnostic.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import bartlett, levene

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import load_stable_hq, add_io_terciles, save_json, OUT
from deepen_estimators import FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_variance_channel_diagnostic.json")


def main():
    t0 = time.time()
    print("=== v9 Test 10: Variance-channel diagnostic ===", flush=True)
    d = load_stable_hq()
    print(f"  stable-HQ: {len(d):,} firm-months", flush=True)

    results = {
        "test": "v9 Test 10: Variance-channel diagnostic on stable-HQ",
        "triage_fix": "Item 17 (High)",
    }

    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n--- {label} ({io_col}) ---", flush=True)
        d_io = d[d[io_col].notna()].copy()
        d_io = add_io_terciles(d_io, io_col)
        d_io = d_io.dropna(subset=FOCAL + ['ret']).copy()

        # focal three-way already in panel
        triples = []
        out = {}
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            triple = sub['mom_x_iv_x_literacy_corr'].dropna().values
            triples.append(triple)
            stats = {
                "n_obs": int(len(triple)),
                "mean_triple": float(np.mean(triple)),
                "var_triple": float(np.var(triple, ddof=1)),
                "sd_triple": float(np.std(triple, ddof=1)),
                "min_triple": float(np.min(triple)),
                "max_triple": float(np.max(triple)),
                "var_mom": float(sub['mom_12_2'].var(ddof=1)),
                "var_iv": float(sub['iv'].var(ddof=1)),
                "var_lit": float(sub['literacy_score_corrected'].var(ddof=1)),
            }
            out[g] = stats
            print(f"    {g}: var(triple)={stats['var_triple']:.4f}, "
                  f"sd(triple)={stats['sd_triple']:.4f}, "
                  f"n={stats['n_obs']:,}", flush=True)

        # Bartlett's and Levene's tests
        try:
            b_stat, b_p = bartlett(*triples)
            print(f"    Bartlett: stat={b_stat:.3f}, p={b_p:.4f}",
                  flush=True)
            out['bartlett_test'] = {
                "statistic": float(b_stat), "p_value": float(b_p),
                "h0": "equal variance across terciles",
            }
        except Exception as e:
            out['bartlett_test'] = {"error": str(e)}

        try:
            # Use median-centered Levene (robust)
            l_stat, l_p = levene(*triples, center='median')
            print(f"    Levene (median): stat={l_stat:.3f}, p={l_p:.4f}",
                  flush=True)
            out['levene_test'] = {
                "statistic": float(l_stat), "p_value": float(l_p),
                "h0": "equal variance across terciles",
            }
        except Exception as e:
            out['levene_test'] = {"error": str(e)}

        # Compute the variance ratio mid:(low+high)/2
        var_mid = out['IO2_mid']['var_triple']
        var_low = out['IO1_low']['var_triple']
        var_high = out['IO3_high']['var_triple']
        avg_extreme = 0.5 * (var_low + var_high)
        ratio = var_mid / avg_extreme if avg_extreme > 0 else None
        out['variance_ratio_mid_to_avg_extreme'] = (
            float(ratio) if ratio is not None else None)
        print(f"    var(mid) / avg(var(low), var(high)) = {ratio}",
              flush=True)
        out['interpretation'] = (
            "If variance_ratio_mid_to_avg_extreme < 1.0 (mid has LOWER "
            "variance of the focal triple), the mid-IO concentration is a "
            "COEFFICIENT phenomenon (mid-IO has less identifying variation "
            "but a larger coef in magnitude). If > 1.0, mid-IO has MORE "
            "identifying variation, and the mid-IO concentration could be "
            "a PRECISION phenomenon."
        )
        results[label] = out

    # Verdict
    pers_ratio = results['persistent_IO']['variance_ratio_mid_to_avg_extreme']
    if pers_ratio is None:
        verdict = "DESCRIPTIVE"
    elif pers_ratio < 0.85:
        verdict = "COEFFICIENT_PHENOMENON"
    elif pers_ratio > 1.15:
        verdict = "PRECISION_PHENOMENON"
    else:
        verdict = "VARIANCE_APPROXIMATELY_EQUAL"
    results['verdict'] = verdict
    results['verdict_notes'] = [
        f"Persistent IO: var(mid) / avg(var(low), var(high)) = "
        f"{pers_ratio}; mid-IO mean(triple) = "
        f"{results['persistent_IO']['IO2_mid']['mean_triple']:.4f}.",
        results['persistent_IO']['interpretation'],
    ]
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
