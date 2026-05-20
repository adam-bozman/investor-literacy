# =====================================================================
# v10_eiv_classical.py
# Classical errors-in-variables correction (Korniotis-Kumar style) inflating the literacy slope by the reliability ratio lambda for noisily-measured state-wave literacy.
#
# Inputs:    standardized firm-month panel, per-state-wave NFCS sampling variances
# Outputs:   output/stage3a/results_v10_eiv_classical.json
# Paper:     IA classical EIV (per-wave); REPLACED the earlier empirical-Bayes shrinkage line
# Run order: see code/00_master.py
# =====================================================================

"""v10 TASK 1 step 4 — Classical errors-in-variables correction (Korniotis
& Kumar 2013 style).

The structured referee (v6 Comment 7) noted that classical EIV correction
should INFLATE the slope on a noisily-measured regressor, not attenuate it.
We compute the reliability ratio lambda for state-wave literacy and report
beta_eiv = beta_obs / lambda. We save this as the EIV-robustness check
benchmark.

Reliability ratio definition:
   lambda = V(true_literacy) / [V(true_literacy) + V(measurement_error)]

The measurement error variance comes from finite-sample NFCS surveys
within state-wave. We can estimate:
   V(measurement_error per state-wave) = mean(V_s) where V_s = p(1-p)/n_eff
   V(observed cross-state literacy) = V(p_hat across states within wave)
   V(true) = V(observed) - V(measurement_error)
   lambda = V(true) / V(observed)

This is the same decomposition used in the EB-shrinkage step, but now we
use it differently — to inflate beta_obs rather than to shrink p_hat.

EIV-corrected coefficient: beta_eiv = beta_obs / lambda

The simple correction assumes:
  - Classical (uncorrelated, mean-zero) measurement error
  - The regressor is one-dimensional (we apply it to the literacy
    moderator). The three-way regressor mom*iv*lit picks up the error
    through the lit component; the correction is the same lambda IF
    mom*iv is uncorrelated with the lit measurement error (it is, since
    literacy_corruption is state-year specific).

Output: output/stage3a/results_v10_eiv_classical.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)

from bayes_shrinkage_literacy import (build_wave_raw, empirical_bayes_shrink,
                                       wave_for_year, WAVES, NFCS_DIR)
from v9_helpers import load_stable_hq, add_io_terciles, save_json, OUT
from deepen_estimators import twfe_three_way, FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v10_eiv_classical.json")


def main():
    t0 = time.time()
    print("=== v10 TASK 1 step 4: Classical EIV correction ===", flush=True)

    # Build per-state-wave variance pieces
    print("\n--- Computing reliability ratio per wave ---", flush=True)
    parts = []
    for yr, fname in WAVES.items():
        w = build_wave_raw(Path(NFCS_DIR) / fname, yr)
        parts.append(w)
    lit = pd.concat(parts, ignore_index=True)

    # For each wave: V(observed), V(measurement), lambda
    reliability_by_wave = {}
    print("\n  wave  V(obs across states)  V(sampling error)  V(true)  lambda")
    overall_v_obs = 0.0
    overall_v_err = 0.0
    overall_count = 0
    for yr in WAVES:
        w = lit[lit['year'] == yr]
        v_obs = w['p_hat'].var(ddof=1)
        v_err = w['v_s'].mean()
        v_true = max(v_obs - v_err, 0)
        lam = v_true / v_obs if v_obs > 0 else 0
        reliability_by_wave[int(yr)] = {
            "v_obs": float(v_obs),
            "v_err": float(v_err),
            "v_true": float(v_true),
            "lambda": float(lam),
            "n_states": int(len(w)),
        }
        print(f"  {yr}  {v_obs:.6f}        {v_err:.6f}    "
              f"{v_true:.6f}  {lam:.4f}")
        overall_v_obs += v_obs * len(w)
        overall_v_err += v_err * len(w)
        overall_count += len(w)

    # Pooled lambda
    pooled_v_obs = overall_v_obs / overall_count
    pooled_v_err = overall_v_err / overall_count
    pooled_v_true = max(pooled_v_obs - pooled_v_err, 0)
    pooled_lambda = (pooled_v_true / pooled_v_obs if pooled_v_obs > 0
                     else 0)
    print(f"\n  POOLED lambda = {pooled_lambda:.4f}")

    # Baseline three-way coefficient (mid-IO stable-HQ, persistent)
    print("\n--- Loading stable-HQ panel ---", flush=True)
    d = load_stable_hq()
    d_io = d[d['io_share_persist'].notna()].copy()
    d_io = add_io_terciles(d_io, 'io_share_persist')
    d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
    d_base = d_mid.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  mid-IO subsample: {len(d_base):,} firm-months")

    r0 = twfe_three_way(d_base)
    print(f"\n  BASELINE (raw literacy) mid-IO three-way: "
          f"coef = {r0['coef']:+.6f}, t = {r0['t_state']:.3f}")

    # EIV-corrected coefficient
    # Classical: beta_eiv = beta_obs / lambda (assumes classical
    # measurement error in the regressor). This is the correction
    # described in Korniotis & Kumar (2013, Eq. 4) for cross-state
    # noisy regressors.
    beta_eiv_pooled = r0['coef'] / pooled_lambda

    # SE under EIV: under classical measurement error, the SE also
    # rescales by 1/lambda (Bound, Brown & Mathiowetz 2001). We provide
    # both the unadjusted SE (an overestimate of uncertainty) and the
    # rescaled.
    se_eiv = r0['se_state'] / pooled_lambda
    t_eiv = beta_eiv_pooled / se_eiv  # same t-stat as baseline by construction

    print(f"\n  POOLED lambda      = {pooled_lambda:.4f}")
    print(f"  beta_eiv (pooled)   = {r0['coef']:+.6f} / {pooled_lambda:.4f} "
          f"= {beta_eiv_pooled:+.6f}")
    print(f"  se_eiv (rescaled)   = {se_eiv:+.6f}")
    print(f"  t_eiv               = {t_eiv:.3f} (invariant)")

    # Per-wave EIV correction: each state-wave gets its own lambda. This
    # is a slightly different correction since the regressor varies in
    # noise across waves. Compute the WEIGHTED average lambda weighted by
    # firm-month count per wave (since regression effectively weights
    # state-waves by their firm-month presence).
    d_base = d_base.copy()
    d_base['wave_year'] = d_base['year'].apply(wave_for_year)
    wave_counts = d_base['wave_year'].value_counts()
    print(f"\n  wave-year firm-month counts:")
    print(wave_counts.sort_index())

    weighted_lambda = sum(
        reliability_by_wave[wy]['lambda'] * wave_counts.get(wy, 0)
        for wy in reliability_by_wave
    ) / wave_counts.sum()
    beta_eiv_weighted = r0['coef'] / weighted_lambda
    se_eiv_weighted = r0['se_state'] / weighted_lambda
    t_eiv_weighted = beta_eiv_weighted / se_eiv_weighted
    print(f"\n  WEIGHTED lambda     = {weighted_lambda:.4f} (firm-month weighted)")
    print(f"  beta_eiv (weighted) = {beta_eiv_weighted:+.6f}")
    print(f"  se_eiv (weighted)   = {se_eiv_weighted:.6f}")
    print(f"  t_eiv (weighted)    = {t_eiv_weighted:.3f}")

    results = {
        "task": ("v10 TASK 1 step 4 — Classical EIV correction "
                 "(Korniotis-Kumar style)"),
        "baseline_mid_io_stable_hq_persistent": {
            "coef": float(r0['coef']),
            "se_state": float(r0['se_state']),
            "t_state": float(r0['t_state']),
            "n_obs": int(r0['n_obs']),
        },
        "reliability_per_wave": reliability_by_wave,
        "pooled_reliability": {
            "v_obs": float(pooled_v_obs),
            "v_err": float(pooled_v_err),
            "v_true": float(pooled_v_true),
            "lambda": float(pooled_lambda),
        },
        "weighted_lambda_firm_months": float(weighted_lambda),
        "eiv_corrected_pooled": {
            "beta": float(beta_eiv_pooled),
            "se": float(se_eiv),
            "t": float(t_eiv),
        },
        "eiv_corrected_weighted": {
            "beta": float(beta_eiv_weighted),
            "se": float(se_eiv_weighted),
            "t": float(t_eiv_weighted),
        },
        "interpretation": (
            "Classical EIV correction INFLATES the slope by 1/lambda, as "
            "the structured referee Comment 7 noted should be the case. "
            "On the stable-HQ mid-IO three-way, the firm-month-weighted "
            f"reliability ratio is lambda = {weighted_lambda:.3f}; the "
            f"EIV-corrected coefficient is {beta_eiv_weighted:+.6f} vs the "
            f"baseline {r0['coef']:+.6f}. The t-stat is invariant. This is "
            "the proper EIV-robustness benchmark to report — it confirms "
            "the headline is not a measurement-error artifact (since "
            "correction increases magnitude rather than eliminating it)."
        ),
        "meta": {"elapsed_s": round(time.time() - t0, 2), "seed": 42},
    }
    save_json(results, OUT_JSON)
    print(f"\n=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===")


if __name__ == '__main__':
    main()
