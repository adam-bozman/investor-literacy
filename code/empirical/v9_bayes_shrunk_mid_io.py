# =====================================================================
# v9_bayes_shrunk_mid_io.py
# Re-fits the mid-IO three-way under empirical-Bayes-shrunk literacy with Bayesian credible intervals (measurement-error robustness).
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); NFCS waves via bayes_shrinkage_literacy; deepen_estimators
# Outputs:   output/stage3a/results_v9_bayes_shrunk_mid_io.json (printed diagnostics + JSON)
# Paper:     IA classical EIV / Bayes section (note: Bayes shrinkage was later replaced by classical EIV in v10)
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 6 — Bayes-shrunk headline mid-IO coefficient on stable-HQ.

Triage [FIX] Item 10 (Critical). Re-fit the mid-IO three-way regression
under empirical-Bayes-shrunk literacy on the stable-HQ subsample only.
Report posterior mean and 90/95% credible intervals using a Bayesian
linear regression around the shrunk-literacy specification.

Approach:
  1. Reproduce empirical-Bayes-shrunk literacy by state-wave (same as
     existing bayes_shrinkage_literacy.py).
  2. Re-merge into the stable-HQ panel; rebuild three-way using shrunk
     literacy; z-score within month.
  3. Restrict to mid-IO tercile of stable-HQ.
  4. Run TWFE three-way: report coef, state-cluster SE, t.
  5. For Bayesian credible intervals: use a simple G-prior / weakly
     informative normal posterior using the GLS / asymptotic-normal point
     estimate and SE. Posterior ~ N(coef, SE) under flat prior; CI
     constructed from quantiles.

Output: output/stage3a/results_v9_bayes_shrunk_mid_io.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import (load_stable_hq, add_io_terciles, save_json, OUT)
from deepen_estimators import twfe_three_way, wild_cluster_bootstrap_state

# Shrinkage builder from existing script
from bayes_shrinkage_literacy import (build_wave_raw, empirical_bayes_shrink,
                                       wave_for_year, WAVES,
                                       CORRECT_CODES, NFCS_DIR)

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_bayes_shrunk_mid_io.json")
B_WCB = 4999


def build_shrunk_lit():
    """Build empirical-Bayes-shrunk literacy by (state, wave year) and
    return a DataFrame with state, year, p_hat, p_shrunk columns."""
    parts = []
    for yr, fname in WAVES.items():
        w = build_wave_raw(Path(NFCS_DIR) / fname, yr)
        w = empirical_bayes_shrink(w)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    print("=== v9 Test 6: Bayes-shrunk mid-IO on stable-HQ ===",
          flush=True)

    print("\n  --- Building empirical-Bayes-shrunk literacy (5 waves) ---",
          flush=True)
    lit = build_shrunk_lit()
    print(f"  state-wave records: {len(lit)}", flush=True)

    print("\n  --- Loading stable-HQ panel ---", flush=True)
    d_stable = load_stable_hq()
    d_stable['wave_year'] = d_stable['year'].apply(wave_for_year)
    print(f"  stable-HQ: {len(d_stable):,} firm-months", flush=True)

    # Merge shrunk literacy on (state, wave_year)
    lit_map = lit.set_index(['state', 'year'])[['p_hat', 'p_shrunk']]
    d_stable = d_stable.merge(
        lit_map.rename(columns={'p_hat': 'lit_raw',
                                'p_shrunk': 'lit_shrunk'}),
        left_on=['hq_state', 'wave_year'], right_index=True, how='left')
    print(f"  after merge: lit_raw missing {d_stable['lit_raw'].isna().sum()}, "
          f"lit_shrunk missing {d_stable['lit_shrunk'].isna().sum()}",
          flush=True)

    # z-score shrunk literacy within month, rebuild three-way
    d = d_stable.dropna(subset=['lit_shrunk', 'mom_12_2', 'iv', 'ret',
                                 'hq_state', 'mom_x_iv']).copy()
    d['lit_shrunk_z'] = d.groupby('ym')['lit_shrunk'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    d['mom_x_lit_shrunk'] = d['mom_12_2'] * d['lit_shrunk_z']
    d['iv_x_lit_shrunk'] = d['iv'] * d['lit_shrunk_z']
    d['mom_x_iv_x_lit_shrunk'] = d['mom_x_iv'] * d['lit_shrunk_z']
    # Re-z-score the three composites within month (to match canonical conv)
    for c in ['mom_x_lit_shrunk', 'iv_x_lit_shrunk', 'mom_x_iv_x_lit_shrunk',
              'lit_shrunk_z']:
        d[c] = d.groupby('ym')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    print(f"  estimation panel: {len(d):,} firm-months, "
          f"{d['permno'].nunique()} permnos", flush=True)

    # Substitute shrunk-literacy variables into the canonical FOCAL order
    # by renaming the standard literacy-three-way names.
    d['literacy_score_corrected_orig'] = d['literacy_score_corrected']
    d['mom_x_literacy_corr_orig'] = d['mom_x_literacy_corr']
    d['iv_x_literacy_corr_orig'] = d['iv_x_literacy_corr']
    d['mom_x_iv_x_literacy_corr_orig'] = d['mom_x_iv_x_literacy_corr']
    d['literacy_score_corrected'] = d['lit_shrunk_z']
    d['mom_x_literacy_corr'] = d['mom_x_lit_shrunk']
    d['iv_x_literacy_corr'] = d['iv_x_lit_shrunk']
    d['mom_x_iv_x_literacy_corr'] = d['mom_x_iv_x_lit_shrunk']

    results = {
        "test": "v9 Test 6: Bayes-shrunk mid-IO on stable-HQ",
        "triage_fix": "Item 10 (Critical)",
        "config": {
            "n_state_waves": int(len(lit)),
            "mean_shrink_weight": float(lit['shrink_weight'].mean()),
            "median_n_eff": float(lit['n_eff'].median()),
            "raw_vs_shrunk_corr": float(lit['p_hat'].corr(lit['p_shrunk'])),
        },
        "sample": {
            "stable_hq_panel_n": int(len(d_stable)),
            "estimation_n_after_merge": int(len(d)),
        },
    }

    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n  --- {label} ({io_col}) ---", flush=True)
        d_io = d[d[io_col].notna()].copy()
        d_io = add_io_terciles(d_io, io_col)

        # First: ALSO report the AGGREGATE (across all three terciles) on
        # the shrunk-literacy panel. Mirror bayes_shrinkage_literacy.py's
        # original aggregate spec for comparison.
        d_agg = d_io.copy()
        r_agg = twfe_three_way(d_agg)
        print(f"    AGGREGATE (all terciles, shrunk literacy): "
              f"gamma={r_agg['coef']:+.6f} state_t={r_agg['t_state']:.2f} "
              f"n={r_agg['n_obs']:,}", flush=True)

        # Mid-IO
        d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
        if len(d_mid) < 100 or d_mid['hq_state'].nunique() < 3:
            results[label] = {"skip": "insufficient mid-IO sample"}
            continue
        r = twfe_three_way(d_mid)
        print(f"    mid-IO shrunk: gamma={r['coef']:+.6f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}", flush=True)
        # WCB for mid-IO
        wcb = wild_cluster_bootstrap_state(d_mid, B=B_WCB, seed=42)
        print(f"    mid-IO WCB p={wcb['p_value']:.4f}", flush=True)

        # Bayesian credible intervals: posterior ~ N(coef, se_state) under
        # flat prior on coef + Gaussian likelihood. CIs from normal quantiles.
        coef = r['coef']
        se = r['se_state']
        ci_90 = (coef + norm.ppf(0.05) * se, coef + norm.ppf(0.95) * se)
        ci_95 = (coef + norm.ppf(0.025) * se, coef + norm.ppf(0.975) * se)
        # Bootstrap-CI: from WCB studentized
        ci_wcb = wcb['ci_studentized']

        # Also: same for low and high IO terciles
        results[label] = {
            "aggregate_shrunk": {
                "coef": float(r_agg['coef']),
                "se_state": float(r_agg['se_state']),
                "t_state": float(r_agg['t_state']),
                "n_obs": int(r_agg['n_obs']),
                "se_kind": r_agg['se_kind'],
            },
            "mid_io_shrunk": {
                "coef": float(coef),
                "se": float(r['se']),
                "t_cgm": float(r['t']),
                "se_state": float(se),
                "t_state": float(r['t_state']),
                "n_obs": int(r['n_obs']),
                "se_kind": r['se_kind'],
                "wcb_p_value": float(wcb['p_value']),
                "wcb_ci_studentized": [float(x) for x in ci_wcb],
                "bayes_posterior_mean": float(coef),
                "bayes_posterior_sd": float(se),
                "bayes_credible_interval_90": [float(ci_90[0]),
                                                float(ci_90[1])],
                "bayes_credible_interval_95": [float(ci_95[0]),
                                                float(ci_95[1])],
            },
        }

        # Now: low and high IO terciles for context
        for g in ['IO1_low', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            if len(sub) < 100 or sub['hq_state'].nunique() < 3:
                continue
            rg = twfe_three_way(sub)
            results[label][f"{g}_shrunk"] = {
                "coef": float(rg['coef']),
                "se_state": float(rg['se_state']),
                "t_state": float(rg['t_state']),
                "n_obs": int(rg['n_obs']),
            }
            print(f"    {g} shrunk: gamma={rg['coef']:+.6f} "
                  f"state_t={rg['t_state']:.2f} n={rg['n_obs']:,}",
                  flush=True)

    # ============================================================
    # Verdict
    # ============================================================
    pers_mid = results.get('persistent_IO', {}).get('mid_io_shrunk', {})
    coef_pers_mid = pers_mid.get('coef')
    p_pers_mid = pers_mid.get('wcb_p_value')

    # Reference: baseline raw-literacy mid-IO state_t from v8 was ~-2.0 with
    # coef -0.032 (mid-IO on stable-HQ persistent). The aggregate shrink was
    # delta -0.011 -> -0.0009. How much does the mid-IO coef attenuate?
    aggregate_baseline = -0.011726888514516753
    aggregate_shrunk = -0.0009038086669183914
    notes = []
    if coef_pers_mid is not None:
        ratio = coef_pers_mid / (-0.032064)
        notes.append(f"Persistent mid-IO baseline (raw lit) = -0.032064 "
                     f"(state-t -2.63); shrunk = {coef_pers_mid:+.6f} "
                     f"(state-t={pers_mid['t_state']:.2f}); ratio = "
                     f"{ratio:.3f}.")
        notes.append(f"Aggregate baseline -> shrunk attenuation: "
                     f"{aggregate_baseline:+.6f} -> {aggregate_shrunk:+.6f} "
                     f"(ratio = "
                     f"{aggregate_shrunk/aggregate_baseline:.3f}). "
                     f"Mid-IO attenuation ratio compared.")
    if (p_pers_mid is not None and pers_mid.get('coef', 0) < 0
            and p_pers_mid < 0.10):
        verdict = "ROBUST"
    elif pers_mid.get('coef', 0) < 0:
        verdict = "ATTENUATED_DIRECTION_OK"
    else:
        verdict = "FLIPPED"
    results['verdict'] = verdict
    results['verdict_notes'] = notes
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
