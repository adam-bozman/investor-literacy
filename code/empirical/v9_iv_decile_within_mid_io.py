# =====================================================================
# v9_iv_decile_within_mid_io.py
# Decomposes the headline mid-IO three-way coefficient by IV decile (within-mid-IO IV-decile mechanism test) with joint-equality and head-tail spread tests.
#
# Inputs:    _dfm_stable_hq.parquet (via v9_helpers.load_stable_hq); deepen_estimators.twfe_three_way
# Outputs:   output/stage3a/results_v9_iv_decile_within_mid_io.json (printed diagnostics + JSON)
# Paper:     T5 tab:iv_within_mid_io (within-mid-IO IV-decile mechanism) + IA within-retail IV-decile detail
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 5 — IV-decile decomposition WITHIN mid-IO of stable-HQ.

Triage [FIX] Item 8 (Critical). The mechanism referee's identified
discriminator. Within the mid-IO tercile of stable-HQ, decompose the
headline coefficient by IV decile (10 decile-specific gamma values):
  - If concentrated in high-IV deciles -> arbitrage-capacity reading.
  - If diffuse / partially cancelling -> compositional heterogeneity leads.

Procedure (parallels v6/v7 within-low-IO IV-decile decomposition):
  1. Restrict to mid-IO tercile (stable-HQ subsample).
  2. Assign each firm to an IV decile by firm-mean IV (persistent decile).
  3. Within each IV decile, fit the headline three-way TWFE (state+month
     FE, two-way CGM SE + state-clustered CR1).
  4. Report 10 decile gammas + state-clustered t.
  5. Joint test of equality across deciles (Wald or simple F).

For both persistent IO and time-varying IO. Persistent is the canonical
headline.

Output: output/stage3a/results_v9_iv_decile_within_mid_io.json
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
from deepen_estimators import twfe_three_way, FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_iv_decile_within_mid_io.json")


def iv_decile_decomp(d, io_col, label):
    """Decompose the headline three-way by IV decile within the mid-IO
    tercile of stable-HQ."""
    d_io = d[d[io_col].notna()].copy()
    d_io = add_io_terciles(d_io, io_col)
    d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].dropna(
        subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"\n  --- {label}: mid-IO subsample n={len(d_mid):,}, "
          f"{d_mid['permno'].nunique()} permnos ---", flush=True)

    # Assign IV deciles by firm-mean IV (persistent assignment within mid-IO)
    perm_iv = d_mid.groupby('permno')['iv'].mean().dropna()
    iv_dec = pd.qcut(perm_iv, 10, labels=range(1, 11),
                     duplicates='drop')
    d_mid['iv_dec'] = d_mid['permno'].map(iv_dec).astype('object')

    decile_results = []
    for k in range(1, 11):
        sub = d_mid[d_mid['iv_dec'] == k]
        if len(sub) < 100 or sub['hq_state'].nunique() < 3:
            decile_results.append({"decile": k, "skip": "insufficient",
                                   "n_obs": int(len(sub))})
            continue
        r = twfe_three_way(sub)
        decile_results.append({
            "decile": k,
            "n_obs": int(r['n_obs']),
            "n_states": int(sub['hq_state'].nunique()),
            "iv_mean": float(sub['iv'].mean()),
            "gamma": float(r['coef']),
            "se": float(r['se']),
            "t_cgm": float(r['t']),
            "se_state": float(r['se_state']),
            "t_state": float(r['t_state']),
            "se_kind": r['se_kind'],
        })
        print(f"    decile {k}: gamma={r['coef']:+.6f} state_t={r['t_state']:+.2f} "
              f"n={r['n_obs']:,} iv_mean={sub['iv'].mean():.3f}", flush=True)

    # Joint equality test: simple chi-sq using each decile's CR1 SE,
    # H0: all gammas equal. Use weighted-average gamma_bar with weights
    # 1/se_k^2; chi-sq = sum( (gamma_k - gamma_bar)^2 / se_k^2 ), df=9.
    valid = [d for d in decile_results if 'gamma' in d]
    if len(valid) >= 2:
        coefs = np.array([d['gamma'] for d in valid])
        ses = np.array([d['se_state'] for d in valid])
        w = 1.0 / (ses ** 2)
        gbar = (w * coefs).sum() / w.sum()
        chi2 = float(((coefs - gbar) ** 2 / (ses ** 2)).sum())
        df = len(valid) - 1
        from scipy.stats import chi2 as chi2_dist
        p_joint = float(1.0 - chi2_dist.cdf(chi2, df))
        # Also: head-tail spread (decile 10 minus decile 1)
        d1 = next((d for d in valid if d['decile'] == 1), None)
        d10 = next((d for d in valid if d['decile'] == 10), None)
        head_tail = None
        if d1 and d10:
            spread = d10['gamma'] - d1['gamma']
            spread_se = np.sqrt(d1['se_state']**2 + d10['se_state']**2)
            head_tail = {
                "spread_d10_minus_d1": float(spread),
                "se_approx_independent": float(spread_se),
                "t_approx": float(spread / spread_se) if spread_se > 0
                            else None,
            }
        # High-IV vs low-IV (top-3 vs bottom-3)
        top3 = [d for d in valid if d['decile'] in (8, 9, 10)]
        bot3 = [d for d in valid if d['decile'] in (1, 2, 3)]
        if top3 and bot3:
            top_avg = np.mean([d['gamma'] for d in top3])
            bot_avg = np.mean([d['gamma'] for d in bot3])
            top_se = np.sqrt(np.mean([d['se_state']**2 for d in top3]) / 3)
            bot_se = np.sqrt(np.mean([d['se_state']**2 for d in bot3]) / 3)
            spread_se = np.sqrt(top_se**2 + bot_se**2)
            top_bot = {
                "top3_avg_gamma": float(top_avg),
                "bot3_avg_gamma": float(bot_avg),
                "top3_minus_bot3": float(top_avg - bot_avg),
                "se_approx": float(spread_se),
                "t_approx": float((top_avg - bot_avg) / spread_se) if spread_se > 0
                            else None,
            }
        else:
            top_bot = None
        joint = {
            "joint_equality_chi2": chi2,
            "df": df,
            "p_value_joint_equality": p_joint,
            "weighted_avg_gamma": float(gbar),
            "head_tail_spread_d10_d1": head_tail,
            "top3_minus_bot3_spread": top_bot,
        }
    else:
        joint = None

    return {
        "io_col": io_col, "label": label,
        "n_mid_io_firm_months": int(len(d_mid)),
        "n_mid_io_firms": int(d_mid['permno'].nunique()),
        "decile_results": decile_results,
        "joint_tests": joint,
    }


def main():
    t0 = time.time()
    print("=== v9 Test 5: IV-decile decomposition WITHIN mid-IO of "
          "stable-HQ ===", flush=True)
    d = load_stable_hq()
    print(f"  stable-HQ: {len(d):,} firm-months", flush=True)

    results = {
        "test": "v9 Test 5: IV-decile decomposition within mid-IO of stable-HQ",
        "triage_fix": "Item 8 (Critical)",
        "sample": {"stable_hq_firm_months": int(len(d))},
    }

    # Persistent first (canonical)
    results['persistent_IO'] = iv_decile_decomp(
        d, 'io_share_persist', 'persistent_IO')
    # Time-varying second
    results['time_varying_IO'] = iv_decile_decomp(
        d, 'io_share', 'time_varying_IO')

    # Verdict
    p_joint_pers = (results['persistent_IO']['joint_tests']
                    .get('p_value_joint_equality'))
    top_bot_pers = (results['persistent_IO']['joint_tests']
                    .get('top3_minus_bot3_spread'))
    notes = []
    if p_joint_pers is not None:
        notes.append(f"Joint equality across IV deciles (persistent): "
                     f"chi2={results['persistent_IO']['joint_tests']['joint_equality_chi2']:.2f}, "
                     f"df=9, p={p_joint_pers:.4f}.")
    if top_bot_pers is not None:
        notes.append(
            f"Top-3 IV (deciles 8,9,10) average gamma "
            f"= {top_bot_pers['top3_avg_gamma']:+.6f}; "
            f"bottom-3 IV average gamma = "
            f"{top_bot_pers['bot3_avg_gamma']:+.6f}; "
            f"spread = {top_bot_pers['top3_minus_bot3']:+.6f} "
            f"(t~{top_bot_pers['t_approx']}).")

    # The mechanism question: is the headline mid-IO concentration concentrated
    # at high-IV (arbitrage-capacity reading) or diffuse/cancelling (compositional)?
    # Compositional reading: spread small relative to average magnitude, joint
    # equality fails to reject. Arbitrage-capacity reading: high-IV deciles much
    # more negative.
    if top_bot_pers is not None and top_bot_pers['top3_minus_bot3'] < -0.02:
        verdict = "ARBITRAGE_CAPACITY_READING"
    elif p_joint_pers and p_joint_pers > 0.10:
        verdict = "DIFFUSE_COMPOSITIONAL"
    else:
        verdict = "MIXED"
    results['verdict'] = verdict
    results['verdict_notes'] = notes
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
