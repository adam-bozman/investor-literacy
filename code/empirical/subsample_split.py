# =====================================================================
# subsample_split.py
# Decisive pre/post-2015 subsample test of the headline TWFE-CGM and FF12-absorbed specs to check whether the coefficient reverses sign or stays negative in both halves.
#
# Inputs:    standardized firm-month panel
# Outputs:   output/stage3a/subsample_split.{json,md}
# Paper:     T8 tab:prepost_grad (pre/post-2015 subsample; IA 5-window grid)
# Run order: see code/00_master.py
# =====================================================================

"""Decisive subsample test, per output/stage4/self_attack_v2.md.

The self-attack v2 flagged: the seed reports FM-NW(12) subsample coefficients
(pre-2015 -0.0185 t=-1.81, post-2015 +0.0023 t=+0.28) — a sign reversal in the
cross-sectional estimator. But the HEADLINE estimator is the TWFE state+month
two-way state x month CGM clustered regression. The REVISE-pass legs (FF12,
K-K, Bayes) were all run on the POOLED sample. The decisive question:

    Does the TWFE-clustered headline coefficient REVERSE sign post-2015,
    or does it stay negative in both subsamples?

If the headline reverses, the paper is genuinely a pre-2015 paper and the title
needs "pre-2015". If the headline stays negative in both subsamples (even if FM
is insignificant post-2015), the full-sample claim holds under the headline
estimator and the FM divergence is a separate (already-disclosed) issue.

This script runs, split pre-2015 (2009-01..2014-12) vs post-2015
(2015-01..2023-12):
    (a) the headline TWFE state+month two-way state x month CGM
    (b) the FF12-absorbed spec
The K-K and Bayes specs are not re-split here — the headline + FF12 split is
the decisive test; if those reverse, the K-K/Bayes splits are moot, and if they
don't, the K-K/Bayes pooled results already stand as robustness.
"""

import os
import json
import numpy as np
import pandas as pd

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "subsample_split.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "subsample_split.md")

FF12_RANGES = [
    (1,  [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    (2,  [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
          (3630, 3659), (3700, 3711), (3714, 3714), (3716, 3716), (3718, 3718), (3720, 3729),
          (3732, 3739), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    (3,  [(2520, 2589), (2600, 2699), (2750, 2769), (2800, 2829), (2840, 2899), (3000, 3099),
          (3200, 3569), (3580, 3621), (3623, 3629), (3700, 3700), (3712, 3713), (3715, 3715),
          (3717, 3717), (3719, 3724), (3726, 3731), (3732, 3739), (3743, 3743), (3760, 3789),
          (3793, 3799), (3800, 3800), (3860, 3899)]),
    (4,  [(1200, 1399), (2900, 2999)]),
    (5,  [(2800, 2829), (2840, 2899)]),
    (6,  [(3570, 3579), (3622, 3622), (3660, 3692), (3694, 3699), (3810, 3839), (7370, 7372),
          (7373, 7373), (7374, 7374), (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378),
          (7379, 7379), (7391, 7391), (8730, 8734)]),
    (7,  [(4800, 4899)]),
    (8,  [(4900, 4949)]),
    (9,  [(5000, 5999), (7200, 7299), (7600, 7699)]),
    (10, [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    (11, [(6000, 6999)]),
]


def sic_to_ff12(sic):
    if pd.isna(sic):
        return 12
    sic = int(sic)
    for code, ranges in FF12_RANGES:
        for lo, hi in ranges:
            if lo <= sic <= hi:
                return code
    return 12


def two_way_clustered_se(y, X, cs, cm):
    n, k = X.shape
    XX_inv = np.linalg.pinv(X.T @ X)
    beta = XX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        m = np.zeros((k, k))
        for gid in np.unique(g):
            idx = np.where(g == gid)[0]
            Xge = X[idx].T @ e[idx]
            m += np.outer(Xge, Xge)
        return m

    M_s = meat(cs)
    M_m = meat(cm)
    n_months = int(cm.max()) + 1
    inter = cs.astype(np.int64) * n_months + cm.astype(np.int64)
    M_i = meat(inter)
    V = XX_inv @ (M_s + M_m - M_i) @ XX_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0))


def run_twfe(d, with_ff12):
    Xcols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
             'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']
    states = pd.get_dummies(d['hq_state'].astype(str), prefix='S', drop_first=True, dtype=float)
    months = pd.get_dummies(d['ym_str'], prefix='M', drop_first=True, dtype=float)
    blocks = [np.ones((len(d), 1)), d[Xcols].values.astype(float), states.values, months.values]
    if with_ff12:
        ff12 = pd.get_dummies(d['ff12'].astype(str), prefix='I', drop_first=True, dtype=float)
        blocks.append(ff12.values)
    X = np.hstack(blocks)
    y = d['ret'].values.astype(float)
    cs = pd.Categorical(d['hq_state']).codes
    cm = pd.Categorical(d['ym_str']).codes
    beta, se = two_way_clustered_se(y, X, cs, cm)
    idx = 1 + Xcols.index('mom_x_iv_x_literacy_corr')
    coef, se_ = float(beta[idx]), float(se[idx])
    return {"coef": coef, "se": se_, "t": coef / se_, "n_obs": int(len(d))}


def main():
    df = pd.read_parquet(PANEL)
    df['ym_str'] = df['date'].dt.to_period('M').astype(str)
    df['ff12'] = df['siccd'].apply(sic_to_ff12)
    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
              'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr',
              'hq_state', 'date']
    d = df.dropna(subset=needed).copy()

    pre = d[d['date'] < '2015-01-01'].copy()
    post = d[d['date'] >= '2015-01-01'].copy()
    print(f"pre-2015: {len(pre)} obs ({pre['date'].min()}..{pre['date'].max()})")
    print(f"post-2015: {len(post)} obs ({post['date'].min()}..{post['date'].max()})")

    results = {}
    for label, sub in [("full", d), ("pre_2015", pre), ("post_2015", post)]:
        twfe = run_twfe(sub, with_ff12=False)
        ff12 = run_twfe(sub, with_ff12=True)
        results[label] = {"twfe_headline": twfe, "twfe_ff12": ff12}
        print(f"\n[{label}] TWFE headline: coef={twfe['coef']:.6f}, se={twfe['se']:.6f}, "
              f"t={twfe['t']:.4f}, n={twfe['n_obs']}")
        print(f"[{label}] TWFE + FF12:   coef={ff12['coef']:.6f}, se={ff12['se']:.6f}, "
              f"t={ff12['t']:.4f}, n={ff12['n_obs']}")

    # decisive verdict
    pre_t = results['pre_2015']['twfe_headline']['t']
    post_t = results['post_2015']['twfe_headline']['t']
    pre_c = results['pre_2015']['twfe_headline']['coef']
    post_c = results['post_2015']['twfe_headline']['coef']
    sign_reverses = (np.sign(pre_c) != np.sign(post_c))
    post_still_neg_sig = (post_c < 0 and post_t < -1.96)

    results['verdict'] = {
        "headline_sign_reverses_post2015": bool(sign_reverses),
        "post2015_still_negative_and_significant": bool(post_still_neg_sig),
        "pre2015_twfe_t": pre_t,
        "post2015_twfe_t": post_t,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Decisive Subsample Test: TWFE-Clustered Headline, Pre/Post-2015\n\n")
        f.write("Per `output/stage4/self_attack_v2.md`'s single-highest-priority concern. "
                "The seed reports FM-NW(12) subsample coefficients showing a sign reversal "
                "(pre-2015 -0.0185 t=-1.81, post-2015 +0.0023 t=+0.28), but the HEADLINE "
                "estimator is the TWFE state+month two-way state x month CGM. This test runs "
                "the headline (and the FF12-absorbed spec) split pre-2015 vs post-2015.\n\n")
        f.write("## Results\n\n")
        f.write("| Sample | Spec | Coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label in ['full', 'pre_2015', 'post_2015']:
            t = results[label]['twfe_headline']
            ff = results[label]['twfe_ff12']
            f.write(f"| {label} | TWFE headline | {t['coef']:.6f} | {t['se']:.6f} | "
                    f"{t['t']:.4f} | {t['n_obs']:,} |\n")
            f.write(f"| {label} | TWFE + FF12 | {ff['coef']:.6f} | {ff['se']:.6f} | "
                    f"{ff['t']:.4f} | {ff['n_obs']:,} |\n")
        f.write("\n## Verdict\n\n")
        if sign_reverses:
            f.write(f"**THE HEADLINE SIGN REVERSES POST-2015.** Pre-2015 TWFE coef = "
                    f"{pre_c:.6f} (t={pre_t:.2f}); post-2015 TWFE coef = {post_c:.6f} "
                    f"(t={post_t:.2f}). The full-sample 'robustly-signed negative' claim is "
                    f"NOT legitimate as a 2009-2023 statement — the regularity is a pre-2015 "
                    f"phenomenon. **The paper must be reframed as a 2009-2014 (pre-2015) "
                    f"cross-sectional regularity**, with the post-2015 sign reversal as a "
                    f"documented feature (consistent with the post-2015 retail-composition "
                    f"shift the mechanism discusses). The title and abstract must say "
                    f"'pre-2015' or '2009-2014'.\n")
        elif post_still_neg_sig:
            f.write(f"**THE HEADLINE SIGN IS STABLE AND SIGNIFICANT IN BOTH SUBSAMPLES.** "
                    f"Pre-2015 TWFE coef = {pre_c:.6f} (t={pre_t:.2f}); post-2015 TWFE coef "
                    f"= {post_c:.6f} (t={post_t:.2f}). The full-sample 'robustly-signed "
                    f"negative' claim IS legitimate under the headline estimator. The FM "
                    f"subsample reversal is a separate (already-disclosed) cross-sectional-"
                    f"estimator issue, not a headline-estimator reversal. No reframe needed; "
                    f"the paper retains the 2009-2023 scope.\n")
        else:
            f.write(f"**THE HEADLINE SIGN IS STABLE (negative in both subsamples) BUT "
                    f"POST-2015 IS NOT SIGNIFICANT.** Pre-2015 TWFE coef = {pre_c:.6f} "
                    f"(t={pre_t:.2f}); post-2015 TWFE coef = {post_c:.6f} (t={post_t:.2f}). "
                    f"The sign does NOT reverse under the headline estimator — both "
                    f"subsamples are negative — but the post-2015 effect is statistically "
                    f"weaker. The honest framing: the negative sign is present throughout "
                    f"the sample but the effect is concentrated (in significance) pre-2015. "
                    f"The paper retains the 2009-2023 scope but must disclose the post-2015 "
                    f"attenuation honestly — it does not need a 'pre-2015' title, but the "
                    f"abstract must note the effect weakens post-2015.\n")
    print(f"\nVERDICT: sign_reverses={sign_reverses}, post_still_neg_sig={post_still_neg_sig}")
    return results


if __name__ == '__main__':
    main()
