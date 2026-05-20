# =====================================================================
# kk_bayes_subsample.py
# Splits the remaining two robustness specs (Korniotis-Kumar business-cycle-controlled and Bayes-shrunk-literacy) pre/post-2015 to back the full-sample scope claim.
#
# Inputs:    standardized firm-month panel, _fred_state_cache.parquet, 5 NFCS waves (raw CSVs)
# Outputs:   output/stage3a/kk_bayes_subsample.{json,md}
# Paper:     IA Korniotis-Kumar section + 5-window subsample grid
# Run order: see code/00_master.py
# =====================================================================

"""Final pre-Stage-5 deepening leg, per output/stage4/scorer_freeform_v2.md
pivotal recommendation and output/stage4/triage_v2.md S79-D.

The decisive pre/post-2015 subsample test (subsample_split.py) split the
HEADLINE and FF12-absorbed specs and both survived (negative in both
subsamples). But the K-K-business-cycle-controlled spec and the
Bayes-shrunk-literacy spec were NOT split — the freeform scorer flags that
the "2009-2023 scope" claim currently rests on 2 of 4 specs being split-tested.

This script splits the remaining two:
    (a) K-K-controlled spec (headline + unemployment business-cycle block),
        pre-2015 vs post-2015.
    (b) Bayes-shrunk-literacy spec, pre-2015 vs post-2015.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical/code")

ROOT = Path(r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical")
PANEL = ROOT / "output" / "seed" / "data" / "processed" / "panel_corrected_standardized.parquet"
CACHE = ROOT / "code" / "empirical" / "_fred_state_cache.parquet"
NFCS_DIR = ROOT / "output" / "seed" / "data" / "raw" / "nfcs"
OUT_JSON = ROOT / "output" / "stage3a" / "kk_bayes_subsample.json"
OUT_MD = ROOT / "output" / "stage3a" / "kk_bayes_subsample.md"

CORRECT_CODES = {"m6": 1, "m7": 3, "m8": 2, "m9": 2, "m10": 1}
STATEQ_MAP = {
    1: "AL", 2: "AK", 3: "AZ", 4: "AR", 5: "CA", 6: "CO", 7: "CT", 8: "DE",
    9: "DC", 10: "FL", 11: "GA", 12: "HI", 13: "ID", 14: "IL", 15: "IN",
    16: "IA", 17: "KS", 18: "KY", 19: "LA", 20: "ME", 21: "MD", 22: "MA",
    23: "MI", 24: "MN", 25: "MS", 26: "MO", 27: "MT", 28: "NE", 29: "NV",
    30: "NH", 31: "NJ", 32: "NM", 33: "NY", 34: "NC", 35: "ND", 36: "OH",
    37: "OK", 38: "OR", 39: "PA", 40: "RI", 41: "SC", 42: "SD", 43: "TN",
    44: "TX", 45: "UT", 46: "VT", 47: "VA", 48: "WA", 49: "WV", 50: "WI",
    51: "WY",
}
WAVES = {2009: "nfcs_2009.csv", 2012: "nfcs_2012.csv", 2015: "nfcs_2015.csv",
         2018: "nfcs_2018.csv", 2021: "nfcs_2021.csv"}


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


def run(d, focal_cols, target, label):
    states = pd.get_dummies(d['hq_state'].astype(str), prefix='S', drop_first=True, dtype=float)
    months = pd.get_dummies(d['ym_str'], prefix='M', drop_first=True, dtype=float)
    X = np.hstack([np.ones((len(d), 1)), d[focal_cols].values.astype(float),
                   states.values, months.values])
    y = d['ret'].values.astype(float)
    cs = pd.Categorical(d['hq_state']).codes
    cm = pd.Categorical(d['ym_str']).codes
    beta, se = two_way_clustered_se(y, X, cs, cm)
    idx = 1 + focal_cols.index(target)
    coef, se_ = float(beta[idx]), float(se[idx])
    print(f"  [{label}] coef={coef:.6f}, se={se_:.6f}, t={coef/se_:.4f}, n={X.shape[0]}")
    return {"coef": coef, "se": se_, "t": coef / se_, "n_obs": int(X.shape[0])}


def build_shrunk_literacy():
    """Empirical-Bayes shrunk state-wave literacy (same as bayes_shrinkage_literacy.py)."""
    parts = []
    for yr, fname in WAVES.items():
        df = pd.read_csv(NFCS_DIR / fname, low_memory=False)
        df.columns = df.columns.str.lower()
        for q, code in CORRECT_CODES.items():
            df[f"{q}_c"] = (pd.to_numeric(df[q], errors="coerce") == code).astype(float)
        df["n_correct"] = df[[f"{q}_c" for q in CORRECT_CODES]].sum(axis=1)
        df["w"] = pd.to_numeric(df.get("wgt_n2", 1.0), errors="coerce").fillna(0) if "wgt_n2" in df.columns else 1.0
        rows = []
        for stq, grp in df.groupby("stateq"):
            try:
                ab = STATEQ_MAP.get(int(stq))
            except (TypeError, ValueError):
                continue
            if ab is None or grp["w"].sum() == 0:
                continue
            w = grp["w"].values
            correct = (grp["n_correct"] >= 3).astype(float).values
            w_sum = w.sum()
            p_hat = (w * correct).sum() / w_sum
            n_eff = (w_sum ** 2) / (w ** 2).sum()
            v_s = p_hat * (1 - p_hat) / n_eff
            rows.append({"state": f"US-{ab}", "year": yr, "p_hat": p_hat, "v_s": v_s})
        wave = pd.DataFrame(rows)
        p = wave["p_hat"].values
        v_s = wave["v_s"].values
        grand = p.mean()
        v_true = max(p.var(ddof=1) - v_s.mean(), 1e-8)
        B = v_s / (v_s + v_true)
        wave["p_shrunk"] = B * grand + (1 - B) * p
        parts.append(wave)
    return pd.concat(parts, ignore_index=True)


def wave_for_year(y):
    if y <= 2011:
        return 2009
    if y <= 2014:
        return 2012
    if y <= 2017:
        return 2015
    if y <= 2020:
        return 2018
    return 2021


def main():
    df = pd.read_parquet(PANEL)
    df['ym_str'] = df['date'].dt.to_period('M').astype(str)
    df['ym'] = df['date'].dt.to_period('M')
    df['st'] = df['hq_state'].astype(str).str.replace('US-', '', regex=False)
    df['wave_year'] = df['year'].apply(wave_for_year)

    # ---- K-K business-cycle merge ----
    macro = pd.read_parquet(CACHE)
    macro['ym'] = macro['date'].dt.to_period('M')
    unemp = macro[macro['var'] == 'unemp'].pivot_table(index='ym', columns='st', values='value')
    unemp_long = unemp.reset_index().melt(id_vars='ym', var_name='st', value_name='unemp')
    unemp_long['unemp_z'] = unemp_long.groupby('ym')['unemp'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    df = df.merge(unemp_long[['ym', 'st', 'unemp_z']], on=['ym', 'st'], how='left')

    # ---- Bayes-shrunk literacy merge ----
    lit = build_shrunk_literacy()
    df = df.merge(lit.set_index(['state', 'year'])[['p_shrunk']],
                  left_on=['hq_state', 'wave_year'], right_index=True, how='left')

    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
              'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr',
              'hq_state', 'unemp_z', 'p_shrunk', 'date']
    d = df.dropna(subset=needed).copy()
    print(f"base sample: {d.shape}")

    results = {}
    for label, sub in [("full", d),
                       ("pre_2015", d[d['date'] < '2015-01-01'].copy()),
                       ("post_2015", d[d['date'] >= '2015-01-01'].copy())]:
        s = sub.copy()
        # K-K spec: build unemployment interactions, z-score within month
        s['mom_x_bc'] = s['mom_12_2'] * s['unemp_z']
        s['iv_x_bc'] = s['iv'] * s['unemp_z']
        s['mom_x_iv_x_bc'] = s['mom_x_iv'] * s['unemp_z']
        for c in ['mom_x_bc', 'iv_x_bc', 'mom_x_iv_x_bc', 'unemp_z']:
            s[c] = s.groupby('ym_str')[c].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
        kk_cols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
                   'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr',
                   'unemp_z', 'mom_x_bc', 'iv_x_bc', 'mom_x_iv_x_bc']
        kk = run(s, kk_cols, 'mom_x_iv_x_literacy_corr', f"{label}: K-K-controlled")

        # Bayes-shrunk spec: z-score shrunk literacy within month, rebuild three-way
        s['lit_shrunk_z'] = s.groupby('ym_str')['p_shrunk'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
        s['mom_x_lsh'] = s['mom_12_2'] * s['lit_shrunk_z']
        s['iv_x_lsh'] = s['iv'] * s['lit_shrunk_z']
        s['mom_x_iv_x_lsh'] = s['mom_x_iv'] * s['lit_shrunk_z']
        for c in ['mom_x_lsh', 'iv_x_lsh', 'mom_x_iv_x_lsh', 'lit_shrunk_z']:
            s[c] = s.groupby('ym_str')[c].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
        bayes_cols = ['mom_12_2', 'iv', 'lit_shrunk_z', 'mom_x_iv',
                      'mom_x_lsh', 'iv_x_lsh', 'mom_x_iv_x_lsh']
        bayes = run(s, bayes_cols, 'mom_x_iv_x_lsh', f"{label}: Bayes-shrunk")

        results[label] = {"kk_controlled": kk, "bayes_shrunk": bayes}

    # verdict
    kk_pre = results['pre_2015']['kk_controlled']
    kk_post = results['post_2015']['kk_controlled']
    by_pre = results['pre_2015']['bayes_shrunk']
    by_post = results['post_2015']['bayes_shrunk']
    kk_sign_stable = (kk_pre['coef'] < 0 and kk_post['coef'] < 0)
    by_sign_stable = (by_pre['coef'] < 0 and by_post['coef'] < 0)
    results['verdict'] = {
        "kk_sign_stable_both_subsamples": bool(kk_sign_stable),
        "bayes_sign_stable_both_subsamples": bool(by_sign_stable),
        "all_four_specs_now_split_tested": True,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# K-K-Controlled and Bayes-Shrunk Specs: Pre/Post-2015 Split\n\n")
        f.write("Per `output/stage4/scorer_freeform_v2.md`'s pivotal recommendation and "
                "`output/stage4/triage_v2.md` S79-D. The decisive subsample test "
                "(`subsample_split.md`) split the HEADLINE and FF12-absorbed specs "
                "(both survived). This completes the picture by splitting the remaining "
                "two REVISE-pass specs — K-K-business-cycle-controlled and "
                "Bayes-shrunk-literacy — pre-2015 vs post-2015.\n\n")
        f.write("## Results\n\n")
        f.write("| Sample | Spec | Coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label in ['full', 'pre_2015', 'post_2015']:
            kk = results[label]['kk_controlled']
            by = results[label]['bayes_shrunk']
            f.write(f"| {label} | K-K-controlled | {kk['coef']:.6f} | {kk['se']:.6f} | "
                    f"{kk['t']:.4f} | {kk['n_obs']:,} |\n")
            f.write(f"| {label} | Bayes-shrunk | {by['coef']:.6f} | {by['se']:.6f} | "
                    f"{by['t']:.4f} | {by['n_obs']:,} |\n")
        f.write("\n## Verdict\n\n")
        f.write(f"- **K-K-controlled spec:** sign {'STABLE (negative in both subsamples)' if kk_sign_stable else 'NOT stable across subsamples'} "
                f"— pre-2015 coef = {kk_pre['coef']:.6f} (t={kk_pre['t']:.2f}), "
                f"post-2015 coef = {kk_post['coef']:.6f} (t={kk_post['t']:.2f}).\n")
        f.write(f"- **Bayes-shrunk spec:** sign {'STABLE (negative in both subsamples)' if by_sign_stable else 'NOT stable across subsamples'} "
                f"— pre-2015 coef = {by_pre['coef']:.6f} (t={by_pre['t']:.2f}), "
                f"post-2015 coef = {by_post['coef']:.6f} (t={by_post['t']:.2f}).\n\n")
        if kk_sign_stable and by_sign_stable:
            f.write("**All four REVISE-pass specifications are now split-tested and all "
                    "four show a negative sign in BOTH the pre-2015 and post-2015 "
                    "subsamples.** Combined with the headline and FF12-absorbed splits "
                    "(`subsample_split.md`), the negative-sign claim is robust across "
                    "the full sample, the pre-2015 window, and the post-2015 window, in "
                    "all four specifications. The freeform scorer's concern that the "
                    "'2009-2023 scope' rested on only 2 of 4 split-tested specs is "
                    "resolved: all 4 are now split-tested and all 4 hold.\n")
        else:
            f.write("**At least one REVISE-pass specification does NOT hold its sign "
                    "across both subsamples.** The paper must disclose which "
                    "specification is subsample-sensitive and frame the scope claim "
                    "accordingly.\n")
    print(f"\nK-K sign stable: {kk_sign_stable}; Bayes sign stable: {by_sign_stable}")
    return results


if __name__ == '__main__':
    main()
