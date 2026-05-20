# =====================================================================
# decile_trend_and_kk_multiproxy.py
# Continuous IV-decile-rank trend test plus a two-proxy Korniotis-Kumar business-cycle block (unemployment + Phila-Fed coincident-index growth) on the literacy three-way.
#
# Inputs:    standardized firm-month panel, _fred_state_cache.parquet
# Outputs:   output/stage3a/decile_trend_kk_multiproxy.{json,md}
# Paper:     IA Korniotis-Kumar section (business-cycle block) + within-retail IV-decile detail
# Run order: see code/00_master.py
# =====================================================================

"""Two pre-Stage-5 tests, per output/stage4/triage_v2.md.

(S79-E) Continuous decile-rank trend test.
    The self-attack flags that decile-9 significance is one of ten decile
    sub-panels and may be a multiple-testing survivor / power artifact rather
    than a real "IV-concentration" pattern. The decisive test: load the
    within-month NYSE-IV decile RANK as a continuous variable and interact it
    with mom*iv*literacy_z. If the four-way `mom*iv*literacy_z*iv_decile_rank`
    coefficient is significantly negative, the negative loading genuinely
    deepens with IV (a real gradient); if it is not, "IV-concentration"
    language must be dropped from Posit 2 and the result framed as a single-
    decile spike only.

(S79-D) K-K multi-proxy business-cycle block.
    The single-proxy K-K test (unemployment only) showed ~23% attenuation.
    The self-attack notes 23% is a lower bound. This adds a SECOND business-
    cycle proxy — the Philadelphia Fed state coincident-index 12-month growth
    (already fetched in _fred_state_cache.parquet) — to the K-K block, so the
    block is {unemp_z, coincident_g12_z} x {1, mom, iv, mom*iv}. Reports how
    much further the literacy three-way attenuates with the richer block.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical/code")

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
CACHE = os.path.join(ROOT, "code", "empirical", "_fred_state_cache.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "decile_trend_kk_multiproxy.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "decile_trend_kk_multiproxy.md")


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
    print(f"  [{label}] {target}: coef={coef:.6f}, se={se_:.6f}, t={coef/se_:.4f}, n={X.shape[0]}")
    return {"coef": coef, "se": se_, "t": coef / se_, "n_obs": int(X.shape[0]), "target": target}


def main():
    df = pd.read_parquet(PANEL)
    df['ym_str'] = df['date'].dt.to_period('M').astype(str)
    df['ym'] = df['date'].dt.to_period('M')
    df['st'] = df['hq_state'].astype(str).str.replace('US-', '', regex=False)

    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
              'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr',
              'hq_state', 'iv_rank']
    d = df.dropna(subset=needed).copy()
    print(f"base sample: {d.shape}")
    print(f"iv_rank range: {d['iv_rank'].min()} .. {d['iv_rank'].max()}, "
          f"unique: {sorted(d['iv_rank'].unique())[:12]}")

    # ============ (S79-E) Continuous decile-rank trend test ============
    print("\n=== (S79-E) continuous decile-rank trend test ===")
    # iv_rank is the within-month NYSE-IV decile (0..9). Standardize it within
    # month and build the four-way interaction.
    d['iv_decile_z'] = d.groupby('ym_str')['iv_rank'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    d['mom_iv_lit_x_decile'] = d['mom_x_iv_x_literacy_corr'] * d['iv_decile_z']
    # z-score the new four-way within month (match seed convention)
    for c in ['mom_iv_lit_x_decile', 'iv_decile_z']:
        d[c] = d.groupby('ym_str')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    # also need the lower-order three-way-with-decile partials for a clean spec:
    # include iv_decile_z, mom*iv*lit (already have), and the four-way.
    trend_cols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
                  'mom_x_literacy_corr', 'iv_x_literacy_corr',
                  'mom_x_iv_x_literacy_corr', 'iv_decile_z', 'mom_iv_lit_x_decile']
    spec_trend = run(d, trend_cols, 'mom_iv_lit_x_decile', "decile-rank trend (four-way)")
    # also report the three-way in the same spec for context
    spec_trend_threeway = run(d, trend_cols, 'mom_x_iv_x_literacy_corr',
                              "three-way in the trend spec")

    trend_neg_sig = (spec_trend['coef'] < 0 and spec_trend['t'] < -1.96)

    # ============ (S79-D) K-K multi-proxy business-cycle block ============
    print("\n=== (S79-D) K-K multi-proxy business-cycle block ===")
    macro = pd.read_parquet(CACHE)
    macro['ym'] = macro['date'].dt.to_period('M')
    unemp = macro[macro['var'] == 'unemp'].pivot_table(index='ym', columns='st', values='value')
    coin = macro[macro['var'] == 'coincident'].pivot_table(index='ym', columns='st', values='value')
    coin_g12 = np.log(coin) - np.log(coin.shift(12))

    def zlong(wide, name):
        lng = wide.reset_index().melt(id_vars='ym', var_name='st', value_name=name)
        lng['_z'] = lng.groupby('ym')[name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
        return lng[['ym', 'st', '_z']].rename(columns={'_z': name + '_z'})

    bc = zlong(unemp, 'unemp').merge(zlong(coin_g12, 'coincident_g12'), on=['ym', 'st'], how='outer')
    d2 = d.merge(bc, on=['ym', 'st'], how='left').dropna(subset=['unemp_z', 'coincident_g12_z']).copy()
    print(f"K-K multi-proxy sample (needs both bc proxies): {d2.shape}")

    for bcvar in ['unemp_z', 'coincident_g12_z']:
        d2[f'mom_x_{bcvar}'] = d2['mom_12_2'] * d2[bcvar]
        d2[f'iv_x_{bcvar}'] = d2['iv'] * d2[bcvar]
        d2[f'mom_iv_x_{bcvar}'] = d2['mom_x_iv'] * d2[bcvar]
    bc_terms = []
    for bcvar in ['unemp_z', 'coincident_g12_z']:
        bc_terms += [bcvar, f'mom_x_{bcvar}', f'iv_x_{bcvar}', f'mom_iv_x_{bcvar}']
    for c in bc_terms:
        d2[c] = d2.groupby('ym_str')[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    base_cols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
                 'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']
    spec_base2 = run(d2, base_cols, 'mom_x_iv_x_literacy_corr',
                     "baseline (multi-proxy sample, no K-K block)")
    spec_multi = run(d2, base_cols + bc_terms, 'mom_x_iv_x_literacy_corr',
                     "with 2-proxy K-K block (unemp + coincident growth)")
    delta_multi = spec_multi['coef'] - spec_base2['coef']
    retained_multi = spec_multi['coef'] / spec_base2['coef']

    res = {
        "decile_trend": {
            "four_way_coef": spec_trend,
            "three_way_in_trend_spec": spec_trend_threeway,
            "trend_negative_and_significant": bool(trend_neg_sig),
        },
        "kk_multiproxy": {
            "spec_baseline": spec_base2,
            "spec_2proxy_kk_block": spec_multi,
            "delta_coef": delta_multi,
            "delta_in_baseline_se": delta_multi / spec_base2['se'],
            "fraction_retained": retained_multi,
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Pre-Stage-5 Tests: Decile-Rank Trend + K-K Multi-Proxy\n\n")
        f.write("Per `output/stage4/triage_v2.md` items S79-E and S79-D.\n\n")
        f.write("## (S79-E) Continuous decile-rank trend test\n\n")
        f.write("Tests whether the negative three-way loading genuinely *deepens* "
                "with IV (a real gradient) or is a single-decile spike. The "
                "within-month NYSE-IV decile rank is standardized and interacted "
                "with `mom x iv x literacy_z` to form a four-way term; a "
                "significantly negative four-way coefficient means the negative "
                "loading deepens monotonically with the IV decile.\n\n")
        f.write("| Term | Coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| `mom x iv x literacy_z x iv_decile_z` (four-way trend) | "
                f"{spec_trend['coef']:.6f} | {spec_trend['se']:.6f} | "
                f"{spec_trend['t']:.4f} | {spec_trend['n_obs']:,} |\n")
        f.write(f"| `mom x iv x literacy_z` (three-way, same spec) | "
                f"{spec_trend_threeway['coef']:.6f} | {spec_trend_threeway['se']:.6f} | "
                f"{spec_trend_threeway['t']:.4f} | {spec_trend_threeway['n_obs']:,} |\n\n")
        if trend_neg_sig:
            f.write(f"**Verdict: REAL IV GRADIENT.** The four-way decile-rank trend "
                    f"coefficient is significantly negative (t = {spec_trend['t']:.2f}). "
                    f"The negative three-way loading genuinely deepens with the IV "
                    f"decile — 'concentrated in high-IV firms' is a real monotone "
                    f"gradient, not a single-decile spike. Posit 2's IV-concentration "
                    f"language is supported by the continuous trend test.\n\n")
        else:
            f.write(f"**Verdict: NOT A MONOTONE GRADIENT.** The four-way decile-rank "
                    f"trend coefficient is {spec_trend['coef']:.6f} "
                    f"(t = {spec_trend['t']:.2f}) — not significantly negative. The "
                    f"negative three-way loading does NOT deepen monotonically with "
                    f"the IV decile. **Posit 2 must drop 'concentrated in high-IV "
                    f"firms' monotone-gradient language** and be reframed as: the "
                    f"family-wise-significant decile-9 result is a single-decile "
                    f"feature under a non-monotone reading, not an IV gradient. This "
                    f"is consistent with the IV-tercile split (tercile 3 positive) "
                    f"already reported.\n\n")
        f.write("## (S79-D) K-K multi-proxy business-cycle block\n\n")
        f.write("The single-proxy K-K test (unemployment only) showed ~23% "
                "attenuation of the literacy three-way. The self-attack notes 23% "
                "is a lower bound. This adds a second business-cycle proxy — the "
                "Philadelphia Fed state coincident-index 12-month growth — so the "
                "K-K block is {unemp_z, coincident_g12_z} x {1, mom, iv, mom x iv}.\n\n")
        f.write("| Specification | lit three-way coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Baseline (multi-proxy sample, no K-K block) | "
                f"{spec_base2['coef']:.6f} | {spec_base2['se']:.6f} | "
                f"{spec_base2['t']:.4f} | {spec_base2['n_obs']:,} |\n")
        f.write(f"| **+ 2-proxy K-K block** | **{spec_multi['coef']:.6f}** | "
                f"**{spec_multi['se']:.6f}** | **{spec_multi['t']:.4f}** | "
                f"**{spec_multi['n_obs']:,}** |\n\n")
        f.write(f"**Change:** {delta_multi:+.6f} ({delta_multi/spec_base2['se']:+.3f} "
                f"baseline SEs); the literacy three-way retains "
                f"{retained_multi*100:.1f}% of its baseline magnitude with the "
                f"2-proxy K-K block (vs. ~77% with the 1-proxy block). ")
        if spec_multi['coef'] < 0 and spec_multi['t'] < -1.96:
            f.write("The coefficient remains negative and significant — the "
                    "literacy channel is distinct from the business cycle even "
                    "under a richer 2-proxy K-K control block, though the "
                    "lower-bound nature of the overlap estimate is confirmed: "
                    "adding the second proxy attenuates further.\n")
        else:
            f.write("The coefficient no longer survives the richer K-K block — "
                    "the business-cycle overlap is substantial and the paper must "
                    "disclose that the literacy channel is not cleanly separable "
                    "from the state business cycle.\n")
    print(f"\ndecile trend: neg+sig = {trend_neg_sig}")
    print(f"K-K multi-proxy: coef {spec_base2['coef']:.6f} -> {spec_multi['coef']:.6f} "
          f"(retains {retained_multi*100:.1f}%)")
    return res


if __name__ == '__main__':
    main()
