# =====================================================================
# kk_orthogonality.py
# Korniotis-Kumar business-cycle orthogonality test: adds a state business-cycle interaction block (unemployment / Phila-Fed coincident index) and checks the literacy three-way survives.
#
# Inputs:    standardized firm-month panel, FRED state series (unemployment, coincident index)
# Outputs:   output/stage3a/kk_orthogonality.{json,md}
# Paper:     IA Korniotis-Kumar section
# Run order: see code/00_master.py
# =====================================================================

"""REVISE pass — Korniotis-Kumar business-cycle orthogonality test.

Per output/stage4/triage_v1.md item (1) and output/stage4/scorer_decision_v1.md
substantive feedback item (1): the mechanism document claims the literacy
three-way coefficient is orthogonal to the Korniotis-Kumar (2013) state-level
business-cycle channel. State + month FE absorb time-invariant state
characteristics and aggregate time variation, but they do NOT absorb the
state-level *business cycle* interacted with the firm-level momentum-IV cross
section. This test adds that block and checks whether the literacy three-way
coefficient survives.

Design:
    ret = a + [7 original focal terms: mom, iv, lit_z, mom*iv, mom*lit, iv*lit,
               mom*iv*lit]
            + [K-K business-cycle block: bc_z, mom*bc_z, iv*bc_z, mom*iv*bc_z]
            + state_FE + month_FE + e
    two-way state x month CGM clustering.

The K-K business-cycle variable `bc_z` is the state-level business-cycle
indicator, z-scored within month. Two operationalizations:
    (a) state unemployment rate (FRED {ST}UR, monthly, seasonally adjusted)
    (b) Philadelphia Fed state coincident index 12-month growth (FRED {ST}PHCI)
Both are forward-filled / aligned to the firm-month panel by HQ state and month.

The orthogonality claim is supported if the literacy three-way coefficient
`mom*iv*lit_z` changes by less than ~1 SE when the K-K block is added.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical/code")
os.environ.setdefault("FRED_API_KEY", "6e9900889d06f337967b57c8a152c123")
from utils.fred_utils import get_series

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "kk_orthogonality.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "kk_orthogonality.md")
CACHE = os.path.join(ROOT, "code", "empirical", "_fred_state_cache.parquet")

# US-XX -> 2-letter code. The panel uses "US-CA" style hq_state.
STATE_CODES = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
               'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
               'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
               'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']


def two_way_clustered_se(y, X, cluster_state, cluster_month):
    n, k = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.pinv(XX)
    beta = XX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        m = np.zeros((k, k))
        for gid in np.unique(g):
            idx = np.where(g == gid)[0]
            Xge = X[idx].T @ e[idx]
            m += np.outer(Xge, Xge)
        return m

    M_state = meat(cluster_state)
    M_month = meat(cluster_month)
    n_months = int(cluster_month.max()) + 1
    inter = cluster_state.astype(np.int64) * n_months + cluster_month.astype(np.int64)
    M_inter = meat(inter)
    V = XX_inv @ (M_state + M_month - M_inter) @ XX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta, se


def fetch_state_macro():
    """Fetch state unemployment rate (monthly) and Phila Fed coincident index
    (monthly) for all 51 states+DC, 2008-2024. Cached to parquet."""
    if os.path.exists(CACHE):
        print(f"loading cached state macro from {CACHE}")
        return pd.read_parquet(CACHE)

    rows = []
    for st in STATE_CODES:
        for suffix, varname in [("UR", "unemp"), ("PHCI", "coincident")]:
            series_id = f"{st}{suffix}"
            for attempt in range(3):
                try:
                    s = get_series(series_id, start="2007-06-01", end="2024-03-01")
                    for dt, val in s.items():
                        rows.append({"st": st, "date": pd.Timestamp(dt),
                                     "var": varname, "value": float(val)})
                    print(f"  {series_id}: {len(s)} obs")
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  {series_id}: FAILED after 3 tries: {str(e)[:80]}")
                    else:
                        time.sleep(1.0)
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"cached {len(df)} state-macro rows to {CACHE}")
    return df


def build_bc_panel(macro):
    """Build a state-month business-cycle panel from the raw FRED series.
    - unemp: level, z-scored within month across states.
    - coincident_g12: 12-month log growth of the coincident index, z-scored
      within month across states.
    """
    macro = macro.copy()
    macro['ym'] = macro['date'].dt.to_period('M')

    # pivot
    unemp = macro[macro['var'] == 'unemp'].pivot_table(
        index='ym', columns='st', values='value')
    coin = macro[macro['var'] == 'coincident'].pivot_table(
        index='ym', columns='st', values='value')

    # 12-month log growth of coincident index
    coin_g12 = np.log(coin) - np.log(coin.shift(12))

    # melt back to long, z-score within month
    def zlong(wide, name):
        long = wide.reset_index().melt(id_vars='ym', var_name='st', value_name=name)
        long['_z'] = long.groupby('ym')[name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
        return long[['ym', 'st', '_z']].rename(columns={'_z': name + '_z'})

    u = zlong(unemp, 'unemp')
    c = zlong(coin_g12, 'coincident_g12')
    bc = u.merge(c, on=['ym', 'st'], how='outer')
    return bc


def run_spec(d, focal_cols, label):
    """Run TWFE state+month with two-way state×month CGM clustering on the
    given focal columns; return the coefficient on mom_x_iv_x_literacy_corr."""
    states = pd.get_dummies(d['hq_state'].astype(str), prefix='S', drop_first=True, dtype=float)
    months = pd.get_dummies(d['ym_str'], prefix='M', drop_first=True, dtype=float)
    X = np.hstack([np.ones((len(d), 1)), d[focal_cols].values.astype(float),
                   states.values, months.values])
    y = d['ret'].values.astype(float)
    state_codes = pd.Categorical(d['hq_state']).codes
    month_codes = pd.Categorical(d['ym_str']).codes
    beta, se = two_way_clustered_se(y, X, state_codes, month_codes)
    idx = 1 + focal_cols.index('mom_x_iv_x_literacy_corr')
    coef, se_ = float(beta[idx]), float(se[idx])
    print(f"  [{label}] n={X.shape[0]}, k={X.shape[1]}: "
          f"lit-three-way coef={coef:.6f}, se={se_:.6f}, t={coef/se_:.4f}")
    return {"coef": coef, "se": se_, "t": coef / se_, "n_obs": int(X.shape[0]),
            "n_params": int(X.shape[1]), "label": label}


def main():
    print("=== fetching state macro from FRED ===")
    macro = fetch_state_macro()
    bc = build_bc_panel(macro)
    print(f"bc panel: {bc.shape}, months {bc['ym'].min()}..{bc['ym'].max()}")

    print("\n=== loading firm-month panel ===")
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M')
    df['ym_str'] = df['ym'].astype(str)
    df['st'] = df['hq_state'].astype(str).str.replace('US-', '', regex=False)

    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected',
              'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
              'mom_x_iv_x_literacy_corr', 'hq_state', 'date']
    d = df.dropna(subset=needed).copy()

    # merge business-cycle vars
    d = d.merge(bc, on=['ym', 'st'], how='left')
    print(f"after bc merge: {d.shape}; "
          f"unemp_z missing: {d['unemp_z'].isna().sum()}, "
          f"coincident_g12_z missing: {d['coincident_g12_z'].isna().sum()}")

    # Build K-K business-cycle interaction terms. Use unemp_z as the primary
    # K-K business-cycle indicator (monthly, complete coverage). Z-score the
    # interaction products WITHIN MONTH to match the seed's standardization.
    d = d.dropna(subset=['unemp_z']).copy()
    d['mom_x_bc'] = d['mom_12_2'] * d['unemp_z']
    d['iv_x_bc'] = d['iv'] * d['unemp_z']
    d['mom_x_iv_x_bc'] = d['mom_x_iv'] * d['unemp_z']

    # z-score the new interaction products within month (match seed convention)
    for col in ['mom_x_bc', 'iv_x_bc', 'mom_x_iv_x_bc', 'unemp_z']:
        d[col] = d.groupby('ym_str')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    print(f"final estimation sample: {d.shape}")

    base_cols = ['mom_12_2', 'iv', 'literacy_score_corrected',
                 'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
                 'mom_x_iv_x_literacy_corr']
    kk_cols = base_cols + ['unemp_z', 'mom_x_bc', 'iv_x_bc', 'mom_x_iv_x_bc']

    print("\n=== running specifications ===")
    # Spec 0: baseline on the bc-merged sample (sample may differ slightly from
    # full 623,896 due to bc coverage — report this sample's baseline for a
    # clean before/after comparison)
    spec_base = run_spec(d, base_cols, "baseline (bc-merged sample, no K-K block)")
    spec_kk = run_spec(d, kk_cols, "with K-K business-cycle block")

    delta = spec_kk['coef'] - spec_base['coef']
    delta_in_se = delta / spec_base['se']
    survives = abs(delta_in_se) < 1.0

    res = {
        "spec_baseline_bc_sample": spec_base,
        "spec_with_kk_block": spec_kk,
        "delta_coef": delta,
        "delta_in_baseline_se": delta_in_se,
        "orthogonality_supported": bool(survives),
        "kk_business_cycle_var": "state unemployment rate (FRED {ST}UR), z-scored within month",
        "note": "Phila Fed coincident-index growth fetched as a secondary bc proxy; "
                "primary test uses state unemployment rate for complete monthly coverage.",
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Korniotis-Kumar Business-Cycle Orthogonality Test\n\n")
        f.write("Per `output/stage4/triage_v1.md` item (1) and the structured "
                "scorer's substantive feedback. Tests whether the literacy "
                "three-way coefficient `mom x IV x literacy_z` survives the "
                "addition of the Korniotis-Kumar (2013) state-level "
                "business-cycle channel interacted with the firm-level "
                "momentum-IV cross section.\n\n")
        f.write("**K-K business-cycle variable:** state unemployment rate "
                "(FRED `{ST}UR`, monthly seasonally-adjusted, 51 states+DC), "
                "z-scored within month across states. The block added is "
                "`unemp_z`, `mom x unemp_z`, `iv x unemp_z`, "
                "`mom x iv x unemp_z` (the business-cycle analogues of the "
                "literacy interaction terms).\n\n")
        f.write("**Specification:** TWFE state + month FE, two-way state x month "
                "CGM clustering, on the bc-merged sample.\n\n")
        f.write("| Specification | lit three-way coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Baseline (bc-merged sample, no K-K block) | "
                f"{spec_base['coef']:.6f} | {spec_base['se']:.6f} | "
                f"{spec_base['t']:.4f} | {spec_base['n_obs']:,} |\n")
        f.write(f"| **+ K-K business-cycle block** | "
                f"**{spec_kk['coef']:.6f}** | **{spec_kk['se']:.6f}** | "
                f"**{spec_kk['t']:.4f}** | **{spec_kk['n_obs']:,}** |\n\n")
        f.write(f"**Change in literacy three-way coefficient:** {delta:+.6f} "
                f"({delta_in_se:+.3f} baseline SEs).\n\n")
        if survives:
            f.write(f"**Verdict: ORTHOGONALITY SUPPORTED.** The literacy "
                    f"three-way coefficient changes by {abs(delta_in_se):.2f} "
                    f"baseline SEs when the Korniotis-Kumar business-cycle "
                    f"block is added — less than the ~1-SE threshold the "
                    f"mechanism document's Implication 5 pre-registered. The "
                    f"literacy channel is empirically distinct from the "
                    f"state-level business-cycle channel; the mechanism's "
                    f"orthogonality claim is now demonstrated, not asserted.\n")
        else:
            f.write(f"**Verdict: ORTHOGONALITY NOT SUPPORTED.** The literacy "
                    f"three-way coefficient changes by {abs(delta_in_se):.2f} "
                    f"baseline SEs when the K-K business-cycle block is added "
                    f"— more than the ~1-SE threshold. The literacy channel is "
                    f"partly confounded with the state business cycle; the "
                    f"mechanism's orthogonality claim must be downgraded and "
                    f"the paper must disclose this honestly.\n")
    print(f"\ndelta = {delta:+.6f} ({delta_in_se:+.3f} baseline SEs), "
          f"orthogonality {'SUPPORTED' if survives else 'NOT SUPPORTED'}")
    return res


if __name__ == '__main__':
    main()
