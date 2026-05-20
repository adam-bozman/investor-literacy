# =====================================================================
# reproduction_check.py
# Stage 3a re-validation: reproduces the headline TWFE three-way coefficient and its two-way state x month CGM SE from the on-disk panel and verifies against the audited seed.
#
# Inputs:    standardized firm-month panel, output/seed/corrected_results.json
# Outputs:   output/stage3a/reproduction_check.{json,md}
# Paper:     Intermediate diagnostic
# Run order: see code/00_master.py
# =====================================================================

"""Stage 3a re-validation: reproduce the headline TWFE coefficient.

Per BRIEF.md, the headline result is:
    TWFE state+month fixed effects, two-way state×month clustered (CGM):
    γ̂ = −0.0117 (SE 0.0022, t = −5.32, p < 10⁻⁷) on standardized
    `mom × IV × literacy_z` three-way.

This script reads the on-disk parquet and reproduces the coefficient and
two-way clustered SE. It is the seeded-mode Stage 3a re-validation step
(BRIEF.md item #2): verify `corrected_results.json` numbers reproduce.

We do not re-run robustness; the audited seed already covers them.
"""

import os
import json
import numpy as np
import pandas as pd

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "reproduction_check.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "reproduction_check.md")
SEED_JSON = os.path.join(ROOT, "output", "seed", "corrected_results.json")


def two_way_clustered_se(y, X, cluster_state, cluster_month):
    """CGM two-way clustered standard errors. Mirrors the seed's
    `02_fm_corrected.py::two_way_clustered_se` implementation (state×month
    intersection follows Cameron-Gelbach-Miller 2011, eq. 2.13).
    """
    n, k = X.shape
    XX = X.T @ X
    XX_inv = np.linalg.pinv(XX)
    beta = XX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        meat_mat = np.zeros((k, k))
        for gid in np.unique(g):
            idx = np.where(g == gid)[0]
            Xg = X[idx]
            eg = e[idx]
            Xge = Xg.T @ eg
            meat_mat += np.outer(Xge, Xge)
        return meat_mat

    M_state = meat(cluster_state)
    M_month = meat(cluster_month)
    # state×month intersection cluster id — use a paired hash
    n_months = int(cluster_month.max()) + 1
    inter_codes = cluster_state.astype(np.int64) * n_months + cluster_month.astype(np.int64)
    M_inter = meat(inter_codes)

    V = XX_inv @ (M_state + M_month - M_inter) @ XX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta, se


def main():
    df = pd.read_parquet(PANEL)
    print(f"panel shape: {df.shape}")
    print(f"date range: {df['date'].min()} to {df['date'].max()}")
    print(f"unique permnos: {df['permno'].nunique()}")
    print(f"unique states: {df['hq_state'].nunique()}")
    print(f"unique months: {df['date'].nunique()}")

    # Drop rows missing the focal triple, ret, hq_state, or date
    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected',
              'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
              'mom_x_iv_x_literacy_corr', 'hq_state', 'date']
    d = df.dropna(subset=needed).copy()
    print(f"after dropna: {d.shape}")

    # z-score within month (the seed standardized within month)
    def zscore_within(g, col):
        x = g[col].values
        s = x.std()
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return (x - x.mean()) / s

    feats_to_z = ['mom_12_2', 'iv', 'literacy_score_corrected',
                  'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
                  'mom_x_iv_x_literacy_corr']
    # parquet variables appear to already be standardized — confirm with stds
    pre_std = d[feats_to_z].std()
    print("std of features in parquet (should be ~1 if already standardized):")
    print(pre_std)

    # Use the variables as-is (they are already standardized within month)
    # Build design with state and month dummies + lower-order interactions
    Xcols = ['mom_12_2', 'iv', 'literacy_score_corrected',
             'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
             'mom_x_iv_x_literacy_corr']
    # Create state and month dummies (drop-first to avoid colinearity)
    states = pd.get_dummies(d['hq_state'].astype(str), prefix='S', drop_first=True, dtype=float)
    d['ym'] = d['date'].dt.to_period('M').astype(str)
    months = pd.get_dummies(d['ym'], prefix='M', drop_first=True, dtype=float)

    X = np.hstack([np.ones((len(d), 1)),
                   d[Xcols].values.astype(float),
                   states.values, months.values])
    y = d['ret'].values.astype(float)

    state_codes = pd.Categorical(d['hq_state']).codes
    month_codes = pd.Categorical(d['ym']).codes

    print(f"design shape: {X.shape} (1 intercept + {len(Xcols)} focal + "
          f"{states.shape[1]} state dummies + {months.shape[1]} month dummies)")

    beta, se = two_way_clustered_se(y, X, state_codes, month_codes)

    # Focal three-way is the 8th coefficient (0=intercept, 1-7=focal cols, 7 is mom_x_iv_x_literacy_corr)
    idx_three = 1 + Xcols.index('mom_x_iv_x_literacy_corr')
    coef = float(beta[idx_three])
    se_ = float(se[idx_three])
    t = coef / se_
    from scipy.stats import t as tdist
    df_resid = X.shape[0] - X.shape[1]
    p = 2 * (1 - tdist.cdf(abs(t), df=df_resid))

    print(f"\n=== HEADLINE THREE-WAY REPRODUCED ===")
    print(f"coef:  {coef:.6f}")
    print(f"SE:    {se_:.6f}")
    print(f"t:     {t:.4f}")
    print(f"p:     {p:.3e}")
    print(f"n_obs: {X.shape[0]}")
    print(f"df:    {df_resid}")

    with open(SEED_JSON) as f:
        seed = json.load(f)
    seed_twfe = seed['fm_full']['contemporaneous']['twfe_imp3']

    print(f"\n=== SEED HEADLINE FROM corrected_results.json ===")
    print(f"coef:  {seed_twfe['coef']:.6f}")
    print(f"SE:    {seed_twfe['se']:.6f}")
    print(f"t:     {seed_twfe['t']:.4f}")
    print(f"p:     {seed_twfe['p']:.3e}")
    print(f"n_obs: {seed_twfe['n_obs']}")

    diff_coef = abs(coef - seed_twfe['coef'])
    diff_se = abs(se_ - seed_twfe['se'])
    matches = diff_coef < 1e-4 and diff_se < 1e-4

    res = {
        "reproduced": {
            "coef": coef, "se": se_, "t": t, "p": p, "n_obs": int(X.shape[0])
        },
        "seed_headline": seed_twfe,
        "diff_coef_abs": diff_coef,
        "diff_se_abs": diff_se,
        "matches_to_1e-4": bool(matches),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Stage 3a Reproduction Check (seeded re-validation)\n\n")
        f.write("**Source:** Re-ran `output/seed/corrected_scripts/02_fm_corrected.py` "
                "logic on `output/seed/data/processed/panel_corrected_standardized.parquet` "
                "to verify the headline TWFE three-way coefficient and its two-way "
                "state×month CGM standard error reproduce.\n\n")
        f.write(f"## Reproduced (this script)\n\n")
        f.write(f"- coef: `{coef:.6f}`\n- SE: `{se_:.6f}`\n- t: `{t:.4f}`\n"
                f"- p: `{p:.3e}`\n- n_obs: `{X.shape[0]:,}`\n\n")
        f.write(f"## Seed headline (`output/seed/corrected_results.json`)\n\n")
        f.write(f"- coef: `{seed_twfe['coef']:.6f}`\n- SE: `{seed_twfe['se']:.6f}`\n"
                f"- t: `{seed_twfe['t']:.4f}`\n- p: `{seed_twfe['p']:.3e}`\n"
                f"- n_obs: `{seed_twfe['n_obs']:,}`\n\n")
        f.write(f"## Match\n\n")
        f.write(f"- |Δcoef| = `{diff_coef:.3e}` (threshold 1e-4)\n")
        f.write(f"- |ΔSE|   = `{diff_se:.3e}` (threshold 1e-4)\n")
        f.write(f"- **Matches to 1e-4: `{matches}`**\n\n")
        if matches:
            f.write("**Verdict: PASS.** The headline coefficient and CGM two-way "
                    "clustered SE reproduce from the on-disk panel. The seed's "
                    "audited result is the finding of record.\n")
        else:
            f.write("**Verdict: DISCREPANCY.** Reproduction does not match the seed "
                    "headline to 1e-4. Investigate before paper-writing.\n")
    return matches


if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
