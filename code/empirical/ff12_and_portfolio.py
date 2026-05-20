# =====================================================================
# ff12_and_portfolio.py
# Long-short portfolio magnitude check (quintile Mom x IV x literacy-percentile sort) and FF12 industry-FE confirmation of the headline TWFE coefficient.
#
# Inputs:    standardized firm-month panel
# Outputs:   output/stage3a/ff12_portfolio.{json,md}
# Paper:     FF12 industry / portfolio robustness (IA)
# Run order: see code/00_master.py
# =====================================================================

"""Pre-Stage-5 empirical strengthening tests, per output/stage4/triage_v1.md:

(2) Long-short portfolio magnitude check.
    Quintile sort on `Mom × IV` interacted with state-`literacy_z` percentile.
    Report long-short Sharpe ratio and annualized return.

(3) FF12 headline confirmation.
    Add FF12 industry fixed effects to the headline TWFE; re-run with two-way
    state×month CGM clustering. Reports γ̂ before/after FF12 absorption.

These are the two cheapest items on the triager's pre-Stage-5 list. They use
only the on-disk panel; no external data fetches.
"""

import os
import json
import numpy as np
import pandas as pd

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "ff12_portfolio.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "ff12_portfolio.md")

# FF12 industry mapping from SIC codes (Fama & French 12-industry, public def)
# Source: Ken French Data Library industry definitions
FF12_RANGES = [
    # (industry_code, [(sic_lo, sic_hi), ...])
    (1,  [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),  # NoDur
    (2,  [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
          (3630, 3659), (3700, 3711), (3714, 3714), (3716, 3716), (3718, 3718), (3720, 3729),
          (3732, 3739), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),  # Durbl
    (3,  [(2520, 2589), (2600, 2699), (2750, 2769), (2800, 2829), (2840, 2899), (3000, 3099),
          (3200, 3569), (3580, 3621), (3623, 3629), (3700, 3700), (3712, 3713), (3715, 3715),
          (3717, 3717), (3719, 3724), (3726, 3731), (3732, 3739), (3743, 3743), (3760, 3789),
          (3793, 3799), (3800, 3800), (3860, 3899)]),  # Manuf
    (4,  [(1200, 1399), (2900, 2999)]),  # Enrgy
    (5,  [(2800, 2829), (2840, 2899)]),  # Chems
    (6,  [(3570, 3579), (3622, 3622), (3660, 3692), (3694, 3699), (3810, 3839), (7370, 7372),
          (7373, 7373), (7374, 7374), (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378),
          (7379, 7379), (7391, 7391), (8730, 8734)]),  # BusEq
    (7,  [(4800, 4899)]),  # Telcm
    (8,  [(4900, 4949)]),  # Utils
    (9,  [(5000, 5999), (7200, 7299), (7600, 7699)]),  # Shops
    (10, [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),  # Hlth
    (11, [(6000, 6999)]),  # Money (Finance)
    # 12 = Other (anything not matched above)
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


def two_way_clustered_se(y, X, cluster_state, cluster_month):
    """CGM two-way clustering."""
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
    n_months = int(cluster_month.max()) + 1
    inter_codes = cluster_state.astype(np.int64) * n_months + cluster_month.astype(np.int64)
    M_inter = meat(inter_codes)

    V = XX_inv @ (M_state + M_month - M_inter) @ XX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta, se


def headline_with_ff12(df):
    """Headline TWFE with state + month + FF12 fixed effects, two-way state×month CGM."""
    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected',
              'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
              'mom_x_iv_x_literacy_corr', 'hq_state', 'date', 'siccd']
    d = df.dropna(subset=needed).copy()
    d['ff12'] = d['siccd'].apply(sic_to_ff12)
    print(f"FF12 distribution:\n{d['ff12'].value_counts().sort_index()}")

    Xcols = ['mom_12_2', 'iv', 'literacy_score_corrected',
             'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
             'mom_x_iv_x_literacy_corr']
    states = pd.get_dummies(d['hq_state'].astype(str), prefix='S', drop_first=True, dtype=float)
    d['ym'] = d['date'].dt.to_period('M').astype(str)
    months = pd.get_dummies(d['ym'], prefix='M', drop_first=True, dtype=float)
    ff12 = pd.get_dummies(d['ff12'].astype(str), prefix='I', drop_first=True, dtype=float)

    X = np.hstack([np.ones((len(d), 1)),
                   d[Xcols].values.astype(float),
                   states.values, months.values, ff12.values])
    y = d['ret'].values.astype(float)
    state_codes = pd.Categorical(d['hq_state']).codes
    month_codes = pd.Categorical(d['ym']).codes

    print(f"design shape with FF12: {X.shape} (1 + {len(Xcols)} focal + {states.shape[1]} state + "
          f"{months.shape[1]} month + {ff12.shape[1]} FF12)")
    beta, se = two_way_clustered_se(y, X, state_codes, month_codes)
    idx_three = 1 + Xcols.index('mom_x_iv_x_literacy_corr')
    coef = float(beta[idx_three])
    se_ = float(se[idx_three])
    t = coef / se_
    return {"coef": coef, "se": se_, "t": t, "n_obs": int(X.shape[0]),
            "specification": "TWFE state+month+FF12, two-way state×month CGM"}


def long_short_portfolio(df):
    """Build a long-short portfolio sorted on the three-way `Mom × IV × Lit_z` signal.

    Construction:
      Each month t, sort firms into quintiles on the standardized signal `mom_x_iv_x_literacy_corr`.
      Long bottom quintile (most-negative signal — predicts highest return per the negative γ),
      short top quintile (most-positive signal — predicts lowest return).
      Equal-weighted within quintile.
      Report monthly long-short return series, mean, SD, t-stat, annualized Sharpe.

    A negative γ on `Mom × IV × Lit_z` means: as the standardized three-way value INCREASES,
    expected return DECREASES. So the quintile with the LOWEST signal value has the HIGHEST
    expected return; the quintile with the HIGHEST signal value has the LOWEST. Long Q1, short Q5.
    """
    needed = ['ret', 'mom_x_iv_x_literacy_corr', 'date']
    d = df.dropna(subset=needed).copy()

    def assign_quintile(g):
        x = g['mom_x_iv_x_literacy_corr'].values
        ranks = pd.qcut(x, q=5, labels=False, duplicates='drop')
        return pd.Series(ranks, index=g.index)

    d['q'] = d.groupby(d['date'])['mom_x_iv_x_literacy_corr'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop'))

    # Equal-weighted within quintile, within month
    pf = d.groupby(['date', 'q'])['ret'].mean().reset_index()
    pf_wide = pf.pivot(index='date', columns='q', values='ret')
    pf_wide.columns = [f'q{int(c)}' for c in pf_wide.columns]
    pf_wide['ls'] = pf_wide['q0'] - pf_wide['q4']
    print(f"portfolio months: {len(pf_wide)}")
    print(pf_wide.describe())

    ls = pf_wide['ls'].dropna()
    mean = float(ls.mean())
    sd = float(ls.std(ddof=1))
    t = mean / (sd / np.sqrt(len(ls)))
    sharpe_annual = mean / sd * np.sqrt(12)
    annualized_return = (1 + mean) ** 12 - 1

    return {
        "n_months": int(len(ls)),
        "monthly_mean_ret": mean,
        "monthly_sd": sd,
        "t_stat": float(t),
        "sharpe_annual": float(sharpe_annual),
        "annualized_return": float(annualized_return),
        "q0_mean": float(pf_wide['q0'].mean()),
        "q4_mean": float(pf_wide['q4'].mean()),
        "interpretation": "Long Q1 (most negative signal), short Q5 (most positive signal). "
                          "Per negative γ: low-signal firms expected high return, high-signal firms "
                          "expected low return."
    }


def main():
    df = pd.read_parquet(PANEL)
    print(f"panel shape: {df.shape}")

    print("\n=== (3) FF12 HEADLINE CONFIRMATION ===")
    ff12_res = headline_with_ff12(df)
    print(f"With FF12 absorbed: coef = {ff12_res['coef']:.6f}, "
          f"SE = {ff12_res['se']:.6f}, t = {ff12_res['t']:.4f}")
    seed_baseline = {"coef": -0.011727, "se": 0.002205, "t": -5.319,
                     "specification": "TWFE state+month, two-way state×month CGM (seed baseline)"}
    delta_pct = (ff12_res['coef'] - seed_baseline['coef']) / abs(seed_baseline['coef']) * 100
    print(f"Seed baseline (no FF12): coef = {seed_baseline['coef']:.6f}, "
          f"SE = {seed_baseline['se']:.6f}, t = {seed_baseline['t']:.4f}")
    print(f"Delta: {delta_pct:+.2f}% on coefficient")

    print("\n=== (2) LONG-SHORT PORTFOLIO MAGNITUDE ===")
    pf_res = long_short_portfolio(df)
    print(f"Monthly mean: {pf_res['monthly_mean_ret']*100:.3f}%")
    print(f"Monthly SD:   {pf_res['monthly_sd']*100:.3f}%")
    print(f"t-stat:       {pf_res['t_stat']:.2f}")
    print(f"Sharpe (ann): {pf_res['sharpe_annual']:.3f}")
    print(f"Annualized return: {pf_res['annualized_return']*100:.2f}%")

    res = {
        "ff12_headline": ff12_res,
        "ff12_seed_baseline": seed_baseline,
        "ff12_delta_pct_on_coef": delta_pct,
        "long_short_portfolio": pf_res,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Pre-Stage-5 Empirical Strengthening: FF12 Absorption + Long-Short Portfolio\n\n")
        f.write("Per `output/stage4/triage_v1.md` items (3) FF12 headline confirmation and (2) "
                "long-short portfolio magnitude check.\n\n")

        f.write("## (3) FF12 industry fixed effects added to headline TWFE\n\n")
        f.write("| Specification | Coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Seed baseline (state + month FE) | {seed_baseline['coef']:.6f} | "
                f"{seed_baseline['se']:.6f} | {seed_baseline['t']:.4f} | 623,896 |\n")
        f.write(f"| **+ FF12 industry FE** | **{ff12_res['coef']:.6f}** | "
                f"**{ff12_res['se']:.6f}** | **{ff12_res['t']:.4f}** | "
                f"**{ff12_res['n_obs']:,}** |\n\n")
        f.write(f"**Coefficient change vs. seed baseline:** {delta_pct:+.2f}% "
                f"({'attenuates' if abs(ff12_res['coef']) < abs(seed_baseline['coef']) else 'amplifies'} "
                f"when FF12 industry composition is absorbed).\n\n")
        f.write("**Interpretation.** ")
        if abs(ff12_res['t']) >= 3.0:
            f.write("The headline TWFE coefficient survives FF12 industry absorption. The Cinelli-Hazlett "
                    "RV concern (RV = 0.0040 vs. recalibrated threshold 0.0487 at FF12 block) is "
                    "addressed by *direct absorption* of the FF12 block, not by sensitivity bound. "
                    "An FF12-magnitude state×industry confounder cannot drive the result because the "
                    "industry block is now in the regression.\n\n")
        else:
            f.write("The headline TWFE coefficient is materially attenuated by FF12 industry absorption. "
                    "The result is *not* robust to FF12 industry FE; the channel is partially "
                    "industry-composition driven, not purely state-literacy driven. The paper must "
                    "disclose this and downgrade the headline accordingly.\n\n")

        f.write("## (2) Long-short portfolio on the standardized three-way signal\n\n")
        f.write(f"Construction: equal-weighted within-month quintile sort on `mom × IV × literacy_z`. "
                f"Long Q1 (most-negative signal — high expected return per negative γ), short Q5 "
                f"(most-positive signal). Monthly rebalanced.\n\n")
        f.write("| Quantity | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Months | {pf_res['n_months']} |\n")
        f.write(f"| Q1 mean monthly return | {pf_res['q0_mean']*100:.3f}% |\n")
        f.write(f"| Q5 mean monthly return | {pf_res['q4_mean']*100:.3f}% |\n")
        f.write(f"| Long-short mean monthly | {pf_res['monthly_mean_ret']*100:.3f}% |\n")
        f.write(f"| Long-short monthly SD | {pf_res['monthly_sd']*100:.3f}% |\n")
        f.write(f"| Long-short t-stat | {pf_res['t_stat']:.2f} |\n")
        f.write(f"| Annualized Sharpe | {pf_res['sharpe_annual']:.3f} |\n")
        f.write(f"| Annualized return | {pf_res['annualized_return']*100:.2f}% |\n\n")
        f.write("**Interpretation.** The implied long-short Sharpe is the magnitude check the "
                "self-attack item (severity 4-6) called for. ")
        if pf_res['sharpe_annual'] > 1.0:
            f.write(f"Annualized Sharpe of {pf_res['sharpe_annual']:.2f} would be a notable "
                    f"economic anomaly; the regression magnitude is consistent with a real, "
                    f"economically interesting effect.\n")
        elif pf_res['sharpe_annual'] > 0.3:
            f.write(f"Annualized Sharpe of {pf_res['sharpe_annual']:.2f} indicates a modest "
                    f"economic effect — comparable to or smaller than published anomaly Sharpes "
                    f"(MOM ~0.5-0.8, BAB ~0.7). The regression magnitude is consistent with a "
                    f"small but real signal.\n")
        else:
            f.write(f"Annualized Sharpe of {pf_res['sharpe_annual']:.2f} is small. The regression "
                    f"magnitude does NOT translate into a tradable anomaly of meaningful size — "
                    f"the within-state-month statistical significance comes from very large n, "
                    f"not from a large per-firm effect. The paper's magnitude story must be "
                    f"calibrated to this Sharpe, not to the regression coefficient alone.\n")

    return res


if __name__ == '__main__':
    res = main()
    print("\nDone — wrote", OUT_JSON, "and", OUT_MD)
