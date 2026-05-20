# =====================================================================
# deepen_college_horserace.py
# Literacy-vs-college horse-race: adds a state college-attainment three-way block to the headline TWFE and reports the literacy triple before vs. after to discipline the education confounder.
#
# Inputs:    standardized firm-month panel, FRED state bachelor's-degree share (GCT1502*)
# Outputs:   output/stage3a/deepen_college_horserace.{json,md} + output/stage3a/tables/tab_college_horserace.tex
# Paper:     tab_college_horserace.tex (literacy vs college horse-race, mid-IO; main results/robustness)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive Item 3 — College-attainment horse-race.

The paper names state college-attainment share as "the most proximate
uncontrolled state characteristic." This script downloads annual state-level
bachelor's-degree-or-higher share (FRED GCT1502{ST}, percent of population 25+
with a bachelor's degree or higher, 2008-2024), merges to the panel by
state-year, z-scores it within month, builds the college-attainment three-way
interaction `mom x IV x college_z` and its lower-order interactions, and re-runs
the headline TWFE specification adding the full college-attainment block.

The test: report the literacy three-way coefficient before vs. after adding the
college block. If literacy_z is merely proxying education, the literacy triple
collapses; if it survives, the paper has disciplined its primary confounder.

The college block is the education analogue of the literacy interaction terms:
  college_z, mom*college_z, iv*college_z, mom*iv*college_z
(all z-scored within month, mirroring the seed convention).

Output: output/stage3a/deepen_college_horserace.{json,md}
      + output/stage3a/tables/tab_college_horserace.tex
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

np.random.seed(42)

sys.path.insert(0, r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical/code")
from utils.fred_utils import get_series

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "deepen_college_horserace.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "deepen_college_horserace.md")
TABLE = os.path.join(ROOT, "output", "stage3a", "tables", "tab_college_horserace.tex")
CACHE = os.path.join(ROOT, "code", "empirical", "_fred_college_cache.parquet")

FOCAL = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
         'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']

STATE_CODES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
               'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
               'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
               'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
               'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']


def fetch_college_shares():
    """FRED GCT1502{ST}: % of population 25+ with bachelor's degree or higher,
    annual, by state. 2008-2024. Cached."""
    if os.path.exists(CACHE):
        print(f"loading cached college shares from {CACHE}")
        return pd.read_parquet(CACHE)
    rows = []
    for st in STATE_CODES:
        sid = f"GCT1502{st}"
        for attempt in range(5):
            try:
                s = get_series(sid, start="2007-01-01", end="2024-12-31")
                for dt, val in s.items():
                    rows.append({"st": st, "year": pd.Timestamp(dt).year,
                                 "college_share": float(val)})
                print(f"  {sid}: {len(s)} obs ({s.iloc[0]:.1f} -> {s.iloc[-1]:.1f})")
                break
            except Exception as e:
                if attempt == 4:
                    print(f"  {sid}: FAILED after 5 tries: {str(e)[:80]}")
                else:
                    time.sleep(1.5)
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"cached {len(df)} state-year college rows ({df['st'].nunique()} "
          f"states) to {CACHE}")
    return df


def two_way_clustered(d, focal_cols, target='mom_x_iv_x_literacy_corr'):
    """TWFE state+month FE, two-way state x month CGM clustering. Returns the
    coefficient + SE on `target`."""
    n = len(d)
    sd = pd.get_dummies(d['hq_state'].astype(str), prefix='S',
                        drop_first=True, dtype=float)
    md = pd.get_dummies(d['ym'], prefix='M', drop_first=True, dtype=float)
    X = np.hstack([np.ones((n, 1)), d[focal_cols].values.astype(float),
                   sd.values, md.values])
    y = d['ret'].values.astype(float)
    k = X.shape[1]
    XX_inv = np.linalg.pinv(X.T @ X)
    beta = XX_inv @ (X.T @ y)
    e = y - X @ beta
    g_state = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    g_month = pd.Categorical(d['ym']).codes.astype(np.int64)

    def meat(g):
        m = np.zeros((k, k))
        for gid in np.unique(g):
            ix = np.where(g == gid)[0]
            s = X[ix].T @ e[ix]
            m += np.outer(s, s)
        return m

    n_months = int(g_month.max()) + 1
    inter = g_state.astype(np.int64) * n_months + g_month.astype(np.int64)
    V = XX_inv @ (meat(g_state) + meat(g_month) - meat(inter)) @ XX_inv
    idx = 1 + focal_cols.index(target)
    coef = float(beta[idx])
    se = float(np.sqrt(max(V[idx, idx], 0.0)))
    return {"coef": coef, "se": se, "t": coef / se, "n_obs": int(n),
            "n_params": int(k)}


def main():
    print("=== fetching state college-attainment shares from FRED ===")
    college = fetch_college_shares()
    print(f"college panel: {college.shape}, "
          f"{college['st'].nunique()} states, "
          f"years {college['year'].min()}-{college['year'].max()}")

    print("\n=== loading firm-month panel ===")
    df = pd.read_parquet(PANEL)
    needed = FOCAL + ['ret', 'hq_state', 'date', 'mom_12_2', 'iv', 'mom_x_iv']
    d = df.dropna(subset=needed).copy()
    d['ym'] = d['date'].dt.to_period('M').astype(str)
    d['year'] = d['date'].dt.year
    d['st'] = d['hq_state'].astype(str).str.replace('US-', '', regex=False)

    # merge college share by state-year
    d = d.merge(college, on=['st', 'year'], how='left')
    miss = d['college_share'].isna().sum()
    print(f"after college merge: {d.shape}; college_share missing: {miss} "
          f"({miss/len(d)*100:.2f}%)")
    if miss > 0:
        miss_states = d.loc[d['college_share'].isna(), 'st'].value_counts()
        print(f"  missing-state breakdown:\n{miss_states}")
    d = d.dropna(subset=['college_share']).copy()
    print(f"estimation sample after dropping missing college: {d.shape}")

    # build the college-attainment block: z-score college_share within month,
    # then build interactions and z-score those within month (seed convention)
    def zwithin(s, by):
        return s.groupby(by).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    d['college_z'] = zwithin(d['college_share'], d['ym'])
    d['mom_x_college'] = d['mom_12_2'] * d['college_z']
    d['iv_x_college'] = d['iv'] * d['college_z']
    d['mom_x_iv_x_college'] = d['mom_x_iv'] * d['college_z']
    for c in ['mom_x_college', 'iv_x_college', 'mom_x_iv_x_college']:
        d[c] = zwithin(d[c], d['ym'])

    # correlation between literacy_z and college_z (the confounding question)
    corr_lit_col = float(d[['literacy_score_corrected', 'college_z']].corr().iloc[0, 1])
    # also state-level correlation of the raw state means
    state_means = d.groupby('st').agg(
        lit=('literacy_score_corrected', 'mean'),
        col=('college_share', 'mean')).reset_index()
    corr_state = float(state_means[['lit', 'col']].corr().iloc[0, 1])
    print(f"\nwithin-panel corr(literacy_z, college_z) = {corr_lit_col:.4f}")
    print(f"state-level corr(mean literacy_z, mean college_share) = "
          f"{corr_state:.4f}")

    base_cols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
                 'mom_x_literacy_corr', 'iv_x_literacy_corr',
                 'mom_x_iv_x_literacy_corr']
    college_cols = base_cols + ['college_z', 'mom_x_college', 'iv_x_college',
                                'mom_x_iv_x_college']

    print("\n=== running specifications ===")
    spec_base = two_way_clustered(d, base_cols)
    print(f"  baseline (college-merged sample): "
          f"lit-three-way coef={spec_base['coef']:.6f}, "
          f"se={spec_base['se']:.6f}, t={spec_base['t']:.4f}")
    spec_col = two_way_clustered(d, college_cols)
    print(f"  + college block: "
          f"lit-three-way coef={spec_col['coef']:.6f}, "
          f"se={spec_col['se']:.6f}, t={spec_col['t']:.4f}")
    # also report the college three-way itself in the joint spec
    spec_col_target = two_way_clustered(d, college_cols,
                                        target='mom_x_iv_x_college')
    print(f"  + college block: "
          f"COLLEGE-three-way coef={spec_col_target['coef']:.6f}, "
          f"se={spec_col_target['se']:.6f}, t={spec_col_target['t']:.4f}")

    delta = spec_col['coef'] - spec_base['coef']
    delta_in_se = delta / spec_base['se']
    retained = spec_col['coef'] / spec_base['coef']
    sign_survives = np.sign(spec_col['coef']) == np.sign(spec_base['coef'])
    still_sig = abs(spec_col['t']) > 1.96
    subsumed = (not still_sig) or (not sign_survives)

    print(f"\n{'='*62}")
    print(f"literacy three-way: baseline {spec_base['coef']:.6f} "
          f"(t={spec_base['t']:.2f}) -> +college {spec_col['coef']:.6f} "
          f"(t={spec_col['t']:.2f})")
    print(f"delta = {delta:+.6f} ({delta_in_se:+.3f} baseline SEs); "
          f"retains {retained:.1%} of baseline coef")
    print(f"sign survives: {sign_survives} | still sig at 5%: {still_sig}")
    print(f"literacy triple {'SUBSUMED by' if subsumed else 'SURVIVES'} "
          f"the college block")
    print(f"{'='*62}")

    res = {
        "college_data": {
            "source": "FRED GCT1502{ST} — % of population 25+ with bachelor's "
                      "degree or higher, annual, by state",
            "n_state_years": int(len(college)),
            "n_states_covered": int(college['st'].nunique()),
            "year_range": [int(college['year'].min()),
                           int(college['year'].max())],
        },
        "merge": {
            "panel_obs_before": int(len(df.dropna(subset=needed))),
            "panel_obs_after_merge": int(len(d)),
            "corr_literacy_z_college_z_within_panel": corr_lit_col,
            "corr_state_mean_literacy_college": corr_state,
        },
        "horserace": {
            "baseline_college_merged_sample": spec_base,
            "with_college_block": spec_col,
            "college_three_way_in_joint_spec": spec_col_target,
            "delta_literacy_three_way": delta,
            "delta_in_baseline_se": delta_in_se,
            "fraction_of_baseline_retained": retained,
            "literacy_sign_survives": bool(sign_survives),
            "literacy_still_sig_5pct": bool(still_sig),
            "literacy_subsumed_by_college": bool(subsumed),
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(TABLE, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Specification & literacy three-way & CGM SE & $t$ & $N$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"Baseline (college-merged sample) & ${spec_base['coef']:.4f}$ & "
                f"${spec_base['se']:.4f}$ & ${spec_base['t']:.2f}$ & "
                f"${spec_base['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad + college-attainment block & ${spec_col['coef']:.4f}$ & "
                f"${spec_col['se']:.4f}$ & ${spec_col['t']:.2f}$ & "
                f"${spec_col['n_obs']:,}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\quad (college three-way, joint spec) & "
                f"${spec_col_target['coef']:.4f}$ & "
                f"${spec_col_target['se']:.4f}$ & "
                f"${spec_col_target['t']:.2f}$ & --- \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Deepen Item 3 — College-Attainment Horse-Race\n\n")
        f.write("## College-attainment data\n\n")
        f.write(f"- Source: FRED `GCT1502{{ST}}` — percent of population 25+ "
                f"with a bachelor's degree or higher, annual, by state.\n")
        f.write(f"- Coverage: {college['st'].nunique()} states, "
                f"{college['year'].min()}-{college['year'].max()}, "
                f"{len(college)} state-years.\n")
        f.write(f"- Merge: panel obs {len(df.dropna(subset=needed)):,} -> "
                f"{len(d):,} after the state-year college merge "
                f"(missing-college rows dropped).\n\n")
        f.write("## Confounding diagnostic\n\n")
        f.write(f"- Within-panel correlation `corr(literacy_z, college_z)` = "
                f"`{corr_lit_col:.4f}`.\n")
        f.write(f"- State-level correlation of mean literacy vs. mean "
                f"college-attainment share = `{corr_state:.4f}`.\n\n")
        f.write("## Horse-race result\n\n")
        f.write("| Specification | literacy three-way | SE | t | N |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Baseline (college-merged sample) | "
                f"{spec_base['coef']:.6f} | {spec_base['se']:.6f} | "
                f"{spec_base['t']:.4f} | {spec_base['n_obs']:,} |\n")
        f.write(f"| **+ college-attainment block** | "
                f"**{spec_col['coef']:.6f}** | **{spec_col['se']:.6f}** | "
                f"**{spec_col['t']:.4f}** | **{spec_col['n_obs']:,}** |\n\n")
        f.write(f"College three-way coefficient in the joint specification: "
                f"`{spec_col_target['coef']:.6f}` "
                f"(t = `{spec_col_target['t']:.4f}`).\n\n")
        f.write(f"**Change in the literacy three-way:** {delta:+.6f} "
                f"({delta_in_se:+.3f} baseline SEs); the literacy triple "
                f"retains {retained:.1%} of its baseline magnitude.\n\n")
        f.write("## Verdict\n\n")
        if subsumed:
            f.write(f"**The literacy triple is SUBSUMED by college attainment.** "
                    f"After adding the college-attainment block, the literacy "
                    f"three-way is {spec_col['coef']:.6f} (t = "
                    f"{spec_col['t']:.2f}) — "
                    f"{'sign-reversed' if not sign_survives else 'no longer significant at 5%'}. "
                    f"The headline literacy moderation is, to a first "
                    f"approximation, an education-attainment effect. The paper "
                    f"must report this honestly: state financial literacy as "
                    f"measured by the NFCS does not survive a horse-race against "
                    f"the most proximate state-level confounder.\n")
        else:
            f.write(f"**The literacy triple SURVIVES the college horse-race.** "
                    f"After adding the full college-attainment interaction "
                    f"block, the literacy three-way is {spec_col['coef']:.6f} "
                    f"(t = {spec_col['t']:.2f}), retaining {retained:.0%} of "
                    f"its baseline magnitude with the sign intact and "
                    f"significance preserved. State financial literacy is not "
                    f"merely proxying college attainment; the paper has "
                    f"disciplined its primary confounder.\n")
    print(f"\njson -> {OUT_JSON}\nmd   -> {OUT_MD}\ntex  -> {TABLE}")
    return res


if __name__ == '__main__':
    main()
