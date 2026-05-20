# =====================================================================
# v7_goodman_bacon.py
# Goodman-Bacon (2021) 2x2 decomposition of the staggered-adoption DiD (state
# personal-finance high-school graduation mandates) on the four-way standardized
# outcome mom*IV*literacy_z, reporting weight shares by comparison type.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_goodman_bacon.json (no tables written)
# Paper:     Internet Appendix difference-in-differences section + Main Table T9
#            tab:fs_iv region (DiD diagnostics)
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Q6 (referee structured Q7; triager row 31): Goodman-Bacon
decomposition for the secondary staggered-adoption DiD around state
personal-finance high-school graduation mandates.

Treatment cohorts (per IA Section H):
  In-window treated: MO ~2009, TN ~2009, VA ~2011, FL 2022
  Pre-window adopters (aging into investing during the sample, treated
    throughout the sample): TX 2007, UT 2008, ID 2008, GA 2007
  Never-treated: all other states (43 states)

The simple two-way fixed-effects DiD estimator on the four-way outcome
(mom*IV*literacy_z) is a weighted average of comparisons (Goodman-Bacon 2021):
  - earlier-treated vs later-treated (later as "control")
  - later-treated vs earlier-treated (earlier as "control")
  - treated vs never-treated
  - treated vs always-treated (here, the pre-window adopters)

This decomposition tells us whether the not-yet-treated comparisons (which
are the "good" comparisons under staggered DiD) are dominating, or whether
the already-treated comparisons (which can be contaminated under TWFE) are
dominating.

We implement the bacondecomp algorithm directly for this 2x2 setup.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)

DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_goodman_bacon.json")

# Treatment-year map (per IA Section H)
TREAT_YR = {
    'US-MO': 2009, 'US-TN': 2009, 'US-VA': 2011, 'US-FL': 2022,
    # pre-window adopters: treated throughout the sample window
    'US-TX': 2007, 'US-UT': 2008, 'US-ID': 2008, 'US-GA': 2007,
}


def assign_treatment_year(state):
    if state in TREAT_YR:
        return TREAT_YR[state]
    return np.nan  # never-treated


def bacon_decomp(d, outcome_col, time_col, unit_col, treat_year_col):
    """Compute the Goodman-Bacon (2021) 2x2 decomposition. Returns a list of
    pair-weighted ATTs.

    The decomposition partitions the TWFE DiD estimator into:
      (k vs l-as-control, l vs k-as-control, k vs never-treated, etc.)
    Each 2x2 pair (g1, g2) contributes a weight and a 2x2 DiD ATT.

    Reference: bacondecomp Stata/R package; Goodman-Bacon 2021 AER.

    Implementation: given groups defined by treat_year_col (NaN = never-
    treated), we form, for each pair of distinct treatment-year groups, the
    2x2 ATT and its weight.
    """
    # collapse to unit-time means of the outcome
    df = d[[unit_col, time_col, outcome_col, treat_year_col]].copy()
    df = df.dropna(subset=[outcome_col])
    # mean outcome per (unit, time)
    g = df.groupby([unit_col, time_col, treat_year_col],
                   dropna=False)[outcome_col].mean().reset_index()
    # ensure each unit's treat_year_col is constant; for now just take first
    unit_treat = g.groupby(unit_col)[treat_year_col].first()
    g[treat_year_col] = g[unit_col].map(unit_treat)

    # distinct treatment cohorts (including never-treated as nan)
    cohorts = sorted([c for c in g[treat_year_col].dropna().unique()])
    has_never = g[treat_year_col].isnull().any()

    # 2x2 utility
    def two_x_two(g1, g2, t_pre_max=None, t_post_min=None):
        """Compute the 2x2 DiD ATT between cohort g1 (treated earlier or only)
        and cohort g2 (control), with pre/post windows defined by g1's
        treatment year (and g2's later treatment year if it has one).
        Returns (att, weight, n_periods, mean_outcome_diff).
        """
        sub1 = g[g[treat_year_col] == g1] if not np.isnan(g1) else \
            g[g[treat_year_col].isnull()]
        sub2 = g[g[treat_year_col] == g2] if not np.isnan(g2) else \
            g[g[treat_year_col].isnull()]
        if len(sub1) == 0 or len(sub2) == 0:
            return None
        # Pre/post definition for g1 vs g2:
        # If g2 is never-treated: pre = years < g1, post = years >= g1
        # If g2 treated at year y2 > g1: pre = years < g1, post = g1 <= years
        # < y2 (g2 still untreated)
        # If g2 treated at year y2 < g1: pre = years < y2 (here pre means
        # g1 not yet treated, g2 ALREADY treated — flip roles below)
        t1 = g1
        t2 = g2
        if np.isnan(t2):
            pre = g[g[time_col] < t1]
            post = g[(g[time_col] >= t1)]
            comp_label = 'g_vs_never'
        elif t2 > t1:
            pre = g[g[time_col] < t1]
            post = g[(g[time_col] >= t1) & (g[time_col] < t2)]
            comp_label = 'earlier_vs_later_treated'
        else:
            pre = g[(g[time_col] >= t2) & (g[time_col] < t1)]
            post = g[g[time_col] >= t1]
            comp_label = 'later_vs_earlier_treated'

        sub1_pre = pre[pre[treat_year_col] == t1] if not np.isnan(t1) \
            else pre[pre[treat_year_col].isnull()]
        sub2_pre = pre[pre[treat_year_col] == t2] if not np.isnan(t2) \
            else pre[pre[treat_year_col].isnull()]
        sub1_post = post[post[treat_year_col] == t1] if not np.isnan(t1) \
            else post[post[treat_year_col].isnull()]
        sub2_post = post[post[treat_year_col] == t2] if not np.isnan(t2) \
            else post[post[treat_year_col].isnull()]
        if (len(sub1_pre) == 0 or len(sub2_pre) == 0 or
                len(sub1_post) == 0 or len(sub2_post) == 0):
            return None

        # ATT = (Y1_post - Y1_pre) - (Y2_post - Y2_pre)
        att = ((sub1_post[outcome_col].mean()
                - sub1_pre[outcome_col].mean())
               - (sub2_post[outcome_col].mean()
                  - sub2_pre[outcome_col].mean()))

        # weights (Goodman-Bacon 2021):
        #   w = (n_pair / N) * ((post_share)*(1-post_share)) * (V_D
        # The simpler version: n_pair * VarD_pair where
        # D = treatment indicator over the (g1+g2) sub-panel
        # VarD = D_bar*(1-D_bar)
        # We compute n_pair, post_share (= n_post/(n_pre+n_post)) of the
        # pair, and VarD = post_share * (1 - post_share). This is the
        # within-pair contribution to the TWFE total variance of D.
        n_pre = len(sub1_pre) + len(sub2_pre)
        n_post = len(sub1_post) + len(sub2_post)
        n_pair = n_pre + n_post
        post_share = n_post / n_pair
        # Var(D) within the pair: D=1 on (g1 post) treated, 0 elsewhere.
        d_bar = len(sub1_post) / n_pair
        var_D = d_bar * (1 - d_bar)
        return {'att': float(att), 'n_pair': int(n_pair),
                'var_D': float(var_D),
                'n_post': int(n_post),
                'comp_label': comp_label,
                'pre_pair': float(n_pre),
                'g1_treat_year': float(t1) if not np.isnan(t1) else None,
                'g2_treat_year': float(t2) if not np.isnan(t2) else None,
                'g1_n_units': int(g[g[treat_year_col] == t1][unit_col].
                                  nunique() if not np.isnan(t1)
                                  else g[g[treat_year_col].isnull()]
                                  [unit_col].nunique()),
                'g2_n_units': int(g[g[treat_year_col] == t2][unit_col].
                                  nunique() if not np.isnan(t2)
                                  else g[g[treat_year_col].isnull()]
                                  [unit_col].nunique())}

    pairs = []
    # All ordered pairs (g1, g2) with g1 treated and g2 either treated or
    # never-treated, where the "treatment" in the 2x2 sense is g1 turning on
    # while g2 is the comparison (either never-treated or not-yet-treated).
    for g1 in cohorts:
        # g1 vs never-treated
        if has_never:
            r = two_x_two(g1, np.nan)
            if r is not None:
                pairs.append(r)
        # g1 (earlier) vs g2 (later) — earlier as treated, later as not-yet-
        # treated control
        for g2 in cohorts:
            if g2 <= g1:
                continue
            r = two_x_two(g1, g2)
            if r is not None:
                pairs.append(r)
        # g1 (later) vs g2 (earlier) — later as treated, earlier as already-
        # treated (problematic) control
        for g2 in cohorts:
            if g2 >= g1:
                continue
            r = two_x_two(g1, g2)
            if r is not None:
                pairs.append(r)
    # Weights: raw_w_p = n_p * var_D_p. Total ATT_TWFE = sum(raw_w * ATT) /
    # sum(raw_w). Each pair's weight share is raw_w_p / sum(raw_w).
    if not pairs:
        return {'pairs': [], 'twfe_implied_att': None,
                'weight_share_by_comp_type': {}}
    raw_w = np.array([p['n_pair'] * p['var_D'] for p in pairs])
    total_w = float(raw_w.sum())
    for k, p in enumerate(pairs):
        p['weight_share'] = float(raw_w[k] / total_w)
    twfe_att = float(sum(p['weight_share'] * p['att'] for p in pairs))
    share_by = {}
    for p in pairs:
        c = p['comp_label']
        share_by.setdefault(c, 0.0)
        share_by[c] += p['weight_share']
    return {'pairs': pairs, 'twfe_implied_att': twfe_att,
            'weight_share_by_comp_type': share_by,
            'total_raw_weight': total_w}


def main():
    print("=== Q7 Goodman-Bacon decomposition for secondary DiD (v7) ===",
          flush=True)
    dfm = pd.read_parquet(DFM_CACHE)
    print(f"merged panel: {len(dfm):,} firm-months, "
          f"{dfm['hq_state'].nunique()} states", flush=True)

    dfm['year'] = dfm['date'].dt.year
    dfm['treat_year'] = dfm['hq_state'].map(assign_treatment_year)
    # Outcome: the four-way standardized term (mom * IV * literacy_z).
    # Aggregating to state-year via firm-month means.
    dfm['four_way_term'] = (
        dfm['mom_12_2'] * dfm['iv'] * dfm['literacy_score_corrected'])
    state_year = (dfm.dropna(subset=['four_way_term', 'hq_state'])
                  .groupby(['hq_state', 'year', 'treat_year'],
                           dropna=False)['four_way_term']
                  .mean().reset_index())
    print(f"state-year cells: {len(state_year):,}", flush=True)

    cohorts = state_year.dropna(subset=['treat_year'])
    print(f"treated states: {cohorts['hq_state'].nunique()}", flush=True)
    print(f"never-treated states: "
          f"{state_year[state_year['treat_year'].isnull()]['hq_state'].nunique()}",
          flush=True)
    print(f"cohort years: "
          f"{sorted(cohorts['treat_year'].unique())}", flush=True)

    # Run decomp on this state-year aggregation
    bacon = bacon_decomp(state_year, outcome_col='four_way_term',
                         time_col='year', unit_col='hq_state',
                         treat_year_col='treat_year')

    print(f"\n=== Goodman-Bacon decomposition ===", flush=True)
    print(f"TWFE-implied ATT (re-aggregated): "
          f"{bacon['twfe_implied_att']:+.6e}", flush=True)
    print(f"Weight share by comparison type:")
    for k, v in bacon['weight_share_by_comp_type'].items():
        print(f"  {k:>32}: {v*100:.2f}%", flush=True)
    print(f"\nTop pairs by weight:")
    top = sorted(bacon['pairs'], key=lambda p: -p['weight_share'])[:15]
    for p in top:
        print(f"  g1={p.get('g1_treat_year')}, g2={p.get('g2_treat_year')}, "
              f"comp={p['comp_label']}, ATT={p['att']:+.4e}, "
              f"w_share={p['weight_share']*100:.2f}%, "
              f"n_units g1={p['g1_n_units']} g2={p['g2_n_units']}",
              flush=True)

    results = {
        'task': 'Q7 (referee structured Q7; triager row 31): Goodman-Bacon '
                '(2021) decomposition for the secondary staggered-adoption '
                'DiD on the four-way standardized outcome (mom*IV*lit_z).',
        'treatment_year_map': {k: int(v) for k, v in TREAT_YR.items()},
        'sample': {
            'state_year_cells': int(len(state_year)),
            'states_total': int(state_year['hq_state'].nunique()),
            'states_treated': int(cohorts['hq_state'].nunique()),
            'states_never_treated': int(
                state_year[state_year['treat_year'].isnull()]
                ['hq_state'].nunique()),
            'cohort_years': sorted(cohorts['treat_year'].unique().tolist()),
        },
        'twfe_implied_att': bacon['twfe_implied_att'],
        'weight_share_by_comp_type': bacon['weight_share_by_comp_type'],
        'pairs': bacon['pairs'],
        'note': 'The four-way standardized outcome is mom * IV * '
                'literacy_z; this is the IA Section H outcome construction. '
                'The TWFE-implied ATT computed here is the simple state-year '
                'TWFE DiD on the AGGREGATED outcome (not the within-firm-'
                'month panel); it is the "first" object the Goodman-Bacon '
                'decomposition partitions.',
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
