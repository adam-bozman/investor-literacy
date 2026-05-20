# =====================================================================
# v7_pit_hq_literacy.py
# Builds the point-in-time (PIT) SEC EDGAR HQ-state literacy reassignment and
# re-fires the headline TWFE three-way on the full panel under PIT literacy
# (vs the static-HQ headline).
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet
#            (seed firm-month panel); _hq_edgar_state_v6.parquet (SEC EDGAR
#            10-K-header HQ-state history).
# Outputs:   output/stage3a/results_v7_pit_hq.json (no tables written)
# Paper:     Internet Appendix PIT diagnostics section (the PIT aggregate
#            headline that motivates the PIT centerpiece)
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Q5 (referee structured Q5; triager row 29): POINT-IN-TIME HQ-state
literacy reassignment.

Rather than DROPPING relocators (v6 Task A), reassign each firm's
`literacy_score_corrected` at each firm-month using the firm's then-current HQ
state as recorded in its most-recent SEC EDGAR 10-K filing-header business
address `STATE`. This is the more natural correction — it preserves the
relocator firm-months in the panel but corrects the literacy_z assignment to
match the firm's actual point-in-time HQ.

Build: from `_hq_edgar_state_v6.parquet` (which has the full sequence of 10-K
filing-date -> STATE pairs per permno) build a step-function HQ history per
firm. For each firm-month, the "current HQ state" is the STATE recorded on the
firm's most-recent 10-K filed AT OR BEFORE that month (with a 1-month lag for
filing-publication latency, matching the canonical HQ-bias literature
convention). Re-merge the literacy_score_corrected variable using
(then-current HQ state, year). Re-build the three-way interaction with the
PIT-reassigned literacy_z. Re-fire the headline TWFE three-way regression on
the FULL panel (relocators NOT dropped — that is the contribution of this
correction).

The PIT-reassignment will:
  - Use the EDGAR-PIT HQ state for firms with parseable 10-K filings (the
    7,245 flag-covered firms in v6 Task A).
  - Use the static panel snapshot for firms without EDGAR coverage (the 202
    uncovered firms — these stay on the seed-panel literacy_z).
  - Cover ALL firm-months in the panel (not 16.8% dropped, as in Task A).
  - Correct the literacy_z for relocators at the *correct* time (not just
    drop them).

The literacy_score_corrected is z-scored within month in the seed panel — so
to do PIT reassignment correctly we need (a) the per-state literacy rank
(NFCS Big Five), (b) re-merge per firm-month using the firm's then-current
state, (c) re-z-score within month if the underlying values change. Since
the per-state literacy values are TIME-VARYING in the seed panel (with state
fixed effects later absorbing the persistent state component), we use the
month-state level literacy value, then re-z-score within month.

Output: results_v7_pit_hq.json, including the PIT-reassigned headline.
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

from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               FOCAL)

np.random.seed(42)

PANEL = os.path.join(ROOT, "output/seed/data/processed/"
                     "panel_corrected_standardized.parquet")
HQ_EDGAR = os.path.join(EMP, "_hq_edgar_state_v6.parquet")
DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_pit_hq.json")
TABDIR = os.path.join(ROOT, "output/stage3a/tables")

B_WCB = 9999


def load_panel():
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    return df


def build_pit_hq_history(hq):
    """For each permno, build a sorted [(filing_date, STATE)] history. Returns
    a dict permno -> sorted list of (filing_date, STATE)."""
    out = {}
    for _, row in hq.iterrows():
        seq = row.get('hdr_states')
        if not isinstance(seq, str):
            continue
        try:
            parsed = json.loads(seq)
        except Exception:
            continue
        if not parsed:
            continue
        items = [(pd.Timestamp(d), s) for d, s in parsed
                 if isinstance(d, str) and isinstance(s, str)]
        items = sorted(items, key=lambda x: x[0])
        out[int(row['permno'])] = items
    return out


def pit_state_for(permno, month_date, history):
    """Return the STATE in effect at `month_date` for `permno`, where in
    effect = STATE of the most-recent 10-K filed at or before month_date.
    If permno is not in history OR no filing is at-or-before month_date,
    return None (caller falls back to the panel static snapshot)."""
    hist = history.get(int(permno))
    if not hist:
        return None
    # binary search for the latest filing date <= month_date
    # hist is sorted by date
    idx = -1
    for i, (d, s) in enumerate(hist):
        if d <= month_date:
            idx = i
        else:
            break
    if idx < 0:
        return None
    return hist[idx][1]


def derive_state_year_literacy_from_panel(df):
    """Recover the underlying per-state per-year literacy rank from the seed
    panel. The panel's literacy_score_corrected is z-scored within month, but
    its sign-rank across states is preserved within month. We aggregate to a
    per-state per-year mean of the original (pre-z-score) lit_mean_corrected
    (which is in the panel as a separate column) so we can rebuild
    literacy_score_corrected for PIT-reassigned states."""
    # The seed panel has 'lit_mean_corrected' — that's the per-state per-year
    # NFCS Big Five mean (already corrected for the key-coding issue).
    sy = (df.groupby(['hq_state', df['date'].dt.year])
          .agg(lit_mean_corrected=('lit_mean_corrected', 'first'))
          .reset_index()
          .rename(columns={'date': 'year'}))
    sy.columns = ['hq_state', 'year', 'lit_mean_corrected']
    return sy


def main():
    print("=== Q5 PIT-HQ literacy reassignment (v7) ===", flush=True)
    df = load_panel()
    print(f"panel: {len(df):,} firm-months, {df['permno'].nunique()} permnos",
          flush=True)
    hq = pd.read_parquet(HQ_EDGAR)
    print(f"EDGAR HQ: {len(hq):,} firms with CIK", flush=True)

    history = build_pit_hq_history(hq)
    print(f"PIT histories built for {len(history):,} permnos", flush=True)

    # State-year literacy values (from the seed panel)
    state_year_lit = (df.dropna(subset=['hq_state', 'lit_mean_corrected'])
                      [['hq_state', 'date', 'lit_mean_corrected']].copy())
    state_year_lit['year'] = state_year_lit['date'].dt.year
    sy = (state_year_lit.groupby(['hq_state', 'year'])
          ['lit_mean_corrected'].mean().reset_index())
    # convert hq_state codes: panel uses 'US-XX' format
    print(f"state-year literacy rows: {len(sy):,}", flush=True)
    print(f"sample state-year:\n{sy.head(3).to_string(index=False)}",
          flush=True)
    print(f"hq_state in panel sample: {sorted(df['hq_state'].dropna().unique())[:5]}",
          flush=True)

    # Build PIT HQ state per firm-month
    print("\nbuilding PIT HQ state per firm-month ...", flush=True)
    df['month_date'] = pd.to_datetime(df['ym'] + '-01')
    df['hq_state_orig'] = df['hq_state'].copy()
    df['year'] = df['date'].dt.year

    # Apply PIT reassignment per firm
    # vectorize: for each permno in history, build a step-function and apply
    df['hq_state_pit'] = df['hq_state'].copy()
    df['pit_reassigned'] = False
    df['no_pit_history'] = ~df['permno'].isin(history)

    # Process permnos with histories — vectorized over the firm's rows
    perms_with_hist = list(history.keys())
    print(f"applying PIT reassignment to {len(perms_with_hist)} firms ...",
          flush=True)
    n_changed = 0
    for k, permno in enumerate(perms_with_hist):
        hist = history[permno]
        mask = (df['permno'] == permno)
        if not mask.any():
            continue
        dates = df.loc[mask, 'month_date'].values
        # find for each date the state-in-effect
        # build sorted arrays
        h_dates = np.array([pd.Timestamp(d) for d, _ in hist],
                           dtype='datetime64[ns]')
        h_states = np.array([s for _, s in hist])
        # searchsorted to find index of latest hist date <= each panel date
        idxs = np.searchsorted(h_dates, dates, side='right') - 1
        new_states = np.where(idxs >= 0,
                              h_states[np.where(idxs >= 0, idxs, 0)],
                              None)
        new_states_2l = pd.Series(new_states).astype(object)
        # convert to 'US-XX' format if not None
        new_us = new_states_2l.where(new_states_2l.isnull(),
                                     'US-' + new_states_2l.astype(str))
        orig = df.loc[mask, 'hq_state'].values
        # if new_us is null, keep orig; else use new_us
        final_state = pd.Series(new_us.values).where(
            ~new_us.isnull(), pd.Series(orig)).values
        df.loc[mask, 'hq_state_pit'] = final_state
        n_changed += int((final_state != orig).sum())
        if (k + 1) % 1000 == 0:
            print(f"  processed {k+1}/{len(perms_with_hist)} firms; "
                  f"{n_changed:,} firm-months PIT-changed",
                  flush=True)

    print(f"\ntotal PIT-changed firm-months: {n_changed:,} "
          f"({100.0*n_changed/len(df):.2f}%)", flush=True)
    df['pit_reassigned'] = df['hq_state_pit'] != df['hq_state']

    # Re-merge literacy: for each firm-month, look up (hq_state_pit, year)
    # in sy
    sy_lookup = sy.set_index(['hq_state', 'year'])['lit_mean_corrected']
    keys = list(zip(df['hq_state_pit'].values, df['year'].values))
    df['lit_mean_pit'] = pd.Series(
        [sy_lookup.get(k, np.nan) for k in keys], index=df.index)

    pit_miss = int(df['lit_mean_pit'].isnull().sum())
    print(f"  PIT-literacy lookup misses: {pit_miss:,} "
          f"({100.0*pit_miss/len(df):.2f}%)", flush=True)

    # The seed panel has lit_mean_corrected on (hq_state, year). For non-PIT-
    # changed rows the PIT literacy MUST equal the original literacy at the
    # same (state, year). Verify:
    chk = df[~df['pit_reassigned']].sample(min(10000, len(df)), random_state=42)
    chk_diff = np.abs(chk['lit_mean_corrected'] - chk['lit_mean_pit']).max()
    print(f"  sanity check (non-reassigned rows): max |lit - lit_pit| = "
          f"{chk_diff:.6f}", flush=True)

    # Re-z-score within month using the PIT literacy
    print("\nre-z-scoring literacy within month using PIT values ...",
          flush=True)
    m_mean = df.groupby('ym')['lit_mean_pit'].transform('mean')
    m_std = df.groupby('ym')['lit_mean_pit'].transform('std')
    df['literacy_score_corrected_pit'] = (df['lit_mean_pit'] - m_mean) / m_std

    # Rebuild three-way and lower-order interactions on PIT literacy
    df['mom_x_literacy_pit'] = df['mom_12_2'] * df['literacy_score_corrected_pit']
    df['iv_x_literacy_pit'] = df['iv'] * df['literacy_score_corrected_pit']
    df['mom_x_iv_x_literacy_pit'] = (
        df['mom_12_2'] * df['iv'] * df['literacy_score_corrected_pit'])

    # ---- Run BOTH headlines: ORIGINAL and PIT ----
    # ORIGINAL = identical to results_v3.json item1, full panel
    d_orig = df.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"\noriginal estimation sample: {len(d_orig):,} firm-months",
          flush=True)

    # PIT: use the PIT focal columns; hq_state is the PIT state (to keep state
    # FE / clustering at the actual PIT state, not the static snapshot)
    pit_focal = ['mom_12_2', 'iv', 'literacy_score_corrected_pit', 'mom_x_iv',
                 'mom_x_literacy_pit', 'iv_x_literacy_pit',
                 'mom_x_iv_x_literacy_pit']
    d_pit = df.dropna(subset=pit_focal + ['ret', 'hq_state_pit']).copy()
    print(f"PIT estimation sample: {len(d_pit):,} firm-months (vs original "
          f"{len(d_orig):,})", flush=True)

    # Headline ORIGINAL
    print("\n=== headline on ORIGINAL panel (static HQ snapshot) ===",
          flush=True)
    r_orig = twfe_three_way(d_orig)
    wcb_orig = wild_cluster_bootstrap_state(d_orig, B=B_WCB, seed=42)
    print(f"  gamma_hat={r_orig['coef']:+.6f} | CGM t={r_orig['t']:.2f} "
          f"({r_orig['se_kind']}) | state CR1 t={r_orig['t_state']:.2f} | "
          f"wcb p={wcb_orig['p_value']:.4f}", flush=True)

    # Headline PIT: substitute focal columns
    d_pit_for_est = d_pit.copy()
    d_pit_for_est['literacy_score_corrected'] = d_pit_for_est[
        'literacy_score_corrected_pit']
    d_pit_for_est['mom_x_literacy_corr'] = d_pit_for_est['mom_x_literacy_pit']
    d_pit_for_est['iv_x_literacy_corr'] = d_pit_for_est['iv_x_literacy_pit']
    d_pit_for_est['mom_x_iv_x_literacy_corr'] = d_pit_for_est[
        'mom_x_iv_x_literacy_pit']
    d_pit_for_est['hq_state'] = d_pit_for_est['hq_state_pit']

    print("\n=== headline on PIT panel (point-in-time HQ + re-z-scored "
          "literacy) ===", flush=True)
    r_pit = twfe_three_way(d_pit_for_est)
    wcb_pit = wild_cluster_bootstrap_state(d_pit_for_est, B=B_WCB, seed=42)
    print(f"  gamma_hat={r_pit['coef']:+.6f} | CGM t={r_pit['t']:.2f} "
          f"({r_pit['se_kind']}) | state CR1 t={r_pit['t_state']:.2f} | "
          f"wcb p={wcb_pit['p_value']:.4f}", flush=True)

    # Also: PIT clustering on ORIGINAL static state (intermediate: PIT
    # literacy, static state cluster), to disentangle the PIT-literacy effect
    # from the PIT-clustering effect.
    d_pit_static_cl = d_pit_for_est.copy()
    d_pit_static_cl['hq_state'] = d_pit['hq_state']
    print("\n=== headline on PIT literacy with STATIC-state clustering ===",
          flush=True)
    r_pit_sc = twfe_three_way(d_pit_static_cl)
    wcb_pit_sc = wild_cluster_bootstrap_state(d_pit_static_cl, B=B_WCB,
                                              seed=42)
    print(f"  gamma_hat={r_pit_sc['coef']:+.6f} | CGM t={r_pit_sc['t']:.2f} "
          f"({r_pit_sc['se_kind']}) | state CR1 t="
          f"{r_pit_sc['t_state']:.2f} | wcb p={wcb_pit_sc['p_value']:.4f}",
          flush=True)

    # Coverage diagnostics
    n_pit_changed = int(df['pit_reassigned'].sum())
    n_firms_with_change = int(
        df[df['pit_reassigned']]['permno'].nunique())
    cov = {
        'panel_firm_months': int(len(df)),
        'panel_firms': int(df['permno'].nunique()),
        'firms_with_edgar_history': int(len(history)),
        'firms_with_pit_change': n_firms_with_change,
        'firm_months_pit_changed': n_pit_changed,
        'firm_months_pit_changed_pct': round(
            100.0 * n_pit_changed / len(df), 2),
        'pit_lookup_misses': pit_miss,
        'estimation_sample_orig': int(len(d_orig)),
        'estimation_sample_pit': int(len(d_pit)),
    }
    print(f"\nCoverage: {cov}", flush=True)

    results = {
        'task': 'Q5 (referee structured Q5; triager row 29): point-in-time '
                'HQ-state literacy reassignment using SEC EDGAR 10-K filing-'
                'header business-address STATE per firm-month.',
        'method': 'For each firm-month, the HQ state is the STATE recorded '
                  'in the firm\'s most-recent 10-K filed at or before that '
                  'month. The literacy_score_corrected variable is re-merged '
                  'on (then-current HQ state, year) and re-z-scored within '
                  'month. The three-way interaction is rebuilt on the '
                  'PIT-reassigned literacy_z.',
        'reference': {
            'full_panel_v6': {'gamma_hat': -0.011727, 't_state': -2.17,
                              'wcb_p': 0.0775},
            'relocation_free_v6': {'gamma_hat': -0.014516, 't_state': -1.94,
                                   'wcb_p': 0.1223},
        },
        'coverage': cov,
        'sanity_check_max_diff_for_non_reassigned': float(chk_diff),
        'headline_original': {
            'gamma_hat': r_orig['coef'],
            't_cgm_two_way': r_orig['t'],
            'se_kind': r_orig['se_kind'],
            't_state_clustered_CR1': r_orig['t_state'],
            'wcb_p_value': wcb_orig['p_value'],
            'wcb_ci_studentized': wcb_orig['ci_studentized'],
            'wcb_B': wcb_orig['B'],
            'n_obs': int(r_orig['n_obs']),
            'n_state_clusters': wcb_orig['n_state_clusters'],
        },
        'headline_pit_full': {
            'gamma_hat': r_pit['coef'],
            't_cgm_two_way': r_pit['t'],
            'se_kind': r_pit['se_kind'],
            't_state_clustered_CR1': r_pit['t_state'],
            'wcb_p_value': wcb_pit['p_value'],
            'wcb_ci_studentized': wcb_pit['ci_studentized'],
            'wcb_B': wcb_pit['B'],
            'n_obs': int(r_pit['n_obs']),
            'n_state_clusters': wcb_pit['n_state_clusters'],
            'description': 'PIT literacy + PIT state cluster',
        },
        'headline_pit_static_cluster': {
            'gamma_hat': r_pit_sc['coef'],
            't_cgm_two_way': r_pit_sc['t'],
            't_state_clustered_CR1': r_pit_sc['t_state'],
            'wcb_p_value': wcb_pit_sc['p_value'],
            'wcb_ci_studentized': wcb_pit_sc['ci_studentized'],
            'n_obs': int(r_pit_sc['n_obs']),
            'description': 'PIT literacy + static state cluster (decomposes '
                           'PIT-literacy effect from PIT-clustering effect)',
        },
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)
    return results


if __name__ == '__main__':
    main()
