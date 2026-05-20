# =====================================================================
# v7_pit_hq_centerpiece.py
# Re-fires the 13F IO sign-flip and within-retail literacy-gradient tests under
# point-in-time (PIT) SEC EDGAR HQ-state literacy reassignment, producing the PIT
# centerpiece tables.
#
# Inputs:    builds/loads _dfm_pit_hq_v7_centerpiece.parquet from the seed panel
#            (via deepen_thomson_io_panel_v6.load_panel) merged with the corrected
#            Thomson s34 IO panel; _hq_edgar_state_v6.parquet (SEC EDGAR 10-K-header
#            HQ-state history).
# Outputs:   output/stage3a/results_v7_pit_hq_centerpiece.json;
#            output/stage3a/tables/tab_13f_stacked_diff_pit.tex;
#            output/stage3a/tables/tab_within_retail_literacy_pit.tex
# Paper:     Main Table T6 tab:nested (PIT) + Internet Appendix PIT diagnostics /
#            PIT mid-IO sections
# Run order: see code/00_master.py
# =====================================================================

"""v7 PIT-HQ CENTERPIECE — operator-mandated re-fire after v7 Q5 aggregate
collapse.

Background: v7 Q5 showed the *aggregate* three-way coefficient collapses
from -0.0117 (wcb p=0.078) under static HQ literacy to -0.0003 (wcb p=0.517)
under point-in-time HQ literacy (the SEC EDGAR 10-K-header reassignment).
The operator's decision: re-run the centerpiece tests (the 13F IO sign flip
and the within-retail literacy gradient) under the same PIT-HQ literacy
correction, and report honestly whether the centerpiece survives.

Two tasks:
  Task 1 - 13F IO sign flip under PIT-HQ literacy.
           Run the stacked fully-interacted spec on low_IO vs high_IO terciles,
           under PIT-HQ-corrected literacy, for both persistent and time-varying
           IO measures.
  Task 2 - Within-retail literacy gradient under PIT-HQ literacy.
           Within the low-IO tercile, split firms by PIT state literacy
           (top half vs bottom half), run TWFE in each subsample, and report
           the stacked low_lit - high_lit difference.

All under the same PIT-HQ literacy as in v7_pit_hq_literacy.py.

Comparison to v6 static-HQ-literacy:
   persistent DIFF (low-high) = -0.0211, wcb p = 0.144
   time-varying DIFF (low-high) = -0.0181, wcb p = 0.264
   within-retail lowlit-hilit DIFF = +0.0022, wcb p ~ 0.07 (wrong-signed marginal)

Output: output/stage3a/results_v7_pit_hq_centerpiece.json (NEW sibling file).
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
                               stacked_io_difference,
                               stacked_state_group_difference, FOCAL)
from deepen_thomson_io_panel_v6 import (load_panel, load_s34_firmquarter,
                                        merge_io_to_panel, LAG_MONTHS)

np.random.seed(42)

HQ_EDGAR = os.path.join(EMP, "_hq_edgar_state_v6.parquet")
DFM_PIT_CACHE = os.path.join(EMP, "_dfm_pit_hq_v7_centerpiece.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a",
                        "results_v7_pit_hq_centerpiece.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

B_WCB = 4999


# =============================================================================
# PIT-HQ literacy reassignment (copy of v7_pit_hq_literacy.py functions)
# =============================================================================

def build_pit_hq_history(hq):
    """For each permno, build sorted [(filing_date, STATE)] history.
    Returns dict permno -> sorted list of (filing_date, STATE)."""
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


def apply_pit_hq(df, history, verbose=True):
    """Reassign hq_state to the firm's PIT HQ-state per firm-month.

    df is the firm-month panel with 'permno', 'month_date', 'hq_state'.
    history is the dict from build_pit_hq_history.
    Returns df with hq_state_pit and pit_reassigned columns set.
    """
    df = df.copy()
    df['hq_state_pit'] = df['hq_state'].copy()
    df['pit_reassigned'] = False

    perms = list(history.keys())
    n_changed = 0
    for k, permno in enumerate(perms):
        hist = history[permno]
        mask = (df['permno'] == permno)
        if not mask.any():
            continue
        dates = df.loc[mask, 'month_date'].values
        h_dates = np.array([pd.Timestamp(d) for d, _ in hist],
                           dtype='datetime64[ns]')
        h_states = np.array([s for _, s in hist])
        idxs = np.searchsorted(h_dates, dates, side='right') - 1
        new_states = np.where(idxs >= 0,
                              h_states[np.where(idxs >= 0, idxs, 0)],
                              None)
        new_states_2l = pd.Series(new_states).astype(object)
        new_us = new_states_2l.where(new_states_2l.isnull(),
                                     'US-' + new_states_2l.astype(str))
        orig = df.loc[mask, 'hq_state'].values
        final_state = pd.Series(new_us.values).where(
            ~new_us.isnull(), pd.Series(orig)).values
        df.loc[mask, 'hq_state_pit'] = final_state
        n_changed += int((final_state != orig).sum())
        if verbose and (k + 1) % 1000 == 0:
            print(f"  PIT applied to {k+1}/{len(perms)} firms; "
                  f"{n_changed:,} firm-months changed", flush=True)

    df['pit_reassigned'] = df['hq_state_pit'] != df['hq_state']
    if verbose:
        print(f"  Total PIT-changed firm-months: {n_changed:,} "
              f"({100.0 * n_changed / len(df):.2f}%)", flush=True)
    return df, n_changed


def rezscore_literacy_pit(df):
    """Re-z-score literacy within month using the PIT HQ state.

    df must have hq_state_pit, year, and lit_mean_corrected. Builds the per-
    state-per-year literacy lookup from the (original) panel and applies it
    via the PIT state. Then re-z-scores within month.

    Returns df with:
      literacy_score_corrected_pit (the new lit-z)
      mom_x_literacy_pit, iv_x_literacy_pit, mom_x_iv_x_literacy_pit
    """
    df = df.copy()
    state_year_lit = (df.dropna(subset=['hq_state', 'lit_mean_corrected'])
                      [['hq_state', 'date', 'lit_mean_corrected']].copy())
    state_year_lit['year'] = state_year_lit['date'].dt.year
    sy = (state_year_lit.groupby(['hq_state', 'year'])
          ['lit_mean_corrected'].mean().reset_index())
    sy_lookup = sy.set_index(['hq_state', 'year'])['lit_mean_corrected']

    keys = list(zip(df['hq_state_pit'].values, df['year'].values))
    df['lit_mean_pit'] = pd.Series(
        [sy_lookup.get(k, np.nan) for k in keys], index=df.index)
    pit_miss = int(df['lit_mean_pit'].isnull().sum())
    print(f"  PIT-literacy lookup misses: {pit_miss:,} "
          f"({100.0 * pit_miss / len(df):.2f}%)", flush=True)

    # Sanity check on non-reassigned rows
    chk = df[~df['pit_reassigned']].sample(min(10000, len(df)), random_state=42)
    chk_diff = float(np.abs(chk['lit_mean_corrected']
                            - chk['lit_mean_pit']).max())
    print(f"  Non-reassigned sanity check: max |lit - lit_pit| = "
          f"{chk_diff:.6e}", flush=True)

    # Re-z-score within month using PIT values
    m_mean = df.groupby('ym')['lit_mean_pit'].transform('mean')
    m_std = df.groupby('ym')['lit_mean_pit'].transform('std')
    df['literacy_score_corrected_pit'] = (df['lit_mean_pit'] - m_mean) / m_std

    # Rebuild interactions
    df['mom_x_literacy_pit'] = (df['mom_12_2']
                                * df['literacy_score_corrected_pit'])
    df['iv_x_literacy_pit'] = df['iv'] * df['literacy_score_corrected_pit']
    df['mom_x_iv_x_literacy_pit'] = (
        df['mom_12_2'] * df['iv'] * df['literacy_score_corrected_pit'])

    return df, pit_miss, chk_diff


def install_pit_focal(d):
    """Overwrite the FOCAL columns to use PIT-reassigned literacy & PIT state.

    deepen_estimators uses FOCAL = ['mom_12_2', 'iv',
    'literacy_score_corrected', 'mom_x_iv', 'mom_x_literacy_corr',
    'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']
    and 'hq_state' / 'ym' / 'ret'.

    We substitute the PIT columns into those field names, and use hq_state_pit
    as hq_state (so FE/clustering live at the PIT state).
    """
    d = d.copy()
    d['literacy_score_corrected'] = d['literacy_score_corrected_pit']
    d['mom_x_literacy_corr'] = d['mom_x_literacy_pit']
    d['iv_x_literacy_corr'] = d['iv_x_literacy_pit']
    d['mom_x_iv_x_literacy_corr'] = d['mom_x_iv_x_literacy_pit']
    d['hq_state'] = d['hq_state_pit']
    return d


# =============================================================================
# IO terciles
# =============================================================================

def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


# =============================================================================
# main
# =============================================================================

def build_pit_dfm():
    """Build the PIT-HQ literacy-corrected firm-month panel, merged with the
    corrected Thomson s34 IO panel. Cached to DFM_PIT_CACHE."""
    if os.path.exists(DFM_PIT_CACHE):
        print(f"=== loading cached PIT-corrected merged panel "
              f"{DFM_PIT_CACHE} ===", flush=True)
        return pd.read_parquet(DFM_PIT_CACHE)

    print("=== building PIT-corrected merged panel from scratch ===",
          flush=True)
    df = load_panel()
    print(f"  panel: {len(df):,} firm-months, "
          f"{df['permno'].nunique()} permnos", flush=True)
    df['month_date'] = pd.to_datetime(df['ym'] + '-01')
    df['year'] = df['date'].dt.year

    # PIT HQ reassignment
    print("  loading EDGAR HQ ...", flush=True)
    hq = pd.read_parquet(HQ_EDGAR)
    print(f"  EDGAR HQ rows: {len(hq):,}", flush=True)
    history = build_pit_hq_history(hq)
    print(f"  PIT histories built for {len(history):,} permnos", flush=True)
    df, n_changed = apply_pit_hq(df, history, verbose=True)

    # PIT literacy re-z-score
    print("  re-z-scoring literacy using PIT values within month ...",
          flush=True)
    df, pit_miss, chk_diff = rezscore_literacy_pit(df)

    # Merge corrected Thomson s34 IO panel
    print("  loading corrected Thomson s34 IO panel ...", flush=True)
    fq = load_s34_firmquarter()
    print(f"  firm-quarter: {len(fq):,} rows, "
          f"{fq['permno'].nunique()} permnos, "
          f"{fq['quarter'].nunique()} quarters", flush=True)
    dfm = merge_io_to_panel(df, fq, verbose=True)

    # cache
    dfm.to_parquet(DFM_PIT_CACHE)
    print(f"  cached PIT-merged panel to {DFM_PIT_CACHE}", flush=True)
    return dfm


def task1_io_sign_flip(dfm, results):
    """Task 1: 13F IO sign flip test under PIT-HQ literacy.

    Persistent and time-varying IO measures.
    """
    print("\n=== TASK 1: 13F IO sign flip under PIT-HQ literacy ===",
          flush=True)

    # Estimation sample: drop rows missing PIT focal or PIT state
    pit_focal_cols = ['mom_12_2', 'iv', 'literacy_score_corrected_pit',
                      'mom_x_iv', 'mom_x_literacy_pit', 'iv_x_literacy_pit',
                      'mom_x_iv_x_literacy_pit']
    d_all = dfm.dropna(subset=pit_focal_cols + ['ret', 'hq_state_pit']).copy()
    print(f"  PIT estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos", flush=True)

    out = {}

    for measure, io_col, label in [
        ('persistent', 'io_share_persist',
         'persistent (between-firm) IO measure (time-mean io_share)'),
        ('time_varying', 'io_share',
         'genuinely time-varying IO measure (most-recent-quarter, lagged '
         f'{LAG_MONTHS} months)'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io, perm_io, _ = add_io_terciles(d_io, io_col)
        terc_means = d_io.groupby('io_grp')[io_col].mean().to_dict()
        n_p = d_io['permno'].nunique()
        n_o = len(d_io)
        print(f"  {measure} IO: {n_p} permnos, {n_o:,} firm-months",
              flush=True)
        print(f"    tercile means: "
              f"{ {k: round(v, 3) for k, v in terc_means.items()} }",
              flush=True)

        # Per-tercile three-way (with PIT focal)
        pertile = {}
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            sub_pit = install_pit_focal(sub)
            r = twfe_three_way(sub_pit)
            pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                            't_state', 'n_obs', 'se_kind']}
            print(f"    {g}: gamma_pit={r['coef']:+.6f} "
                  f"cgm_t={r['t']:.2f} state_t={r['t_state']:.2f} "
                  f"n={r['n_obs']:,}", flush=True)

        # Stacked difference (low - high)
        d_stack = d_io[d_io['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
        d_stack['io_grp_bin'] = np.where(d_stack['io_grp'] == 'IO1_low',
                                         'low', 'high')
        d_stack_pit = install_pit_focal(d_stack)
        print(f"  running stacked difference (B={B_WCB}) ...", flush=True)
        sd = stacked_io_difference(d_stack_pit, B=B_WCB, seed=42)
        print(f"    DIFF (low-high) = {sd['difference_coef']:+.6f} | "
              f"state-cl SE {sd['se_state_clustered']:.6f} "
              f"t={sd['t_state_clustered']:.2f} | "
              f"CGM t={sd['t_cgm_two_way']} | "
              f"wcb p={sd['wcb_p_value']:.4f}", flush=True)

        out[measure] = {
            "io_measure": label,
            "tercile_means": {k: float(v) for k, v in terc_means.items()},
            "per_tercile_threeway_pit": pertile,
            "stacked_difference_pit": sd,
            "n_permnos": int(n_p),
            "n_firm_months": int(n_o),
        }

    results['task1_13f_io_sign_flip_pit'] = out
    return d_all, out


def task2_within_retail_gradient(dfm, results):
    """Task 2: within-retail literacy gradient under PIT-HQ literacy.

    Within the low-IO (high-retail) tercile, split firms by PIT state literacy
    (top half vs bottom half), run TWFE in each, and compute the stacked
    low_lit - high_lit difference.

    Both persistent and time-varying IO terciles are tested.
    """
    print("\n=== TASK 2: within-retail literacy gradient under PIT-HQ ===",
          flush=True)
    pit_focal_cols = ['mom_12_2', 'iv', 'literacy_score_corrected_pit',
                      'mom_x_iv', 'mom_x_literacy_pit', 'iv_x_literacy_pit',
                      'mom_x_iv_x_literacy_pit']
    d_all = dfm.dropna(subset=pit_focal_cols + ['ret', 'hq_state_pit']).copy()

    out = {}

    for measure, io_col, label in [
        ('persistent', 'io_share_persist', 'persistent IO measure'),
        ('time_varying', 'io_share', 'time-varying IO measure'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io, _, _ = add_io_terciles(d_io, io_col)
        d_lowio = d_io[d_io['io_grp'] == 'IO1_low'].copy()
        d_lowio_pit = install_pit_focal(d_lowio)
        print(f"  low-IO (high-retail): {len(d_lowio_pit):,} firm-months, "
              f"{d_lowio_pit['permno'].nunique()} permnos", flush=True)

        # Three-way within low-IO under PIT
        r_low = twfe_three_way(d_lowio_pit)
        print(f"  three-way within low-IO (PIT): coef={r_low['coef']:+.6f} "
              f"cgm_t={r_low['t']:.2f} state_t={r_low['t_state']:.2f}",
              flush=True)
        wcb_low = wild_cluster_bootstrap_state(d_lowio_pit, B=B_WCB, seed=42)
        print(f"    wcb p={wcb_low['p_value']:.4f}", flush=True)

        # Split by PIT state literacy: PIT state -> mean literacy_z_pit within
        # the low-IO subsample.
        state_lit = (d_lowio_pit.groupby('hq_state')
                     ['literacy_score_corrected'].mean())
        lit_median = state_lit.median()
        hi_lit_states = set(state_lit[state_lit >= lit_median].index)
        lo_lit_states = set(state_lit[state_lit < lit_median].index)
        d_hilit = d_lowio_pit[d_lowio_pit['hq_state']
                              .isin(hi_lit_states)].copy()
        d_lolit = d_lowio_pit[d_lowio_pit['hq_state']
                              .isin(lo_lit_states)].copy()
        r_hl = twfe_three_way(d_hilit)
        r_ll = twfe_three_way(d_lolit)
        print(f"    low-IO + high-lit-PIT: {r_hl['coef']:+.6f} "
              f"(state_t={r_hl['t_state']:.2f}, n={r_hl['n_obs']:,})",
              flush=True)
        print(f"    low-IO + low-lit-PIT:  {r_ll['coef']:+.6f} "
              f"(state_t={r_ll['t_state']:.2f}, n={r_ll['n_obs']:,})",
              flush=True)

        # Stacked difference (lowlit - hilit) within low-IO
        d_stack = d_lowio_pit.copy()
        d_stack['lit_grp'] = np.where(
            d_stack['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')
        print("  stacked low-lit minus high-lit difference within low-IO "
              "(B=4999) ...", flush=True)
        sd_lit = stacked_state_group_difference(d_stack, 'lit_grp', 'low_lit',
                                                B=B_WCB, seed=42)
        print(f"    DIFF(lowlit - hilit | low-IO) = "
              f"{sd_lit['difference_coef']:+.6f} "
              f"state-t={sd_lit['t_state_clustered']:.2f} "
              f"wcb-p={sd_lit['wcb_p_value']:.4f}", flush=True)

        out[measure] = {
            "io_measure": label,
            "threeway_within_lowIO_pit": {
                **{k: r_low[k] for k in ['coef', 'se', 't', 'se_state',
                                         't_state', 'n_obs', 'se_kind']},
                "wild_cluster_bootstrap": wcb_low,
            },
            "lowIO_by_PIT_state_literacy": {
                "high_literacy_states": {
                    **{k: r_hl[k] for k in ['coef', 'se', 't', 'se_state',
                                            't_state', 'n_obs', 'se_kind']},
                    "n_states": len(hi_lit_states)},
                "low_literacy_states": {
                    **{k: r_ll[k] for k in ['coef', 'se', 't', 'se_state',
                                            't_state', 'n_obs', 'se_kind']},
                    "n_states": len(lo_lit_states)},
                "stacked_lowlit_minus_highlit": sd_lit,
            },
        }

    results['task2_within_retail_gradient_pit'] = out
    return out


def write_tables(results):
    """LaTeX tables for the PIT-HQ centerpiece tests."""
    # Task 1: IO sign flip table
    fname = os.path.join(TABDIR, "tab_13f_stacked_diff_pit.tex")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma_{pit}$ & CGM SE & "
                "state-cl. SE & $N$ \\\\\n")
        for mlabel, mdisp in [
            ('persistent', 'Panel A: persistent IO + PIT-HQ literacy'),
            ('time_varying', 'Panel B: time-varying IO + PIT-HQ literacy')
        ]:
            b = results['task1_13f_io_sign_flip_pit'][mlabel]
            pt = b['per_tercile_threeway_pit']
            sd = b['stacked_difference_pit']
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            for g, disp in [('IO1_low', 'Low IO (high retail)'),
                            ('IO2_mid', 'Mid IO'),
                            ('IO3_high', 'High IO (institutional)')]:
                r = pt[g]
                f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                        f"${r['se_state']:.4f}$ & ${r['n_obs']:,}$ \\\\\n")
            f.write(f"Difference (low $-$ high), stacked & "
                    f"${sd['difference_coef']:.4f}$ & "
                    f"${sd['se_cgm_two_way'] if sd['se_cgm_two_way'] is not None else 'n.p.s.d.'}$ & "
                    f"${sd['se_state_clustered']:.4f}$ & "
                    f"${sd['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad state-clustered $t$ & "
                    f"\\multicolumn{{4}}{{l}}{{"
                    f"${sd['t_state_clustered']:.2f}$}} \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sd['wcb_p_value']:.3f}$ "
                    f"($B={sd['B']}$, {sd['n_state_clusters']} state "
                    f"clusters)}} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")
    print(f"  wrote {fname}", flush=True)

    # Task 2: within-retail literacy-gradient table
    fname = os.path.join(TABDIR, "tab_within_retail_literacy_pit.tex")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Subsample & three-way $\\hat\\gamma_{pit}$ & CGM $t$ & "
                "state-cl. $t$ & $N$ \\\\\n")
        for mlabel, mdisp in [
            ('persistent',
             'Panel A: low-IO tercile (persistent IO) + PIT-HQ literacy'),
            ('time_varying',
             'Panel B: low-IO tercile (time-varying IO) + PIT-HQ literacy')
        ]:
            dd = results['task2_within_retail_gradient_pit'][mlabel]
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            tw = dd['threeway_within_lowIO_pit']
            f.write(f"Within low-IO (high-retail) & ${tw['coef']:.4f}$ & "
                    f"${tw['t']:.2f}$ & ${tw['t_state']:.2f}$ & "
                    f"${tw['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{"
                    f"${tw['wild_cluster_bootstrap']['p_value']:.3f}$}}"
                    f" \\\\\n")
            hl = dd['lowIO_by_PIT_state_literacy']['high_literacy_states']
            ll = dd['lowIO_by_PIT_state_literacy']['low_literacy_states']
            sdl = dd['lowIO_by_PIT_state_literacy'][
                'stacked_lowlit_minus_highlit']
            f.write(f"\\quad low-IO $\\times$ PIT high-lit states & "
                    f"${hl['coef']:.4f}$ & ${hl['t']:.2f}$ & "
                    f"${hl['t_state']:.2f}$ & ${hl['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad low-IO $\\times$ PIT low-lit states & "
                    f"${ll['coef']:.4f}$ & ${ll['t']:.2f}$ & "
                    f"${ll['t_state']:.2f}$ & ${ll['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad difference (PIT lowlit $-$ hilit), stacked & "
                    f"${sdl['difference_coef']:.4f}$ & --- & "
                    f"${sdl['t_state_clustered']:.2f}$ & "
                    f"${sdl['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sdl['wcb_p_value']:.3f}$}}"
                    f" \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")
    print(f"  wrote {fname}", flush=True)


def main():
    dfm = build_pit_dfm()

    results = {
        "task": "v7 PIT-HQ centerpiece test: re-fire the 13F IO sign flip "
                "(Task 1) and the within-retail literacy gradient (Task 2) "
                "under the same SEC EDGAR PIT-HQ literacy reassignment that "
                "collapsed the v7 Q5 aggregate three-way.",
        "reference_v6_static_HQ": {
            "io_persistent_DIFF": -0.0211,
            "io_persistent_wcb_p": 0.144,
            "io_time_varying_DIFF": -0.0181,
            "io_time_varying_wcb_p": 0.264,
            "within_retail_lowlit_hilit_DIFF": 0.0022,
            "within_retail_lowlit_hilit_wcb_p": 0.07,
        },
        "reference_v7_pit_aggregate": {
            "headline_original_gamma": -0.0117,
            "headline_original_wcb_p": 0.078,
            "headline_pit_gamma": -0.0003,
            "headline_pit_wcb_p": 0.517,
        },
    }

    d_all, _ = task1_io_sign_flip(dfm, results)
    _ = task2_within_retail_gradient(dfm, results)

    results['meta'] = {
        "panel_firm_months": int(len(dfm)),
        "panel_firms": int(dfm['permno'].nunique()),
        "pit_estimation_sample": int(len(d_all)),
        "B_wcb": B_WCB,
        "lag_months": LAG_MONTHS,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)

    write_tables(results)
    return results


if __name__ == '__main__':
    main()
