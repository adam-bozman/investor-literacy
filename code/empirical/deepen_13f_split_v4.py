# =====================================================================
# deepen_13f_split_v4.py
# Re-runs the 13F split (items b/c/d) on the EDGAR IO panel with proper stacked clustering: stacked low-IO-minus-high-IO difference, size-tercile placebo on balanced panels, and the within-retail literacy-gradient test.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet (via deepen_io_panel); EDGAR 13F quarters on disk (_edgar_13f_*_cusip9.parquet or *_ticker.parquet); deepen_estimators.py
# Outputs:   output/stage3a/results_v4.json; output/stage3a/tables/tab_13f_stacked_diff_v4.tex, tab_within_retail_literacy_v4.tex
# Paper:     SUPERSEDED by deepen_13f_split_v6.py — kept for provenance (EDGAR-proxy round-1; feeds the IA within-retail IV-decile / EDGAR-proxy cross-check)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 — items (b), (c), (d). Re-runnable.

Runs on whatever EDGAR 13F quarters are on disk (auto-selected by deepen_io_panel:
v4 cusip9 quarters if present, else round-0 ticker quarters). Re-run after the v4
download adds quarters to refresh every result on the extended panel.

Item (b): STACKED 13F difference test with state-clustered + two-way CGM + wild-
          cluster bootstrap inference on the low-IO-minus-high-IO three-way
          difference. Replaces the round-0 t = -3.56 which assumed tercile
          independence. Run on (1) the persistent IO measure (round-0-comparable),
          and (2) the time-varying IO measure if >=4 quarters available.
Item (c): size-tercile placebo on the BALANCED panel — restricted to the same
          firms covered by the 13F IO measure (and, separately, to permnos present
          all 180 months), so the size placebo and the IO split share a sample.
Item (d): within-retail (low-IO tercile) literacy-gradient test — does the
          three-way mom x IV x literacy_z survive WITHIN the high-retail subsample?
          If yes, the literacy reading is pinned; if flat, the channel is generic
          retail/institutional.

Outputs: output/stage3a/results_v4.json (machine-readable), the markdown writeup is
hand-assembled into empirical_analysis_v4.md, LaTeX tables to
output/stage3a/tables/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepen_io_panel import (load_panel, build_io_firmquarter,
                             build_io_firmquarter_ticker_only,
                             merge_io_to_panel)
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               stacked_io_difference,
                               stacked_state_group_difference, FOCAL)

np.random.seed(42)

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "results_v4.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

# the three-way is mom_x_iv_x_literacy_corr (already z-scored within month in the
# panel build per data_inventory). literacy_z for item d is literacy_score_corrected.
LIT_Z = 'literacy_score_corrected'


def add_io_terciles(d, io_col):
    """Assign IO terciles on the firm-level mean of io_col, return d with io_grp."""
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


def main():
    results = {"meta": {}, "item_b": {}, "item_c": {}, "item_d": {}}

    print("=== load panel + build IO ===")
    df = load_panel()
    d_all = df.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos")

    fq = build_io_firmquarter(df)
    quarters = sorted(fq['quarter'].unique())
    path = fq['path'].iloc[0]
    diag = fq.attrs['diag']
    print(f"IO firm-quarter table: {len(fq):,} rows, {len(quarters)} quarters "
          f"{quarters}, path={path}")
    print(diag.to_string(index=False))

    dfm = merge_io_to_panel(df, fq, verbose=True)
    # restrict to estimation sample
    dfm = dfm[dfm.index.isin(d_all.index)] if False else dfm.merge(
        d_all[['permno', 'date']], on=['permno', 'date'], how='inner')
    # re-drop focal NAs (merge is inner on estimation rows already)
    dfm = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    results['meta'] = {
        "io_quarters": quarters,
        "n_io_quarters": len(quarters),
        "io_path": path,
        "io_temporal_span": f"{quarters[0]}..{quarters[-1]}" if quarters else None,
        "per_quarter_diag": diag.to_dict(orient='records'),
        "estimation_sample_firm_months": int(len(d_all)),
        "estimation_sample_permnos": int(d_all['permno'].nunique()),
    }

    # ===================================================================
    # ITEM (b): stacked IO-difference test, proper clustering
    # ===================================================================
    print("\n=== ITEM (b): stacked IO-difference, proper clustering ===")
    # --- (b1) persistent IO measure (round-0-comparable) ---
    dperm = dfm[dfm['io_share_persist'].notna()].copy()
    dperm, perm_io_p, _ = add_io_terciles(dperm, 'io_share_persist')
    terc_means_p = dperm.groupby('io_grp')['io_share_persist'].mean().to_dict()
    print(f"  persistent IO: {dperm['permno'].nunique()} permnos, "
          f"{len(dperm):,} firm-months; tercile means {terc_means_p}")

    # per-tercile three-way (for context / reproduce round-0)
    pertile = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = dperm[dperm['io_grp'] == g]
        r = twfe_three_way(sub)
        pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']}
        print(f"    {g}: coef={r['coef']:.6f} cgm_t={r['t']:.2f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}")

    # stacked difference: low vs high
    dstack_p = dperm[dperm['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    dstack_p['io_grp_bin'] = np.where(dstack_p['io_grp'] == 'IO1_low',
                                      'low', 'high')
    print("  running stacked difference (persistent IO, B=4999)...")
    sd_p = stacked_io_difference(dstack_p, B=4999, seed=42)
    print(f"    DIFF (low-high) = {sd_p['difference_coef']:+.6f} | "
          f"state-clustered SE {sd_p['se_state_clustered']:.6f} "
          f"t={sd_p['t_state_clustered']:.2f} | "
          f"CGM t={sd_p['t_cgm_two_way']} | "
          f"wild-cluster bootstrap p={sd_p['wcb_p_value']:.4f}")

    results['item_b']['persistent'] = {
        "io_measure": "time-mean io_share across available 13F quarters",
        "n_quarters": len(quarters),
        "tercile_means": {k: float(v) for k, v in terc_means_p.items()},
        "per_tercile_threeway": pertile,
        "stacked_difference": sd_p,
        "n_permnos": int(dperm['permno'].nunique()),
        "n_firm_months": int(len(dperm)),
    }

    # --- (b2) time-varying IO measure, only meaningful with >=4 quarters ---
    if len(quarters) >= 4:
        dtv = dfm[dfm['io_share'].notna()].copy()
        dtv, _, _ = add_io_terciles(dtv, 'io_share')
        terc_means_tv = dtv.groupby('io_grp')['io_share'].mean().to_dict()
        dstack_tv = dtv[dtv['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
        dstack_tv['io_grp_bin'] = np.where(dstack_tv['io_grp'] == 'IO1_low',
                                           'low', 'high')
        print("  running stacked difference (time-varying IO, B=4999)...")
        sd_tv = stacked_io_difference(dstack_tv, B=4999, seed=42)
        print(f"    [time-varying] DIFF = {sd_tv['difference_coef']:+.6f} | "
              f"state-clustered t={sd_tv['t_state_clustered']:.2f} | "
              f"wcb p={sd_tv['wcb_p_value']:.4f}")
        results['item_b']['time_varying'] = {
            "io_measure": "most-recent-quarter io_share, no look-ahead "
                          "(merge_asof backward)",
            "n_quarters": len(quarters),
            "tercile_means": {k: float(v) for k, v in terc_means_tv.items()},
            "stacked_difference": sd_tv,
            "n_permnos": int(dtv['permno'].nunique()),
            "n_firm_months": int(len(dtv)),
        }
    else:
        results['item_b']['time_varying'] = {
            "status": f"SKIPPED — only {len(quarters)} quarters; a time-varying "
                      f"split needs >=4 quarters to differ meaningfully from the "
                      f"persistent measure. Persistent measure used."
        }

    # --- (b3) round-0 ticker-measure comparison: re-run the round-0
    #     ticker-matched IO measure through the SAME stacked proper-clustering
    #     spec. This isolates whether the round-0 difference t = -3.56 dies from
    #     (a) proper stacked clustering alone, or (b) the source-level CUSIP9
    #     dedup. If the ticker measure ALSO survives proper clustering, the
    #     source-dedup is what kills it; if the ticker measure dies under proper
    #     clustering too, the round-0 t = -3.56 was an inference artifact.
    print("  --- (b3) round-0 ticker-measure under proper stacked clustering ---")
    try:
        fq_tk = build_io_firmquarter_ticker_only(df, verbose=True)
        tk_quarters = sorted(fq_tk['quarter'].unique())
        dfm_tk = merge_io_to_panel(df, fq_tk, verbose=False).merge(
            d_all[['permno', 'date']], on=['permno', 'date'], how='inner')
        dfm_tk = dfm_tk.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
        dperm_tk = dfm_tk[dfm_tk['io_share_persist'].notna()].copy()
        dperm_tk, _, _ = add_io_terciles(dperm_tk, 'io_share_persist')
        tk_terc_means = dperm_tk.groupby(
            'io_grp')['io_share_persist'].mean().to_dict()
        pertile_tk = {}
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            r = twfe_three_way(dperm_tk[dperm_tk['io_grp'] == g])
            pertile_tk[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                               't_state', 'n_obs', 'se_kind']}
            print(f"    [ticker] {g}: coef={r['coef']:.6f} "
                  f"cgm_t={r['t']:.2f} state_t={r['t_state']:.2f}")
        dstack_tk = dperm_tk[
            dperm_tk['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
        dstack_tk['io_grp_bin'] = np.where(
            dstack_tk['io_grp'] == 'IO1_low', 'low', 'high')
        print("  running stacked difference (round-0 ticker measure, B=4999)...")
        sd_tk = stacked_io_difference(dstack_tk, B=4999, seed=42)
        print(f"    [ticker] DIFF (low-high) = {sd_tk['difference_coef']:+.6f} "
              f"| state-clustered t={sd_tk['t_state_clustered']:.2f} "
              f"| CGM t={sd_tk['t_cgm_two_way']} "
              f"| wcb p={sd_tk['wcb_p_value']:.4f}")
        results['item_b']['round0_ticker_measure'] = {
            "io_measure": "round-0 ticker-matched IO measure (NO source-level "
                          "CUSIP9 dedup), re-run through the v4 stacked "
                          "proper-clustering spec",
            "quarters": tk_quarters,
            "n_quarters": len(tk_quarters),
            "tercile_means": {k: float(v) for k, v in tk_terc_means.items()},
            "per_tercile_threeway": pertile_tk,
            "stacked_difference": sd_tk,
            "n_permnos": int(dperm_tk['permno'].nunique()),
            "n_firm_months": int(len(dperm_tk)),
        }
    except FileNotFoundError as e:
        print(f"    round-0 ticker files not found: {e}")
        results['item_b']['round0_ticker_measure'] = {"status": str(e)}

    # ===================================================================
    # ITEM (c): size-tercile placebo on the BALANCED panel
    # ===================================================================
    print("\n=== ITEM (c): size placebo on balanced panel ===")
    # balanced panel definition 1: permnos covered by the IO measure (same
    # firms as the IO split) — the directive's primary ask.
    io_permnos = set(dperm['permno'].unique())
    d_iocov = d_all[d_all['permno'].isin(io_permnos)].copy()
    # balanced panel definition 2: permnos present all 180 months
    counts = d_all.groupby('permno').size()
    bal180 = set(counts[counts == counts.max()].index)
    d_bal180 = d_all[d_all['permno'].isin(bal180)].copy()

    def size_placebo(dd, label):
        perm_me = dd.groupby('permno')['me'].mean()
        terc = pd.qcut(perm_me, 3, labels=['T1_small', 'T2_mid', 'T3_large'])
        dd = dd.copy()
        dd['size_grp'] = dd['permno'].map(terc).astype('object')
        out = {}
        for g in ['T1_small', 'T2_mid', 'T3_large']:
            sub = dd[dd['size_grp'] == g]
            r = twfe_three_way(sub)
            out[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']}
        dstack = dd[dd['size_grp'].isin(['T1_small', 'T3_large'])].copy()
        dstack['io_grp_bin'] = np.where(dstack['size_grp'] == 'T1_small',
                                        'low', 'high')
        sd = stacked_io_difference(dstack, B=4999, seed=42)
        out['_stacked_difference'] = sd
        out['_n_permnos'] = int(dd['permno'].nunique())
        out['_n_firm_months'] = int(len(dd))
        print(f"  [{label}] small={out['T1_small']['coef']:.6f} "
              f"large={out['T3_large']['coef']:.6f} "
              f"DIFF(small-large)={sd['difference_coef']:+.6f} "
              f"state-t={sd['t_state_clustered']:.2f} "
              f"wcb-p={sd['wcb_p_value']:.4f}")
        return out

    print("  balanced panel A: IO-covered firms (same sample as the IO split)")
    sp_iocov = size_placebo(d_iocov, "IO-covered")
    print("  balanced panel B: permnos present all 180 months")
    sp_bal180 = size_placebo(d_bal180, "balanced-180")
    print("  full panel (reference)")
    sp_full = size_placebo(d_all, "full-panel")

    results['item_c'] = {
        "balanced_io_covered": sp_iocov,
        "balanced_180month": sp_bal180,
        "full_panel_reference": sp_full,
        "note": "Size placebo: a generic firm-size proxy should NOT reproduce "
                "the 13F IO sign flip. balanced_io_covered restricts to the "
                "exact firms in the IO split (the directive's primary ask); "
                "balanced_180month restricts to permnos present all 180 months "
                "(survivorship-balanced).",
    }

    # ===================================================================
    # ITEM (d): within-retail (low-IO) literacy-gradient test
    # ===================================================================
    print("\n=== ITEM (d): within-retail literacy-gradient test ===")
    # within the low-IO (high-retail) tercile, run the three-way TWFE; then
    # split that subsample by state literacy (high vs low literacy) and run
    # the three-way in each. mechanism predicts the negative three-way is
    # concentrated in low-literacy + high-retail.
    d_lowio = dperm[dperm['io_grp'] == 'IO1_low'].copy()
    d_highio = dperm[dperm['io_grp'] == 'IO3_high'].copy()
    print(f"  low-IO (high-retail) subsample: {len(d_lowio):,} firm-months, "
          f"{d_lowio['permno'].nunique()} permnos")

    # (d1) three-way within low-IO, with wild-cluster bootstrap
    r_lowio = twfe_three_way(d_lowio)
    print(f"  three-way within low-IO: coef={r_lowio['coef']:.6f} "
          f"cgm_t={r_lowio['t']:.2f} state_t={r_lowio['t_state']:.2f}")
    print("  wild-cluster bootstrap within low-IO (B=4999)...")
    wcb_lowio = wild_cluster_bootstrap_state(d_lowio, B=4999, seed=42)
    print(f"    wcb p={wcb_lowio['p_value']:.4f} "
          f"CI={wcb_lowio['ci_studentized']}")

    # (d2) within low-IO, split by state literacy. literacy_z is z-scored
    # within month; take each state's mean literacy and split states at median.
    state_lit = d_lowio.groupby('hq_state')[LIT_Z].mean()
    lit_median = state_lit.median()
    hi_lit_states = set(state_lit[state_lit >= lit_median].index)
    lo_lit_states = set(state_lit[state_lit < lit_median].index)
    d_lowio_hilit = d_lowio[d_lowio['hq_state'].isin(hi_lit_states)].copy()
    d_lowio_lolit = d_lowio[d_lowio['hq_state'].isin(lo_lit_states)].copy()
    print(f"  low-IO x high-literacy: {len(d_lowio_hilit):,} fm, "
          f"{len(hi_lit_states)} states")
    print(f"  low-IO x low-literacy:  {len(d_lowio_lolit):,} fm, "
          f"{len(lo_lit_states)} states")
    r_hl = twfe_three_way(d_lowio_hilit)
    r_ll = twfe_three_way(d_lowio_lolit)
    print(f"    low-IO+high-lit three-way: {r_hl['coef']:.6f} "
          f"(cgm_t={r_hl['t']:.2f}, state_t={r_hl['t_state']:.2f})")
    print(f"    low-IO+low-lit  three-way: {r_ll['coef']:.6f} "
          f"(cgm_t={r_ll['t']:.2f}, state_t={r_ll['t_state']:.2f})")

    # (d3) cleanest: WITHIN the low-IO subsample, the three-way coefficient IS
    # mom x IV x literacy_z. so r_lowio above already IS the literacy-gradient
    # test within high-retail. the d2 split (by state literacy level) is the
    # confirmatory cut: the gradient should be steeper / the level more negative
    # in low-literacy states. Also run the stacked low-lit-minus-high-lit diff.
    dstack_lit = d_lowio.copy()
    dstack_lit['lit_grp'] = np.where(
        dstack_lit['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')
    # the group is STATE-defined, so use stacked_state_group_difference (a group
    # main effect / group x lower-order would be collinear with the state FE).
    # difference = low_lit group three-way minus high_lit group three-way;
    # mechanism predicts < 0 (more negative three-way in low-literacy states).
    print("  stacked low-lit-minus-high-lit difference within low-IO (B=4999)...")
    sd_lit = stacked_state_group_difference(dstack_lit, 'lit_grp', 'low_lit',
                                            B=4999, seed=42)
    print(f"    DIFF(lowlit-hilit | low-IO) = {sd_lit['difference_coef']:+.6f} "
          f"state-t={sd_lit['t_state_clustered']:.2f} "
          f"wcb-p={sd_lit['wcb_p_value']:.4f}")

    # for comparison: same literacy split WITHIN the high-IO tercile (placebo —
    # mechanism predicts NO literacy gradient where institutions set prices)
    state_lit_hi = d_highio.groupby('hq_state')[LIT_Z].mean()
    lit_median_hi = state_lit_hi.median()
    hi_lit_states_h = set(state_lit_hi[state_lit_hi >= lit_median_hi].index)
    lo_lit_states_h = set(state_lit_hi[state_lit_hi < lit_median_hi].index)
    dstack_lit_hi = d_highio.copy()
    dstack_lit_hi['lit_grp'] = np.where(
        dstack_lit_hi['hq_state'].isin(lo_lit_states_h), 'low_lit', 'high_lit')
    print("  stacked low-lit-minus-high-lit difference within HIGH-IO (B=4999)...")
    sd_lit_hi = stacked_state_group_difference(dstack_lit_hi, 'lit_grp',
                                               'low_lit', B=4999, seed=42)
    print(f"    DIFF(lowlit-hilit | high-IO) = "
          f"{sd_lit_hi['difference_coef']:+.6f} "
          f"state-t={sd_lit_hi['t_state_clustered']:.2f} "
          f"wcb-p={sd_lit_hi['wcb_p_value']:.4f}")

    results['item_d'] = {
        "threeway_within_lowIO": {
            **{k: r_lowio[k] for k in ['coef', 'se', 't', 'se_state',
                                       't_state', 'n_obs', 'se_kind']},
            "wild_cluster_bootstrap": wcb_lowio,
            "interpretation": "This three-way IS mom x IV x literacy_z estimated "
                              "WITHIN the high-retail (low-IO) subsample. A "
                              "negative, significant coefficient here pins the "
                              "literacy-modulated reading: it is the literacy "
                              "gradient operating where retail demand is "
                              "marginal and institutional arbitrage capacity is "
                              "low by construction.",
        },
        "lowIO_by_state_literacy": {
            "high_literacy_states": {
                **{k: r_hl[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']},
                "n_states": len(hi_lit_states)},
            "low_literacy_states": {
                **{k: r_ll[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']},
                "n_states": len(lo_lit_states)},
            "stacked_lowlit_minus_highlit": sd_lit,
            "mechanism_prediction": "three-way more negative in low-literacy "
                                    "states WITHIN the high-retail group => "
                                    "stacked difference (lowlit - hilit) < 0",
        },
        "highIO_by_state_literacy_placebo": {
            "stacked_lowlit_minus_highlit": sd_lit_hi,
            "note": "Placebo: within the high-IO (institutional) tercile, the "
                    "mechanism predicts NO literacy gradient — institutions set "
                    "prices, so retail literacy should not modulate the "
                    "interaction. A flat / insignificant difference here "
                    "supports the literacy reading.",
        },
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===")

    # ---- LaTeX tables ----
    _write_tables(results)
    return results


def _write_tables(results):
    # Table: item (b) stacked difference
    with open(os.path.join(TABDIR, "tab_13f_stacked_diff_v4.tex"), 'w',
              encoding='utf-8') as f:
        b = results['item_b']['persistent']
        pt = b['per_tercile_threeway']
        sd = b['stacked_difference']
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma$ & CGM SE & state-cl. SE "
                "& $N$ \\\\\n\\hline\n")
        for g, disp in [('IO1_low', 'Low IO (high retail)'),
                        ('IO2_mid', 'Mid IO'),
                        ('IO3_high', 'High IO (institutional)')]:
            r = pt[g]
            f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                    f"${r['se_state']:.4f}$ & ${r['n_obs']:,}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"Difference (low $-$ high), stacked & "
                f"${sd['difference_coef']:.4f}$ & "
                f"${sd['se_cgm_two_way']:.4f}$ & "
                f"${sd['se_state_clustered']:.4f}$ & "
                f"${sd['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad state-clustered $t$ & "
                f"\\multicolumn{{4}}{{l}}{{${sd['t_state_clustered']:.2f}$}} "
                f"\\\\\n")
        f.write(f"\\quad wild-cluster bootstrap $p$ & "
                f"\\multicolumn{{4}}{{l}}{{${sd['wcb_p_value']:.3f}$ "
                f"($B={sd['B']}$, {sd['n_state_clusters']} state clusters)}} "
                f"\\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    # Table: item (d) within-retail literacy gradient
    with open(os.path.join(TABDIR, "tab_within_retail_literacy_v4.tex"), 'w',
              encoding='utf-8') as f:
        dd = results['item_d']
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Subsample & three-way $\\hat\\gamma$ & CGM $t$ & "
                "state-cl. $t$ & $N$ \\\\\n\\hline\n")
        tw = dd['threeway_within_lowIO']
        f.write(f"Within low-IO (high-retail) & ${tw['coef']:.4f}$ & "
                f"${tw['t']:.2f}$ & ${tw['t_state']:.2f}$ & "
                f"${tw['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad wild-cluster bootstrap $p$ & "
                f"\\multicolumn{{4}}{{l}}{{"
                f"${tw['wild_cluster_bootstrap']['p_value']:.3f}$}} \\\\\n")
        f.write("\\hline\n")
        hl = dd['lowIO_by_state_literacy']['high_literacy_states']
        ll = dd['lowIO_by_state_literacy']['low_literacy_states']
        sdl = dd['lowIO_by_state_literacy']['stacked_lowlit_minus_highlit']
        f.write(f"\\quad low-IO $\\times$ high-literacy states & "
                f"${hl['coef']:.4f}$ & ${hl['t']:.2f}$ & "
                f"${hl['t_state']:.2f}$ & ${hl['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad low-IO $\\times$ low-literacy states & "
                f"${ll['coef']:.4f}$ & ${ll['t']:.2f}$ & "
                f"${ll['t_state']:.2f}$ & ${ll['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad difference (low-lit $-$ high-lit), stacked & "
                f"${sdl['difference_coef']:.4f}$ & --- & "
                f"${sdl['t_state_clustered']:.2f}$ & ${sdl['n_obs']:,}$ \\\\\n")
        f.write(f"\\quad\\quad wild-cluster bootstrap $p$ & "
                f"\\multicolumn{{4}}{{l}}{{${sdl['wcb_p_value']:.3f}$}} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    print(f"  wrote LaTeX tables to {TABDIR}")


if __name__ == '__main__':
    main()
