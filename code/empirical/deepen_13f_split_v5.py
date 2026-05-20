# =====================================================================
# deepen_13f_split_v5.py
# Re-runs the full 13F split (items a-d) on the real 60-quarter Thomson s34 IO panel: coverage report, stacked low-IO-minus-high-IO difference (persistent + time-varying), size placebo, and within-retail literacy-gradient test.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet and code/empirical/_thomson_s34_firmquarter.parquet (via deepen_thomson_io_panel); deepen_estimators.py
# Outputs:   output/stage3a/results_v5.json; output/stage3a/tables/tab_13f_stacked_diff_v5.tex, tab_within_retail_literacy_v5.tex
# Paper:     SUPERSEDED by deepen_13f_split_v6.py — kept for provenance (pre-denominator-correction Thomson split; feeds T2/T3/T4 lineage + IA Thomson s34 construction section)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 (WRDS-recovered re-fire) — items (a)-(d) on the REAL
Thomson Reuters s34 13F institutional-ownership panel.

This is the round-1 deepen done with the canonical data source. The round-1
referees converged that the EDGAR 2-quarter / 6-quarter proxy is too weak to be
the centerpiece — they asked for the real quarterly Thomson s34 panel. WRDS has
recovered; this script consumes deepen_thomson_io_panel (the genuine 60-quarter
2009Q1-2023Q4 institutional-ownership panel) and re-runs the entire 13F split.

Methodology CARRIES OVER from v4 unchanged — deepen_estimators.py (stacked
fully-interacted difference SE, restricted wild-cluster bootstrap, state-defined
group difference). Only the IO data source is swapped: EDGAR-proxy ->
Thomson s34.

Item (a): panel coverage report — quarters, firm-month coverage %, IO_share
          distribution + io>1.0 rate (printed by deepen_thomson_io_panel).
Item (b): STACKED 13F difference test, state-clustered + two-way CGM + wild-
          cluster bootstrap, on (1) the persistent (time-mean) IO measure and
          (2) the genuinely time-varying no-look-ahead IO measure. The real
          Thomson panel is the arbiter: does the sign flip survive on the
          genuinely time-varying panel with proper clustering?
Item (c): size-tercile placebo on the balanced panel (IO-covered firms and
          180-month-balanced firms) — does a generic size proxy reproduce the
          sign flip? It should NOT.
Item (d): within-retail (low-IO tercile) literacy-gradient test — does the
          three-way mom x IV x literacy_z vary with state literacy WITHIN the
          high-retail subsample? This pins (or bounds) the title's literacy
          construct.

Outputs: output/stage3a/results_v5.json, LaTeX tables to output/stage3a/tables/
*_v5.tex. The markdown writeup is hand-assembled into empirical_analysis_v5.md.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepen_thomson_io_panel import (load_panel, load_s34_firmquarter,
                                     io_share_distribution, merge_io_to_panel,
                                     LAG_MONTHS)
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               stacked_io_difference,
                               stacked_state_group_difference, FOCAL)

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "results_v5.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

LIT_Z = 'literacy_score_corrected'


def add_io_terciles(d, io_col):
    """Assign IO terciles on the firm-level mean of io_col."""
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


def main():
    results = {"meta": {}, "item_a": {}, "item_b": {}, "item_c": {},
               "item_d": {}}

    print("=== load panel + Thomson s34 IO ===", flush=True)
    df = load_panel()
    d_all = df.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos", flush=True)

    fq = load_s34_firmquarter()
    quarters = sorted(fq['quarter'].unique())
    diag = io_share_distribution(fq)
    n_over_fq = int((fq['io_share_raw'] > 1.0).sum())
    print(f"Thomson s34 firm-quarter: {len(fq):,} rows, "
          f"{fq['permno'].nunique()} permnos, {len(quarters)} quarters "
          f"({quarters[0]}..{quarters[-1]})", flush=True)
    print(f"  io_share_raw > 1.0: {n_over_fq:,} "
          f"({100.0*n_over_fq/len(fq):.2f}%)", flush=True)

    dfm = merge_io_to_panel(df, fq, verbose=True)
    dfm = dfm.merge(d_all[['permno', 'date']], on=['permno', 'date'],
                    how='inner')
    dfm = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    # ---- item (a): coverage report ----
    cov_tv = int(dfm['io_share'].notna().sum())
    cov_p = int(dfm['io_share_persist'].notna().sum())
    panel_n = int(len(df))
    results['item_a'] = {
        "data_source": "Thomson Reuters s34 13F institutional holdings (WRDS "
                       "tr_13f.s34), aggregated cusip x rdate, summed across "
                       "managers; cusip->permno via crsp.stocknames ncusip "
                       "valid at rdate (common equity shrcd 10/11); shrout "
                       "denominator from crsp.msf at the rdate month-end.",
        "quarters": quarters,
        "n_quarters": len(quarters),
        "temporal_span": f"{quarters[0]}..{quarters[-1]}",
        "firm_quarter_rows": int(len(fq)),
        "firm_quarter_permnos": int(fq['permno'].nunique()),
        "io_raw_over_one_count": n_over_fq,
        "io_raw_over_one_pct": round(100.0 * n_over_fq / len(fq), 3),
        "io_share_raw_describe": {k: float(v) for k, v in
                                  fq['io_share_raw'].describe().items()},
        "panel_firm_months_total": panel_n,
        "time_varying_coverage_firm_months": cov_tv,
        "time_varying_coverage_pct": round(100.0 * cov_tv / panel_n, 2),
        "persistent_coverage_firm_months": cov_p,
        "persistent_coverage_pct": round(100.0 * cov_p / panel_n, 2),
        "estimation_sample_firm_months": int(len(d_all)),
        "estimation_sample_permnos": int(d_all['permno'].nunique()),
        "no_look_ahead_lag_months": LAG_MONTHS,
        "per_quarter_diag": diag.to_dict(orient='records'),
    }
    print(f"  [item a] panel coverage: time-varying {cov_tv:,} "
          f"({100.0*cov_tv/panel_n:.1f}%), persistent {cov_p:,} "
          f"({100.0*cov_p/panel_n:.1f}%) of {panel_n:,} firm-months", flush=True)

    # ===================================================================
    # ITEM (b): stacked IO-difference test, proper clustering
    # ===================================================================
    print("\n=== ITEM (b): stacked IO-difference, proper clustering ===",
          flush=True)

    # --- (b1) persistent IO measure (between-firm split) ---
    dperm = dfm[dfm['io_share_persist'].notna()].copy()
    dperm, perm_io_p, _ = add_io_terciles(dperm, 'io_share_persist')
    terc_means_p = dperm.groupby('io_grp')['io_share_persist'].mean().to_dict()
    print(f"  persistent IO: {dperm['permno'].nunique()} permnos, "
          f"{len(dperm):,} firm-months; tercile means "
          f"{ {k: round(v,3) for k,v in terc_means_p.items()} }", flush=True)

    pertile = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = dperm[dperm['io_grp'] == g]
        r = twfe_three_way(sub)
        pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']}
        print(f"    {g}: coef={r['coef']:.6f} cgm_t={r['t']:.2f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}", flush=True)

    dstack_p = dperm[dperm['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    dstack_p['io_grp_bin'] = np.where(dstack_p['io_grp'] == 'IO1_low',
                                      'low', 'high')
    print("  running stacked difference (persistent IO, B=4999)...", flush=True)
    sd_p = stacked_io_difference(dstack_p, B=4999, seed=42)
    print(f"    DIFF (low-high) = {sd_p['difference_coef']:+.6f} | "
          f"state-clustered SE {sd_p['se_state_clustered']:.6f} "
          f"t={sd_p['t_state_clustered']:.2f} | "
          f"CGM t={sd_p['t_cgm_two_way']} | "
          f"wild-cluster bootstrap p={sd_p['wcb_p_value']:.4f}", flush=True)

    results['item_b']['persistent'] = {
        "io_measure": "time-mean io_share across the firm's 60-quarter "
                      "Thomson s34 history (between-firm split variable)",
        "n_quarters": len(quarters),
        "tercile_means": {k: float(v) for k, v in terc_means_p.items()},
        "per_tercile_threeway": pertile,
        "stacked_difference": sd_p,
        "n_permnos": int(dperm['permno'].nunique()),
        "n_firm_months": int(len(dperm)),
    }

    # --- (b2) genuinely time-varying IO measure, no look-ahead ---
    dtv = dfm[dfm['io_share'].notna()].copy()
    dtv, _, _ = add_io_terciles(dtv, 'io_share')
    terc_means_tv = dtv.groupby('io_grp')['io_share'].mean().to_dict()
    print(f"  time-varying IO: {dtv['permno'].nunique()} permnos, "
          f"{len(dtv):,} firm-months; tercile means "
          f"{ {k: round(v,3) for k,v in terc_means_tv.items()} }", flush=True)
    pertile_tv = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = dtv[dtv['io_grp'] == g]
        r = twfe_three_way(sub)
        pertile_tv[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                           't_state', 'n_obs', 'se_kind']}
        print(f"    [tv] {g}: coef={r['coef']:.6f} cgm_t={r['t']:.2f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}", flush=True)
    dstack_tv = dtv[dtv['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    dstack_tv['io_grp_bin'] = np.where(dstack_tv['io_grp'] == 'IO1_low',
                                       'low', 'high')
    print("  running stacked difference (time-varying IO, B=4999)...",
          flush=True)
    sd_tv = stacked_io_difference(dstack_tv, B=4999, seed=42)
    print(f"    [time-varying] DIFF = {sd_tv['difference_coef']:+.6f} | "
          f"state-clustered t={sd_tv['t_state_clustered']:.2f} | "
          f"CGM t={sd_tv['t_cgm_two_way']} | "
          f"wcb p={sd_tv['wcb_p_value']:.4f}", flush=True)
    results['item_b']['time_varying'] = {
        "io_measure": f"most-recent-quarter io_share, no look-ahead "
                      f"(merge_asof backward, rdate lagged {LAG_MONTHS} "
                      f"months — the genuinely time-varying within-stock "
                      f"contrast the referees asked for)",
        "n_quarters": len(quarters),
        "tercile_means": {k: float(v) for k, v in terc_means_tv.items()},
        "per_tercile_threeway": pertile_tv,
        "stacked_difference": sd_tv,
        "n_permnos": int(dtv['permno'].nunique()),
        "n_firm_months": int(len(dtv)),
    }

    # ===================================================================
    # ITEM (c): size-tercile placebo on the BALANCED panel
    # ===================================================================
    print("\n=== ITEM (c): size placebo on balanced panel ===", flush=True)
    io_permnos = set(dperm['permno'].unique())
    d_iocov = d_all[d_all['permno'].isin(io_permnos)].copy()
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
              f"wcb-p={sd['wcb_p_value']:.4f}", flush=True)
        return out

    print("  balanced panel A: IO-covered firms (same sample as the IO split)",
          flush=True)
    sp_iocov = size_placebo(d_iocov, "IO-covered")
    print("  balanced panel B: permnos present all 180 months", flush=True)
    sp_bal180 = size_placebo(d_bal180, "balanced-180")
    print("  full panel (reference)", flush=True)
    sp_full = size_placebo(d_all, "full-panel")

    results['item_c'] = {
        "balanced_io_covered": sp_iocov,
        "balanced_180month": sp_bal180,
        "full_panel_reference": sp_full,
        "note": "Size placebo: a generic firm-size proxy should NOT reproduce "
                "the Thomson 13F IO sign flip. balanced_io_covered restricts "
                "to the exact firms in the IO split; balanced_180month "
                "restricts to permnos present all 180 months.",
    }

    # ===================================================================
    # ITEM (d): within-retail (low-IO) literacy-gradient test
    # ===================================================================
    print("\n=== ITEM (d): within-retail literacy-gradient test ===",
          flush=True)
    # Run on BOTH the persistent and time-varying IO terciles. The persistent
    # terciles give the same firms throughout (cleaner subsample); the
    # time-varying terciles are the contemporaneous-ownership version.
    for measure_label, dframe, io_col in [
            ("persistent", dperm, 'io_share_persist'),
            ("time_varying", dtv, 'io_share')]:
        print(f"  --- within-retail literacy gradient on the {measure_label} "
              f"IO terciles ---", flush=True)
        d_lowio = dframe[dframe['io_grp'] == 'IO1_low'].copy()
        d_highio = dframe[dframe['io_grp'] == 'IO3_high'].copy()
        print(f"    low-IO (high-retail): {len(d_lowio):,} firm-months, "
              f"{d_lowio['permno'].nunique()} permnos", flush=True)

        r_lowio = twfe_three_way(d_lowio)
        print(f"    three-way within low-IO: coef={r_lowio['coef']:.6f} "
              f"cgm_t={r_lowio['t']:.2f} state_t={r_lowio['t_state']:.2f}",
              flush=True)
        wcb_lowio = wild_cluster_bootstrap_state(d_lowio, B=4999, seed=42)
        print(f"      wcb p={wcb_lowio['p_value']:.4f} "
              f"CI={[round(c,5) for c in wcb_lowio['ci_studentized']]}",
              flush=True)

        # split low-IO by state literacy
        state_lit = d_lowio.groupby('hq_state')[LIT_Z].mean()
        lit_median = state_lit.median()
        hi_lit_states = set(state_lit[state_lit >= lit_median].index)
        lo_lit_states = set(state_lit[state_lit < lit_median].index)
        d_lowio_hilit = d_lowio[d_lowio['hq_state'].isin(hi_lit_states)].copy()
        d_lowio_lolit = d_lowio[d_lowio['hq_state'].isin(lo_lit_states)].copy()
        r_hl = twfe_three_way(d_lowio_hilit)
        r_ll = twfe_three_way(d_lowio_lolit)
        print(f"      low-IO+high-lit: {r_hl['coef']:.6f} "
              f"(state_t={r_hl['t_state']:.2f}, n={r_hl['n_obs']:,})",
              flush=True)
        print(f"      low-IO+low-lit:  {r_ll['coef']:.6f} "
              f"(state_t={r_ll['t_state']:.2f}, n={r_ll['n_obs']:,})",
              flush=True)

        dstack_lit = d_lowio.copy()
        dstack_lit['lit_grp'] = np.where(
            dstack_lit['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')
        print("    stacked low-lit-minus-high-lit difference within low-IO "
              "(B=4999)...", flush=True)
        sd_lit = stacked_state_group_difference(dstack_lit, 'lit_grp',
                                                'low_lit', B=4999, seed=42)
        print(f"      DIFF(lowlit-hilit | low-IO) = "
              f"{sd_lit['difference_coef']:+.6f} "
              f"state-t={sd_lit['t_state_clustered']:.2f} "
              f"wcb-p={sd_lit['wcb_p_value']:.4f}", flush=True)

        # placebo: same literacy split WITHIN the high-IO tercile
        state_lit_hi = d_highio.groupby('hq_state')[LIT_Z].mean()
        lit_median_hi = state_lit_hi.median()
        lo_lit_states_h = set(
            state_lit_hi[state_lit_hi < lit_median_hi].index)
        dstack_lit_hi = d_highio.copy()
        dstack_lit_hi['lit_grp'] = np.where(
            dstack_lit_hi['hq_state'].isin(lo_lit_states_h),
            'low_lit', 'high_lit')
        print("    stacked low-lit-minus-high-lit difference within HIGH-IO "
              "(placebo, B=4999)...", flush=True)
        sd_lit_hi = stacked_state_group_difference(
            dstack_lit_hi, 'lit_grp', 'low_lit', B=4999, seed=42)
        print(f"      DIFF(lowlit-hilit | high-IO) = "
              f"{sd_lit_hi['difference_coef']:+.6f} "
              f"state-t={sd_lit_hi['t_state_clustered']:.2f} "
              f"wcb-p={sd_lit_hi['wcb_p_value']:.4f}", flush=True)

        results['item_d'][measure_label] = {
            "threeway_within_lowIO": {
                **{k: r_lowio[k] for k in ['coef', 'se', 't', 'se_state',
                                           't_state', 'n_obs', 'se_kind']},
                "wild_cluster_bootstrap": wcb_lowio,
                "interpretation": "This three-way IS mom x IV x literacy_z "
                                  "estimated WITHIN the high-retail (low-IO) "
                                  "subsample. A negative, significant "
                                  "coefficient here pins the "
                                  "literacy-modulated reading.",
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
                "mechanism_prediction": "three-way more negative in "
                                        "low-literacy states WITHIN the "
                                        "high-retail group => stacked "
                                        "difference (lowlit - hilit) < 0",
            },
            "highIO_by_state_literacy_placebo": {
                "stacked_lowlit_minus_highlit": sd_lit_hi,
                "note": "Placebo: within the high-IO (institutional) tercile, "
                        "the mechanism predicts NO literacy gradient.",
            },
        }

    results['meta'] = {
        "io_quarters": quarters,
        "n_io_quarters": len(quarters),
        "io_source": "Thomson Reuters s34 (WRDS tr_13f.s34)",
        "io_temporal_span": f"{quarters[0]}..{quarters[-1]}",
        "estimation_sample_firm_months": int(len(d_all)),
        "estimation_sample_permnos": int(d_all['permno'].nunique()),
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)

    _write_tables(results)
    return results


def _write_tables(results):
    # Table: item (b) stacked difference — persistent AND time-varying
    with open(os.path.join(TABDIR, "tab_13f_stacked_diff_v5.tex"), 'w',
              encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma$ & CGM SE & state-cl. SE "
                "& $N$ \\\\\n")
        for mlabel, mdisp in [('persistent', 'Panel A: persistent '
                               '(between-firm) IO measure'),
                              ('time_varying', 'Panel B: genuinely '
                               'time-varying IO measure (no look-ahead)')]:
            b = results['item_b'][mlabel]
            pt = b['per_tercile_threeway']
            sd = b['stacked_difference']
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            for g, disp in [('IO1_low', 'Low IO (high retail)'),
                            ('IO2_mid', 'Mid IO'),
                            ('IO3_high', 'High IO (institutional)')]:
                r = pt[g]
                f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                        f"${r['se_state']:.4f}$ & ${r['n_obs']:,}$ \\\\\n")
            cgm = (f"${sd['se_cgm_two_way']:.4f}$"
                   if sd['se_cgm_two_way'] is not None
                   and not (isinstance(sd['se_cgm_two_way'], float)
                            and np.isnan(sd['se_cgm_two_way']))
                   else "n.p.s.d.")
            f.write(f"Difference (low $-$ high), stacked & "
                    f"${sd['difference_coef']:.4f}$ & {cgm} & "
                    f"${sd['se_state_clustered']:.4f}$ & "
                    f"${sd['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad state-clustered $t$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sd['t_state_clustered']:.2f}$}}"
                    f" \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sd['wcb_p_value']:.3f}$ "
                    f"($B={sd['B']}$, {sd['n_state_clusters']} state "
                    f"clusters)}} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    # Table: item (d) within-retail literacy gradient (time-varying measure)
    with open(os.path.join(TABDIR, "tab_within_retail_literacy_v5.tex"), 'w',
              encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Subsample & three-way $\\hat\\gamma$ & CGM $t$ & "
                "state-cl. $t$ & $N$ \\\\\n")
        for mlabel, mdisp in [('persistent', 'Panel A: low-IO tercile, '
                               'persistent IO measure'),
                              ('time_varying', 'Panel B: low-IO tercile, '
                               'time-varying IO measure')]:
            dd = results['item_d'][mlabel]
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            tw = dd['threeway_within_lowIO']
            f.write(f"Within low-IO (high-retail) & ${tw['coef']:.4f}$ & "
                    f"${tw['t']:.2f}$ & ${tw['t_state']:.2f}$ & "
                    f"${tw['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{"
                    f"${tw['wild_cluster_bootstrap']['p_value']:.3f}$}} \\\\\n")
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
                    f"${sdl['t_state_clustered']:.2f}$ & "
                    f"${sdl['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sdl['wcb_p_value']:.3f}$}} "
                    f"\\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    print(f"  wrote LaTeX tables to {TABDIR}", flush=True)


if __name__ == '__main__':
    main()
