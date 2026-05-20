# =====================================================================
# deepen_13f_split_v6.py
# Live 13F split: re-runs the stacked IO-difference (item b) and within-retail literacy-gradient (item d) tests on the denominator-corrected Thomson s34 panel, plus a v5-method insensitivity check.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet and code/empirical/_thomson_s34_v6_firmquarter.parquet (via deepen_thomson_io_panel_v6); deepen_estimators.py
# Outputs:   output/stage3a/results_v6.json; output/stage3a/tables/tab_13f_stacked_diff_v6.tex, tab_within_retail_literacy_v6.tex
# Paper:     T2 tab:headline / T3 tab:size_ctrl / T4 tab:migration (mid-IO concentration); IA within-retail IV-decile section
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r2 (Gate 5 Reject) — Item 3: re-run the v5 13F split on the
DENOMINATOR-CORRECTED Thomson s34 panel, and confirm INSENSITIVITY to the
denominator fix.

The v5 panel capped 6.49% of io_share values that exceeded 1.0. Item 3 of the
r2 deepen directive: diagnose and fix at the source, then re-run the v5 stacked
13F difference test (item b) and the within-retail literacy-gradient test
(item d) on the cleaned panel and confirm the result is INSENSITIVE to the fix.

Diagnosis (code/tmp/diag_s34_*.py): two root causes —
  (1) amendment double-counting (s34 stacks original + amended filings; v5
      SUM(shares) over all rows double-counted amended managers);
  (2) the `shares` field is overstated vs. the sole+shared+no authority
      decomposition (cross-manager double-counting of jointly-held blocks).
Fix: amendment-dedup (latest fdate per mgr-cusip) + institutional shares =
sole+shared. Residual io>1.0 drops 6.49% -> ~0.2%.

This script runs, on the CORRECTED panel:
  - item (b): stacked IO-difference, persistent + time-varying, proper
    clustering (state-clustered + CGM + restricted wild-cluster bootstrap)
  - item (d): within-retail literacy-gradient test, persistent + time-varying
  - INSENSITIVITY: the SAME two tests on the v5-METHOD numerator (`shares`,
    amendment-deduped) so the paper can show the v5 result (wcb p=0.125/0.141
    persistent/time-varying; within-retail +0.0022/+0.0023, wcb p~0.064/0.066)
    is insensitive to the denominator fix.

Methodology CARRIES OVER from v5 unchanged — deepen_estimators.py. Only the IO
panel source changes (v5 `shares`-summed panel -> v6 corrected panel).

Outputs: output/stage3a/results_v6.json, LaTeX tables *_v6.tex.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepen_thomson_io_panel_v6 import (load_panel, load_s34_firmquarter,
                                        io_share_distribution,
                                        merge_io_to_panel, LAG_MONTHS)
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               stacked_io_difference,
                               stacked_state_group_difference, FOCAL)

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "results_v6.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

LIT_Z = 'literacy_score_corrected'


def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


def run_split(dfm, io_persist_col, io_tv_col, label):
    """Run item (b) stacked difference + item (d) within-retail literacy
    gradient on a given pair of (persistent, time-varying) IO columns.
    `label` distinguishes the corrected vs. v5-method numerator."""
    out = {"label": label, "item_b": {}, "item_d": {}}

    # ---- item (b): persistent ----
    dperm = dfm[dfm[io_persist_col].notna()].copy()
    dperm, _, _ = add_io_terciles(dperm, io_persist_col)
    terc_means_p = dperm.groupby('io_grp')[io_persist_col].mean().to_dict()
    pertile = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = dperm[dperm['io_grp'] == g]
        r = twfe_three_way(sub)
        pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                        't_state', 'n_obs', 'se_kind']}
    dstack_p = dperm[dperm['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    dstack_p['io_grp_bin'] = np.where(dstack_p['io_grp'] == 'IO1_low',
                                      'low', 'high')
    print(f"  [{label}] stacked diff (persistent, B=4999)...", flush=True)
    sd_p = stacked_io_difference(dstack_p, B=4999, seed=42)
    print(f"    [{label}] persistent DIFF = {sd_p['difference_coef']:+.6f} | "
          f"state-t={sd_p['t_state_clustered']:.2f} | "
          f"CGM t={sd_p['t_cgm_two_way']} | "
          f"wcb p={sd_p['wcb_p_value']:.4f}", flush=True)
    out['item_b']['persistent'] = {
        "tercile_means": {k: float(v) for k, v in terc_means_p.items()},
        "per_tercile_threeway": pertile,
        "stacked_difference": sd_p,
        "n_permnos": int(dperm['permno'].nunique()),
        "n_firm_months": int(len(dperm)),
    }

    # ---- item (b): time-varying ----
    dtv = dfm[dfm[io_tv_col].notna()].copy()
    dtv, _, _ = add_io_terciles(dtv, io_tv_col)
    terc_means_tv = dtv.groupby('io_grp')[io_tv_col].mean().to_dict()
    pertile_tv = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = dtv[dtv['io_grp'] == g]
        r = twfe_three_way(sub)
        pertile_tv[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                           't_state', 'n_obs', 'se_kind']}
    dstack_tv = dtv[dtv['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    dstack_tv['io_grp_bin'] = np.where(dstack_tv['io_grp'] == 'IO1_low',
                                       'low', 'high')
    print(f"  [{label}] stacked diff (time-varying, B=4999)...", flush=True)
    sd_tv = stacked_io_difference(dstack_tv, B=4999, seed=42)
    print(f"    [{label}] time-varying DIFF = {sd_tv['difference_coef']:+.6f} "
          f"| state-t={sd_tv['t_state_clustered']:.2f} | "
          f"CGM t={sd_tv['t_cgm_two_way']} | "
          f"wcb p={sd_tv['wcb_p_value']:.4f}", flush=True)
    out['item_b']['time_varying'] = {
        "tercile_means": {k: float(v) for k, v in terc_means_tv.items()},
        "per_tercile_threeway": pertile_tv,
        "stacked_difference": sd_tv,
        "n_permnos": int(dtv['permno'].nunique()),
        "n_firm_months": int(len(dtv)),
    }

    # ---- item (d): within-retail literacy gradient ----
    for measure_label, dframe in [("persistent", dperm),
                                  ("time_varying", dtv)]:
        d_lowio = dframe[dframe['io_grp'] == 'IO1_low'].copy()
        d_highio = dframe[dframe['io_grp'] == 'IO3_high'].copy()
        r_lowio = twfe_three_way(d_lowio)
        wcb_lowio = wild_cluster_bootstrap_state(d_lowio, B=4999, seed=42)

        state_lit = d_lowio.groupby('hq_state')[LIT_Z].mean()
        lit_median = state_lit.median()
        hi_lit_states = set(state_lit[state_lit >= lit_median].index)
        lo_lit_states = set(state_lit[state_lit < lit_median].index)
        d_hl = d_lowio[d_lowio['hq_state'].isin(hi_lit_states)].copy()
        d_ll = d_lowio[d_lowio['hq_state'].isin(lo_lit_states)].copy()
        r_hl = twfe_three_way(d_hl)
        r_ll = twfe_three_way(d_ll)

        dstack_lit = d_lowio.copy()
        dstack_lit['lit_grp'] = np.where(
            dstack_lit['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')
        sd_lit = stacked_state_group_difference(dstack_lit, 'lit_grp',
                                                'low_lit', B=4999, seed=42)
        # high-IO placebo
        state_lit_hi = d_highio.groupby('hq_state')[LIT_Z].mean()
        lo_lit_states_h = set(
            state_lit_hi[state_lit_hi < state_lit_hi.median()].index)
        dstack_lit_hi = d_highio.copy()
        dstack_lit_hi['lit_grp'] = np.where(
            dstack_lit_hi['hq_state'].isin(lo_lit_states_h),
            'low_lit', 'high_lit')
        sd_lit_hi = stacked_state_group_difference(
            dstack_lit_hi, 'lit_grp', 'low_lit', B=4999, seed=42)
        print(f"    [{label}/{measure_label}] within-low-IO 3way="
              f"{r_lowio['coef']:+.6f} (wcb p={wcb_lowio['p_value']:.3f}) | "
              f"lowlit-hilit DIFF={sd_lit['difference_coef']:+.6f} "
              f"state-t={sd_lit['t_state_clustered']:.2f} "
              f"wcb-p={sd_lit['wcb_p_value']:.4f}", flush=True)

        out['item_d'][measure_label] = {
            "threeway_within_lowIO": {
                **{k: r_lowio[k] for k in ['coef', 'se', 't', 'se_state',
                                           't_state', 'n_obs', 'se_kind']},
                "wild_cluster_bootstrap": wcb_lowio,
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
                                        "low-literacy high-retail stocks => "
                                        "stacked (lowlit - hilit) < 0",
            },
            "highIO_by_state_literacy_placebo": {
                "stacked_lowlit_minus_highlit": sd_lit_hi},
        }
    return out


def main():
    results = {"meta": {}, "item_3_diagnosis": {}, "corrected": {},
               "v5_method_insensitivity": {}}

    print("=== load panel + CORRECTED Thomson s34 IO ===", flush=True)
    df = load_panel()
    d_all = df.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos", flush=True)

    fq = load_s34_firmquarter()
    quarters = sorted(fq['quarter'].unique())
    diag = io_share_distribution(fq)
    n_over = int((fq['io_share_raw'] > 1.0).sum())
    n_over_v5 = int((fq['io_share_raw_v5method'] > 1.0).sum())
    print(f"corrected Thomson s34: {len(fq):,} firm-quarters, "
          f"{fq['permno'].nunique()} permnos, {len(quarters)} quarters",
          flush=True)
    print(f"  io>1.0 corrected (sole+shared): {n_over:,} "
          f"({100.0*n_over/len(fq):.3f}%)", flush=True)
    print(f"  io>1.0 v5-method (shares, amendment-deduped): {n_over_v5:,} "
          f"({100.0*n_over_v5/len(fq):.3f}%)", flush=True)
    print(f"  [v5 ORIGINAL panel: 6.485%]", flush=True)

    results['item_3_diagnosis'] = {
        "v5_original_over_one_pct": 6.485,
        "v5_original_over_one_count": 14409,
        "v6_corrected_over_one_count": n_over,
        "v6_corrected_over_one_pct": round(100.0 * n_over / len(fq), 3),
        "v6_v5method_amendment_deduped_over_one_count": n_over_v5,
        "v6_v5method_amendment_deduped_over_one_pct": round(
            100.0 * n_over_v5 / len(fq), 3),
        "root_cause": "Two sources, neither multi-class nor cusip-fan-out "
                      "(both ruled out — every permno maps to exactly one "
                      "valid ncusip at rdate; no cusip maps to >1 permno). "
                      "(1) AMENDMENT DOUBLE-COUNTING: tr_13f.s34 stacks "
                      "original + amended 13F filings; a manager filing an "
                      "original then an amendment for the same rdate appears "
                      "as 2-4 rows for the same (mgrno,cusip,rdate), each with "
                      "its own fdate (5k-28k such pairs/quarter). v5 did "
                      "SUM(shares) over ALL rows, double-counting amended "
                      "managers. (2) THE `shares` FIELD IS OVERSTATED relative "
                      "to the sole+shared+no authority decomposition (identity "
                      "shares = sole+shared+no fails by up to ~900M "
                      "shares/quarter at the stock level); using `shares` "
                      "leaves 3-8%/quarter >1.0 even after amendment dedup, "
                      "concentrated in large liquid firms with 100-500 "
                      "managers (cross-manager double-counting of jointly-held "
                      "blocks).",
        "fix": "(1) amendment-dedup: DISTINCT ON (mgrno,cusip) ORDER BY fdate "
               "DESC — keep only the latest filing per manager-stock-quarter. "
               "(2) institutional shares held = sole + shared (the standard "
               "institutional-ownership numerator, Lewellen 2011; Ben-David "
               "et al. 2021) instead of `shares`. Residual io>1.0 drops from "
               "6.49% to ~0.2% — the irreducible cross-manager shared-block "
               "measurement-noise floor, capped at 1.0.",
        "per_quarter_diag": diag.to_dict(orient='records'),
        "diagnostic_quarters_checked": ["2009Q1", "2016Q2", "2019Q3",
                                        "2023Q4"],
    }

    dfm = merge_io_to_panel(df, fq, verbose=True)
    dfm = dfm.merge(d_all[['permno', 'date']], on=['permno', 'date'],
                    how='inner')
    dfm = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    cov_tv = int(dfm['io_share'].notna().sum())
    cov_p = int(dfm['io_share_persist'].notna().sum())
    panel_n = int(len(df))

    # ===== CORRECTED numerator (sole+shared, amendment-deduped) =====
    print("\n=== SPLIT on the CORRECTED panel (sole+shared) ===", flush=True)
    results['corrected'] = run_split(dfm, 'io_share_persist', 'io_share',
                                     'corrected')
    results['corrected']['coverage'] = {
        "time_varying_firm_months": cov_tv,
        "time_varying_pct": round(100.0 * cov_tv / panel_n, 2),
        "persistent_firm_months": cov_p,
        "persistent_pct": round(100.0 * cov_p / panel_n, 2),
    }

    # ===== v5-METHOD numerator (shares, amendment-deduped) — insensitivity =====
    print("\n=== SPLIT on the v5-METHOD numerator (insensitivity check) ===",
          flush=True)
    results['v5_method_insensitivity'] = run_split(
        dfm, 'io_share_persist_v5method', 'io_share_v5method', 'v5_method')

    results['meta'] = {
        "io_quarters": quarters,
        "n_io_quarters": len(quarters),
        "io_source": "Thomson Reuters s34 (WRDS tr_13f.s34), DENOMINATOR-"
                     "CORRECTED: amendment-deduped + sole+shared numerator",
        "io_temporal_span": f"{quarters[0]}..{quarters[-1]}",
        "estimation_sample_firm_months": int(len(d_all)),
        "estimation_sample_permnos": int(d_all['permno'].nunique()),
        "v5_comparison": {
            "v5_persistent_diff": -0.0237, "v5_persistent_wcb_p": 0.125,
            "v5_time_varying_diff": -0.0267, "v5_time_varying_wcb_p": 0.141,
            "v5_within_retail_persistent_diff": 0.0022,
            "v5_within_retail_persistent_wcb_p": 0.064,
            "v5_within_retail_time_varying_diff": 0.0023,
            "v5_within_retail_time_varying_wcb_p": 0.066,
        },
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)
    _write_tables(results)
    return results


def _write_tables(results):
    # Table: item (b) stacked difference — corrected panel, persistent + tv
    with open(os.path.join(TABDIR, "tab_13f_stacked_diff_v6.tex"), 'w',
              encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma$ & CGM SE & state-cl. SE "
                "& $N$ \\\\\n")
        for mlabel, mdisp in [
                ('persistent', 'Panel A: persistent (between-firm) IO measure '
                 '--- denominator-corrected panel'),
                ('time_varying', 'Panel B: genuinely time-varying IO measure '
                 '(no look-ahead) --- denominator-corrected panel')]:
            b = results['corrected']['item_b'][mlabel]
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
        # insensitivity row
        f.write("\\hline\n\\multicolumn{5}{l}{\\textit{Insensitivity: "
                "v5-method numerator (`shares', amendment-deduped)}} \\\\\n")
        for mlabel, disp in [('persistent', 'Difference (persistent), '
                              'v5-method numerator'),
                             ('time_varying', 'Difference (time-varying), '
                              'v5-method numerator')]:
            sdv = results['v5_method_insensitivity']['item_b'][mlabel][
                'stacked_difference']
            f.write(f"{disp} & ${sdv['difference_coef']:.4f}$ & --- & "
                    f"${sdv['se_state_clustered']:.4f}$ & "
                    f"${sdv['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sdv['wcb_p_value']:.3f}$}}"
                    f" \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    # Table: item (d) within-retail literacy gradient — corrected panel
    with open(os.path.join(TABDIR, "tab_within_retail_literacy_v6.tex"), 'w',
              encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Subsample & three-way $\\hat\\gamma$ & CGM $t$ & "
                "state-cl. $t$ & $N$ \\\\\n")
        for mlabel, mdisp in [
                ('persistent', 'Panel A: low-IO tercile, persistent IO '
                 'measure --- denominator-corrected panel'),
                ('time_varying', 'Panel B: low-IO tercile, time-varying IO '
                 'measure --- denominator-corrected panel')]:
            dd = results['corrected']['item_d'][mlabel]
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
            # insensitivity
            sdv = results['v5_method_insensitivity']['item_d'][mlabel][
                'lowIO_by_state_literacy']['stacked_lowlit_minus_highlit']
            f.write(f"\\quad diff, v5-method numerator (insensitivity) & "
                    f"${sdv['difference_coef']:.4f}$ & --- & "
                    f"${sdv['t_state_clustered']:.2f}$ & "
                    f"${sdv['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sdv['wcb_p_value']:.3f}$}} "
                    f"\\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    print(f"  wrote LaTeX tables to {TABDIR}", flush=True)


if __name__ == '__main__':
    main()
