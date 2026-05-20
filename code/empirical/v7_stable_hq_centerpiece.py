# =====================================================================
# v7_stable_hq_centerpiece.py
# Re-fires the 13F IO sign-flip and within-retail literacy-gradient tests on the
# stable-HQ subsample (firms with no in-sample HQ relocation) under static HQ
# literacy, producing the stable-HQ headline tables.
#
# Inputs:    _dfm_pit_hq_v7_centerpiece.parquet (cached PIT firm-month panel used
#            as the IO-merged vehicle) and _hq_edgar_state_v6.parquet (SEC EDGAR
#            10-K-header HQ-state history); writes/loads _dfm_stable_hq.parquet.
# Outputs:   output/stage3a/results_v7_stable_hq_centerpiece.json;
#            output/stage3a/tables/tab_13f_stacked_diff_stable_hq.tex;
#            output/stage3a/tables/tab_within_retail_literacy_stable_hq.tex
# Paper:     Main Table T2 tab:headline / T5 tab:iv_within_mid_io region (stable-HQ
#            headline) + Internet Appendix stable-HQ construction section
# Run order: see code/00_master.py
# =====================================================================

"""v7 STABLE-HQ CENTERPIECE — operator-mandated re-fire after v7 Q5/PIT-HQ
collapse.

Background and theoretical motivation (operator-supplied):

  v7 Q5 / PIT-HQ centerpiece showed that the point-in-time HQ literacy
  reassignment collapses every quantitative claim of v6 (aggregate
  three-way -0.0117 -> -0.0003, IO sign flip DIFF -0.0211 -> +0.00029).

  The operator's decision: the *stable-HQ subsample* — firms with no HQ
  relocation during the sample window — is the THEORETICALLY CORRECT
  sample for testing the local-bias channel, not a methodologically
  defensive restriction.

  The channel operates through community ties that take time to form.
  For firms with a stable HQ throughout the sample, the local-investor
  base / community-ties relationship is well-defined and the state
  literacy assignment is the correct moderator. For firms that relocate
  HQ during the sample, the local-investor base is in transition (old
  ties decaying, new ones not yet formed) and is *ambiguous by
  construction* — neither the old-HQ nor the new-HQ state literacy
  correctly characterizes the effective local-investor base during/around
  the relocation window. The stable-HQ subsample is therefore the clean
  test of the seed's channel.

Two tasks:

  Task 1 - 13F IO sign flip on stable-HQ (static HQ literacy, the seed's
           convention; restricted to firms with no HQ relocation
           in-sample). Stacked fully-interacted spec on low_IO vs high_IO
           terciles, persistent + time-varying IO.

  Task 2 - Within-retail (low-IO) literacy-gradient test on stable-HQ:
           split firms in the low-IO (high-retail) tercile by static HQ
           state literacy (median split), run TWFE in each subsample,
           report stacked low_lit - high_lit difference + WCB p.

Sample definition (same as v6 Task A drop test):

  relocator = flag_hdr_changed OR flag_sec_disagree
    flag_hdr_changed = 10-K-header STATE changed across the firm's
      in-sample 10-Ks (genuine point-in-time relocation).
    flag_sec_disagree = most-recent 10-K-header STATE disagrees with the
      panel's static hq_state snapshot, firm has >=2 parsed 10-K headers.
  stable-HQ = NOT relocator (over firms with >=1 parsed 10-K) PLUS firms
    not covered by the flag at all are KEPT (consistent with v6 drop test
    which only drops *confirmed* relocators). 1,366 confirmed relocators
    are dropped.

Output: output/stage3a/results_v7_stable_hq_centerpiece.json.
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
from deepen_thomson_io_panel_v6 import LAG_MONTHS

np.random.seed(42)

HQ_EDGAR = os.path.join(EMP, "_hq_edgar_state_v6.parquet")
DFM_PIT_CACHE = os.path.join(EMP, "_dfm_pit_hq_v7_centerpiece.parquet")
DFM_STABLE_CACHE = os.path.join(EMP, "_dfm_stable_hq.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a",
                        "results_v7_stable_hq_centerpiece.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

B_WCB = 4999


def last_hdr_state(s):
    """Get the LAST 10-K-header STATE from the JSON-encoded hdr_states column."""
    if not isinstance(s, str):
        return None
    try:
        seq = json.loads(s)
    except Exception:
        return None
    return seq[-1][1] if seq else None


def build_relocator_set(hq, panel_hq):
    """Build the 1,366-firm relocator permno set using the v6 Task A definition:
       reloc = flag_hdr_changed OR flag_sec_disagree.
    Returns (relocator_permnos: set, diag: dict)."""
    panel_hq = panel_hq.copy()
    panel_hq['hq_state_2l'] = panel_hq['hq_state'].str.replace(
        'US-', '', regex=False)
    m = panel_hq.merge(hq, on='permno', how='left')
    m['flag_hdr_changed'] = (m['hdr_changed'] == True)
    m['hdr_state_last'] = m['hdr_states'].apply(last_hdr_state)
    m['n_10k'] = m['n_10k'].fillna(0).astype(int)
    m['flag_sec_disagree'] = (
        (m['n_10k'] >= 2)
        & m['hdr_state_last'].notna()
        & m['hq_state_2l'].notna()
        & (m['hdr_state_last'] != m['hq_state_2l'])
    )
    m['reloc'] = m['flag_hdr_changed'] | m['flag_sec_disagree']
    m['flag_covered'] = m['n_10k'] >= 1
    diag = {
        "panel_firms": int(len(m)),
        "flag_covered_firms": int(m['flag_covered'].sum()),
        "flag1_hdr_changed_count": int(m['flag_hdr_changed'].sum()),
        "flag2_sec_disagree_count": int(m['flag_sec_disagree'].sum()),
        "union_reloc_count": int(m['reloc'].sum()),
        "definition": "reloc = flag_hdr_changed OR flag_sec_disagree "
                      "(SAME as v6 Task A drop test); 1,366 firms expected.",
    }
    print(f"  panel firms: {diag['panel_firms']:,}", flush=True)
    print(f"  flag-covered firms (>=1 parsed 10-K): "
          f"{diag['flag_covered_firms']:,}", flush=True)
    print(f"  flag_hdr_changed (10-K hdr STATE changed in-sample): "
          f"{diag['flag1_hdr_changed_count']:,}", flush=True)
    print(f"  flag_sec_disagree (last 10-K hdr != panel snapshot, n_10k>=2): "
          f"{diag['flag2_sec_disagree_count']:,}", flush=True)
    print(f"  UNION relocator firms: {diag['union_reloc_count']:,}",
          flush=True)
    reloc_permnos = set(m.loc[m['reloc'], 'permno'].astype(int))
    return reloc_permnos, diag


def build_stable_dfm():
    """Build the stable-HQ subsample by restricting the cached PIT panel to
    firms with no in-sample relocation (per the v6 Task A definition).
    NOTE: we use the cached PIT panel as a vehicle for the IO-merged firm-
    month data, but we keep STATIC HQ literacy (literacy_score_corrected,
    mom_x_iv_x_literacy_corr) — we are NOT applying the PIT literacy
    correction. Output is cached to DFM_STABLE_CACHE."""
    if os.path.exists(DFM_STABLE_CACHE):
        print(f"=== loading cached stable-HQ panel "
              f"{DFM_STABLE_CACHE} ===", flush=True)
        d = pd.read_parquet(DFM_STABLE_CACHE)
        # also re-build relocator diagnostics from EDGAR for the report
        hq = pd.read_parquet(HQ_EDGAR)
        dfm_pit = pd.read_parquet(DFM_PIT_CACHE)
        panel_hq = (dfm_pit.groupby('permno')
                    .agg(hq_state=('hq_state', 'first'),
                         n_months=('date', 'size'))
                    .reset_index())
        _, diag = build_relocator_set(hq, panel_hq)
        return d, diag

    print("=== building stable-HQ subsample from cached PIT panel ===",
          flush=True)
    dfm = pd.read_parquet(DFM_PIT_CACHE)
    print(f"  full PIT panel: {len(dfm):,} firm-months, "
          f"{dfm['permno'].nunique():,} permnos", flush=True)

    panel_hq = (dfm.groupby('permno')
                .agg(hq_state=('hq_state', 'first'),
                     n_months=('date', 'size'))
                .reset_index())
    hq = pd.read_parquet(HQ_EDGAR)
    print(f"  EDGAR HQ-state pull: {len(hq):,} firms", flush=True)
    reloc_permnos, diag = build_relocator_set(hq, panel_hq)

    n_before = len(dfm)
    nperm_before = dfm['permno'].nunique()
    d_stable = dfm[~dfm['permno'].isin(reloc_permnos)].copy()
    print(f"  stable-HQ panel: {len(d_stable):,} firm-months "
          f"(dropped {n_before - len(d_stable):,} = "
          f"{100.0*(n_before-len(d_stable))/n_before:.1f}%), "
          f"{d_stable['permno'].nunique():,} permnos "
          f"(dropped {nperm_before - d_stable['permno'].nunique():,})",
          flush=True)
    d_stable.to_parquet(DFM_STABLE_CACHE)
    print(f"  cached stable-HQ panel to {DFM_STABLE_CACHE}", flush=True)
    return d_stable, diag


def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d, perm_io, terc


def task1_io_sign_flip(dfm, results):
    """Task 1: 13F IO sign flip test on stable-HQ subsample with STATIC HQ
    literacy. Persistent + time-varying IO."""
    print("\n=== TASK 1: 13F IO sign flip on stable-HQ "
          "(static HQ literacy) ===", flush=True)

    # Estimation sample: drop rows missing STATIC focal or static state
    d_all = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  static-literacy estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos, "
          f"{d_all['hq_state'].nunique()} HQ states", flush=True)

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

        # Per-tercile three-way (static)
        pertile = {}
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            sub = d_io[d_io['io_grp'] == g]
            r = twfe_three_way(sub)
            pertile[g] = {k: r[k] for k in ['coef', 'se', 't', 'se_state',
                                            't_state', 'n_obs', 'se_kind']}
            print(f"    {g}: gamma_static={r['coef']:+.6f} "
                  f"cgm_t={r['t']:.2f} state_t={r['t_state']:.2f} "
                  f"n={r['n_obs']:,}", flush=True)

        # Stacked difference (low - high)
        d_stack = d_io[d_io['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
        d_stack['io_grp_bin'] = np.where(d_stack['io_grp'] == 'IO1_low',
                                         'low', 'high')
        print(f"  running stacked difference (B={B_WCB}) ...", flush=True)
        sd = stacked_io_difference(d_stack, B=B_WCB, seed=42)
        print(f"    DIFF (low-high) = {sd['difference_coef']:+.6f} | "
              f"state-cl SE {sd['se_state_clustered']:.6f} "
              f"t={sd['t_state_clustered']:.2f} | "
              f"CGM t={sd['t_cgm_two_way']} | "
              f"wcb p={sd['wcb_p_value']:.4f}", flush=True)

        out[measure] = {
            "io_measure": label,
            "tercile_means": {k: float(v) for k, v in terc_means.items()},
            "per_tercile_threeway_static": pertile,
            "stacked_difference_static": sd,
            "n_permnos": int(n_p),
            "n_firm_months": int(n_o),
        }

    results['task1_13f_io_sign_flip_stable'] = out
    return d_all, out


def task2_within_retail_gradient(dfm, results):
    """Task 2: within-retail literacy gradient on stable-HQ (static literacy).
    Within low-IO tercile, median-split firms by static HQ state literacy."""
    print("\n=== TASK 2: within-retail literacy gradient on stable-HQ "
          "(static HQ literacy) ===", flush=True)
    d_all = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    out = {}

    for measure, io_col, label in [
        ('persistent', 'io_share_persist', 'persistent IO measure'),
        ('time_varying', 'io_share', 'time-varying IO measure'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io, _, _ = add_io_terciles(d_io, io_col)
        d_lowio = d_io[d_io['io_grp'] == 'IO1_low'].copy()
        print(f"  low-IO (high-retail): {len(d_lowio):,} firm-months, "
              f"{d_lowio['permno'].nunique()} permnos", flush=True)

        # Three-way within low-IO under static
        r_low = twfe_three_way(d_lowio)
        print(f"  three-way within low-IO (static): "
              f"coef={r_low['coef']:+.6f} "
              f"cgm_t={r_low['t']:.2f} state_t={r_low['t_state']:.2f}",
              flush=True)
        wcb_low = wild_cluster_bootstrap_state(d_lowio, B=B_WCB, seed=42)
        print(f"    wcb p={wcb_low['p_value']:.4f}", flush=True)

        # Split by static state literacy: state -> mean
        # literacy_score_corrected within low-IO subsample.
        state_lit = (d_lowio.groupby('hq_state')
                     ['literacy_score_corrected'].mean())
        lit_median = state_lit.median()
        hi_lit_states = set(state_lit[state_lit >= lit_median].index)
        lo_lit_states = set(state_lit[state_lit < lit_median].index)
        d_hilit = d_lowio[d_lowio['hq_state'].isin(hi_lit_states)].copy()
        d_lolit = d_lowio[d_lowio['hq_state'].isin(lo_lit_states)].copy()
        r_hl = twfe_three_way(d_hilit)
        r_ll = twfe_three_way(d_lolit)
        print(f"    low-IO + high-lit-static: {r_hl['coef']:+.6f} "
              f"(state_t={r_hl['t_state']:.2f}, n={r_hl['n_obs']:,})",
              flush=True)
        print(f"    low-IO + low-lit-static:  {r_ll['coef']:+.6f} "
              f"(state_t={r_ll['t_state']:.2f}, n={r_ll['n_obs']:,})",
              flush=True)

        # Stacked difference (lowlit - hilit) within low-IO
        d_stack = d_lowio.copy()
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
            "threeway_within_lowIO_static": {
                **{k: r_low[k] for k in ['coef', 'se', 't', 'se_state',
                                         't_state', 'n_obs', 'se_kind']},
                "wild_cluster_bootstrap": wcb_low,
            },
            "lowIO_by_static_state_literacy": {
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

    results['task2_within_retail_gradient_stable'] = out
    return out


def write_tables(results):
    """LaTeX tables for the stable-HQ centerpiece tests."""
    fname = os.path.join(TABDIR, "tab_13f_stacked_diff_stable_hq.tex")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma$ & CGM SE & "
                "state-cl. SE & $N$ \\\\\n")
        for mlabel, mdisp in [
            ('persistent',
             'Panel A: persistent IO + static HQ literacy, stable-HQ firms'),
            ('time_varying',
             'Panel B: time-varying IO + static HQ literacy, stable-HQ firms')
        ]:
            b = results['task1_13f_io_sign_flip_stable'][mlabel]
            pt = b['per_tercile_threeway_static']
            sd = b['stacked_difference_static']
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            for g, disp in [('IO1_low', 'Low IO (high retail)'),
                            ('IO2_mid', 'Mid IO'),
                            ('IO3_high', 'High IO (institutional)')]:
                r = pt[g]
                f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                        f"${r['se_state']:.4f}$ & ${r['n_obs']:,}$ \\\\\n")
            cgm_se_str = (f"{sd['se_cgm_two_way']:.4f}"
                          if sd['se_cgm_two_way'] is not None
                          else "n.p.s.d.")
            f.write(f"Difference (low $-$ high), stacked & "
                    f"${sd['difference_coef']:.4f}$ & "
                    f"${cgm_se_str}$ & "
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

    fname = os.path.join(TABDIR, "tab_within_retail_literacy_stable_hq.tex")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Subsample & three-way $\\hat\\gamma$ & CGM $t$ & "
                "state-cl. $t$ & $N$ \\\\\n")
        for mlabel, mdisp in [
            ('persistent',
             'Panel A: low-IO tercile (persistent IO), stable-HQ firms'),
            ('time_varying',
             'Panel B: low-IO tercile (time-varying IO), stable-HQ firms')
        ]:
            dd = results['task2_within_retail_gradient_stable'][mlabel]
            f.write("\\hline\n\\multicolumn{5}{l}{\\textit{" + mdisp
                    + "}} \\\\\n")
            tw = dd['threeway_within_lowIO_static']
            f.write(f"Within low-IO (high-retail) & ${tw['coef']:.4f}$ & "
                    f"${tw['t']:.2f}$ & ${tw['t_state']:.2f}$ & "
                    f"${tw['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{"
                    f"${tw['wild_cluster_bootstrap']['p_value']:.3f}$}}"
                    f" \\\\\n")
            hl = dd['lowIO_by_static_state_literacy']['high_literacy_states']
            ll = dd['lowIO_by_static_state_literacy']['low_literacy_states']
            sdl = dd['lowIO_by_static_state_literacy'][
                'stacked_lowlit_minus_highlit']
            f.write(f"\\quad low-IO $\\times$ high-lit states & "
                    f"${hl['coef']:.4f}$ & ${hl['t']:.2f}$ & "
                    f"${hl['t_state']:.2f}$ & ${hl['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad low-IO $\\times$ low-lit states & "
                    f"${ll['coef']:.4f}$ & ${ll['t']:.2f}$ & "
                    f"${ll['t_state']:.2f}$ & ${ll['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad difference (lowlit $-$ hilit), stacked & "
                    f"${sdl['difference_coef']:.4f}$ & --- & "
                    f"${sdl['t_state_clustered']:.2f}$ & "
                    f"${sdl['n_obs']:,}$ \\\\\n")
            f.write(f"\\quad\\quad wild-cluster bootstrap $p$ & "
                    f"\\multicolumn{{4}}{{l}}{{${sdl['wcb_p_value']:.3f}$}}"
                    f" \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")
    print(f"  wrote {fname}", flush=True)


def main():
    dfm, reloc_diag = build_stable_dfm()

    results = {
        "task": "v7 STABLE-HQ centerpiece test: re-fire the 13F IO sign "
                "flip (Task 1) and the within-retail literacy gradient "
                "(Task 2) on the stable-HQ subsample (firms with no "
                "in-sample HQ relocation per the v6 Task A definition), "
                "keeping STATIC HQ literacy (the seed's convention). "
                "Theoretical motivation: the local-bias channel operates "
                "through community ties that take time to form; for firms "
                "that relocate HQ during the sample, the effective "
                "local-investor base is ambiguous by construction. The "
                "stable-HQ subsample is the clean test of the seed's "
                "channel.",
        "sample_definition": (
            "stable-HQ = NOT relocator, where reloc = flag_hdr_changed OR "
            "flag_sec_disagree using the v6 Task A EDGAR-based flag (same "
            "1,366-firm definition that produced the v6 relocation-free "
            "panel headline gamma=-0.0145, wcb p=0.122)."
        ),
        "relocation_flag_diagnostics": reloc_diag,
        "reference_v6_full_panel_static": {
            "headline_gamma": -0.0117,
            "headline_wcb_p": 0.078,
            "io_persistent_DIFF_low_minus_high": -0.0211,
            "io_persistent_wcb_p": 0.144,
            "io_time_varying_DIFF_low_minus_high": -0.0181,
            "io_time_varying_wcb_p": 0.264,
            "within_retail_lowlit_minus_hilit_DIFF": 0.0022,
            "within_retail_lowlit_minus_hilit_wcb_p_approx": 0.07,
            "note": "v6 numbers come from the FULL panel (no stable-HQ "
                    "restriction) with STATIC HQ literacy. The v6 "
                    "relocation-free drop-test gave gamma=-0.0145, "
                    "wcb p=0.122 on the aggregate three-way."
        },
        "reference_v7_pit_full_panel": {
            "headline_pit_gamma": -0.0003,
            "headline_pit_wcb_p": 0.517,
            "io_persistent_DIFF_pit": 0.00029,
            "io_persistent_pit_wcb_p": 0.714,
            "note": "PIT-HQ literacy correction on the FULL panel collapses "
                    "the v6 aggregate and the IO sign flip. The operator's "
                    "decision is to ask whether the v6 centerpiece survives "
                    "on the THEORETICALLY CORRECT stable-HQ subsample under "
                    "STATIC literacy (this script)."
        },
    }

    d_all, _ = task1_io_sign_flip(dfm, results)
    _ = task2_within_retail_gradient(dfm, results)

    results['meta'] = {
        "stable_hq_panel_firm_months": int(len(dfm)),
        "stable_hq_panel_firms": int(dfm['permno'].nunique()),
        "stable_hq_static_estimation_sample": int(len(d_all)),
        "B_wcb": B_WCB,
        "lag_months": LAG_MONTHS,
        "seed": 42,
        "inputs": {
            "hq_edgar": HQ_EDGAR,
            "dfm_pit_cache": DFM_PIT_CACHE,
            "dfm_stable_cache": DFM_STABLE_CACHE,
        },
    }

    # Directional-survival verdict
    p_static = results['task1_13f_io_sign_flip_stable']['persistent']
    sd_p = p_static['stacked_difference_static']
    pt_p = p_static['per_tercile_threeway_static']
    diff_p = sd_p['difference_coef']
    wcb_p_p = sd_p['wcb_p_value']
    low_p = pt_p['IO1_low']['coef']
    high_p = pt_p['IO3_high']['coef']

    tv_static = results['task1_13f_io_sign_flip_stable']['time_varying']
    sd_t = tv_static['stacked_difference_static']
    pt_t = tv_static['per_tercile_threeway_static']
    diff_t = sd_t['difference_coef']
    wcb_p_t = sd_t['wcb_p_value']
    low_t = pt_t['IO1_low']['coef']
    high_t = pt_t['IO3_high']['coef']

    # Survival means low-IO < high-IO (low-IO more negative).
    surv_p = bool(low_p < high_p and diff_p < 0)
    surv_t = bool(low_t < high_t and diff_t < 0)

    if surv_p and surv_t:
        verdict = "YES"
    elif surv_p or surv_t:
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    results['directional_survival_verdict'] = {
        "verdict": verdict,
        "rule": ("YES = both persistent and time-varying IO show low-IO < "
                 "high-IO with DIFF<0 (low-IO more negative). PARTIAL = "
                 "one of the two satisfies. NO = neither does (sign flips "
                 "or magnitude vanishes)."),
        "persistent_low_minus_high": diff_p,
        "persistent_wcb_p": wcb_p_p,
        "persistent_low_IO_gamma": low_p,
        "persistent_high_IO_gamma": high_p,
        "persistent_survives_direction": surv_p,
        "time_varying_low_minus_high": diff_t,
        "time_varying_wcb_p": wcb_p_t,
        "time_varying_low_IO_gamma": low_t,
        "time_varying_high_IO_gamma": high_t,
        "time_varying_survives_direction": surv_t,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)

    write_tables(results)
    print(f"\n=== DIRECTIONAL SURVIVAL VERDICT: {verdict} ===", flush=True)
    return results


if __name__ == '__main__':
    main()
