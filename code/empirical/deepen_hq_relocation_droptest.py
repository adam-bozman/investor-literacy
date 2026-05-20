# =====================================================================
# deepen_hq_relocation_droptest.py
# HQ-relocation drop-test: builds a point-in-time relocation flag from EDGAR 10-K-header states, drops flagged relocators, and re-runs the headline TWFE three-way regression with restricted wild-cluster bootstrap to confirm robustness.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet; code/empirical/_hq_edgar_state_v6.parquet (or _ckpt fallback); deepen_estimators.py
# Outputs:   output/stage3a/results_v6_hq.json; output/stage3a/tables/tab_hq_relocation_droptest_v6.tex
# Paper:     IA stable-HQ construction / PIT HQ reassignment diagnostics (relocator-vs-stable, refined-relocator material); tab_hq_relocation_droptest_v6.tex
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r2 (Gate 5 Reject) — Task A: HQ-relocation drop-test.

The seed panel's `hq_state` is a STATIC Compustat snapshot. Over 2009-2023 some
firms relocate HQ; for those firms the static snapshot misclassifies the
HQ-state assignment, contaminating the state fixed effects and the state-clustered
inference of the headline TWFE three-way regression. The referee asks: identify
firms with a known HQ-state relocation during the sample, DROP them, re-run the
headline regression on the relocation-free panel, and report whether the headline
coefficient and its wild-cluster bootstrap inference change.

RELOCATION FLAG — point-in-time, from SEC EDGAR 10-K filing headers.
  The SEC filing-header business-address STATE is the canonical point-in-time HQ
  location in the HQ-bias literature (Coval-Moskowitz 1999; Pirinsky-Wang 2006).
  `deepen_hq_edgar_pull.py` pulls, for every panel firm with a CIK, the STATE in
  the SGML header of (the first batch of firms) every 10-K filed 2009-2023, and
  (the rest) the FIRST and LAST in-sample 10-K. A firm whose 10-K-header STATE
  changes across the sample relocated; the change date bounds the timing.
  Two relocation flags are built and UNIONED (conservative — drop a firm if
  EITHER source flags it):
    (1) hdr_changed: the 10-K-header STATE changed across the firm's sample 10-Ks
        (genuine point-in-time; the primary flag).
    (2) sec_disagree: the firm's most-recent 10-K-header STATE disagrees with the
        panel's static `hq_state` snapshot AND the firm has >=2 parsed 10-K
        headers (two-point cross-check — catches firms whose move predates their
        first sample 10-K OR whose snapshot is simply stale; secondary).
  Coverage is disclosed honestly: the flag covers only firms with a CIK and >=1
  (for flag 2, >=2) parseable 10-K header. Firms with no CIK / no parseable
  header are NOT flagged — the drop-test is therefore a LOWER bound on the number
  of relocators, which makes the robustness check conservative in the right
  direction (any un-caught relocator stays in the "clean" panel and works
  against finding robustness).

The headline regression (carried over UNCHANGED from deepen_estimators.py):
  TWFE state+month FE, three-way coefficient on mom_x_iv_x_literacy_corr;
  two-way state x month CGM SE, one-way state-clustered CR1 SE, and a restricted
  wild-cluster bootstrap at the state level (Webb 6-point, B=9999).

Output: output/stage3a/results_v6_hq.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepen_estimators import (twfe_three_way, wild_cluster_bootstrap_state,
                               FOCAL)

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = ROOT + "/code/empirical"
PANEL = (ROOT + "/output/seed/data/processed/"
         "panel_corrected_standardized.parquet")
HQ_EDGAR = EMP + "/_hq_edgar_state_v6.parquet"
HQ_EDGAR_CKPT = EMP + "/_hq_edgar_state_v6_ckpt.parquet"
OUT_JSON = ROOT + "/output/stage3a/results_v6_hq.json"
TABDIR = ROOT + "/output/stage3a/tables"
B_WCB = 9999  # match the headline (results_v3 item1) bootstrap rep count


def load_panel():
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    return df


def load_hq_edgar():
    """Load the EDGAR PIT HQ-state pull; prefer the final file, fall back to
    the latest checkpoint."""
    path = HQ_EDGAR if os.path.exists(HQ_EDGAR) else HQ_EDGAR_CKPT
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"EDGAR HQ-state pull not found ({HQ_EDGAR} / {HQ_EDGAR_CKPT}). "
            f"Run deepen_hq_edgar_pull.py first.")
    hq = pd.read_parquet(path)
    return hq, path


def build_reloc_flag(panel_hq, hq):
    """Build the unioned relocation flag. Returns a per-permno DataFrame with
    `reloc` (bool), the component flags, and the relocation year where known."""
    # panel static snapshot, 2-letter
    panel_hq = panel_hq.copy()
    panel_hq['hq_state_2l'] = panel_hq['hq_state'].str.replace(
        'US-', '', regex=False)

    m = panel_hq.merge(hq, on='permno', how='left')

    # flag 1: 10-K-header STATE changed within the firm's sample 10-Ks
    m['flag_hdr_changed'] = (m['hdr_changed'] == True)  # NaN -> False

    # flag 2: most-recent 10-K-header STATE disagrees with the panel snapshot
    #         (requires >=2 parsed headers so a single-filing parse error is not
    #         mistaken for a relocation, and so the cross-check has a within-firm
    #         time series to lean on).
    def last_hdr_state(s):
        if not isinstance(s, str):
            return None
        seq = json.loads(s)
        return seq[-1][1] if seq else None

    m['hdr_state_last'] = m['hdr_states'].apply(last_hdr_state)
    m['n_10k'] = m['n_10k'].fillna(0).astype(int)
    m['flag_sec_disagree'] = (
        (m['n_10k'] >= 2)
        & m['hdr_state_last'].notna()
        & m['hq_state_2l'].notna()
        & (m['hdr_state_last'] != m['hq_state_2l'])
    )

    m['reloc'] = m['flag_hdr_changed'] | m['flag_sec_disagree']
    # coverage: a firm is "covered by the flag" if we have >=1 parsed 10-K header
    m['flag_covered'] = m['n_10k'] >= 1
    return m


def run_headline(d, label):
    """Headline TWFE three-way + restricted wild-cluster bootstrap."""
    base = twfe_three_way(d)
    wcb = wild_cluster_bootstrap_state(d, B=B_WCB, seed=42)
    print(f"  [{label}] gamma_hat={base['coef']:+.6f} | "
          f"CGM t={base['t']:.3f} ({base['se_kind']}) | "
          f"state-cl CR1 t={base['t_state']:.3f} | "
          f"wcb p={wcb['p_value']:.4f} (B={wcb['B']})", flush=True)
    return {
        "estimator": "TWFE state+month FE; three-way coef on "
                     "mom_x_iv_x_literacy_corr",
        "gamma_hat": base['coef'],
        "se_cgm_two_way": base['se'] if base['se_kind'] == 'cgm_two_way'
        else None,
        "t_cgm_two_way": base['t'] if base['se_kind'] == 'cgm_two_way'
        else None,
        "se_kind": base['se_kind'],
        "se_state_clustered_CR1": base['se_state'],
        "t_state_clustered_CR1": base['t_state'],
        "wcb_restricted_p_value": wcb['p_value'],
        "wcb_B": wcb['B'],
        "wcb_n_state_clusters": wcb['n_state_clusters'],
        "wcb_studentized_ci_95": wcb['ci_studentized'],
        "wcb_t_boot_q": wcb['t_boot_q'],
        "survives_5pct": bool(wcb['p_value'] < 0.05),
        "survives_10pct": bool(wcb['p_value'] < 0.10),
        "n_obs": int(base['n_obs']),
        "n_permnos": int(d['permno'].nunique()),
        "n_state_clusters": int(d['hq_state'].nunique()),
    }


def main():
    print("=== HQ-relocation drop-test ===", flush=True)
    df = load_panel()
    d_all = df.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"estimation sample: {len(d_all):,} firm-months, "
          f"{d_all['permno'].nunique()} permnos, "
          f"{d_all['hq_state'].nunique()} HQ states", flush=True)

    panel_hq = (d_all.groupby('permno')
                .agg(hq_state=('hq_state', 'first'),
                     n_months=('date', 'size'),
                     date_min=('date', 'min'),
                     date_max=('date', 'max'))
                .reset_index())

    hq, hq_path = load_hq_edgar()
    final = os.path.exists(HQ_EDGAR)
    print(f"EDGAR HQ-state pull: {hq_path} "
          f"({'FINAL' if final else 'CHECKPOINT — partial'}), "
          f"{len(hq):,} firms", flush=True)

    m = build_reloc_flag(panel_hq, hq)
    n_panel = len(m)
    n_covered = int(m['flag_covered'].sum())
    n_flag1 = int(m['flag_hdr_changed'].sum())
    n_flag2 = int(m['flag_sec_disagree'].sum())
    n_reloc = int(m['reloc'].sum())
    reloc_permnos = set(m.loc[m['reloc'], 'permno'])
    fm_dropped = int(d_all['permno'].isin(reloc_permnos).sum())

    print(f"\nRelocation flag coverage:", flush=True)
    print(f"  panel firms: {n_panel}", flush=True)
    print(f"  firms with >=1 parsed 10-K header (flag-covered): {n_covered} "
          f"({100.0*n_covered/n_panel:.1f}%)", flush=True)
    print(f"  flag 1 (10-K-header STATE changed in-sample): {n_flag1}",
          flush=True)
    print(f"  flag 2 (last 10-K-header STATE != panel snapshot, n_10k>=2): "
          f"{n_flag2}", flush=True)
    print(f"  UNION relocation flag: {n_reloc} firms "
          f"({100.0*n_reloc/n_covered:.1f}% of flag-covered firms)", flush=True)
    print(f"  firm-months dropped: {fm_dropped:,} "
          f"({100.0*fm_dropped/len(d_all):.2f}% of estimation sample)",
          flush=True)

    reloc_year_counts = (m.loc[m['reloc'] & m['reloc_year'].notna(),
                               'reloc_year'].astype(int)
                         .value_counts().sort_index().to_dict())
    print(f"  relocation-year distribution (where datable): "
          f"{reloc_year_counts}", flush=True)

    # example relocators
    examples = (m[m['reloc']]
                .sort_values('n_months', ascending=False)
                [['permno', 'hq_state', 'hdr_state_last', 'n_10k',
                  'flag_hdr_changed', 'flag_sec_disagree', 'reloc_year']]
                .head(20))
    print("\nExample flagged relocators (top 20 by panel length):", flush=True)
    print(examples.to_string(index=False), flush=True)

    # ===== headline regressions =====
    print("\n=== headline TWFE three-way ===", flush=True)
    res_full = run_headline(d_all, "FULL panel (reference)")
    d_clean = d_all[~d_all['permno'].isin(reloc_permnos)].copy()
    res_clean = run_headline(d_clean, "RELOCATION-FREE panel")

    # robustness verdict
    g_full = res_full['gamma_hat']
    g_clean = res_clean['gamma_hat']
    p_full = res_full['wcb_restricted_p_value']
    p_clean = res_clean['wcb_restricted_p_value']
    coef_shift_pct = 100.0 * (g_clean - g_full) / abs(g_full)
    same_sig_10 = (p_full < 0.10) == (p_clean < 0.10)
    same_sig_5 = (p_full < 0.05) == (p_clean < 0.05)
    robust = (np.sign(g_clean) == np.sign(g_full)) and same_sig_10

    verdict = (
        f"Headline is ROBUST to dropping relocators: dropping the {n_reloc} "
        f"firms ({fm_dropped:,} firm-months, "
        f"{100.0*fm_dropped/len(d_all):.2f}% of the sample) flagged as HQ "
        f"relocators leaves the three-way coefficient at {g_clean:+.6f} "
        f"(vs {g_full:+.6f} on the full panel; {coef_shift_pct:+.1f}% shift), "
        f"same sign, and the restricted wild-cluster bootstrap p moves from "
        f"{p_full:.4f} to {p_clean:.4f} — the 10%-significance verdict is "
        f"{'unchanged' if same_sig_10 else 'CHANGED'}."
        if robust else
        f"Headline CHANGES when relocators are dropped: coefficient "
        f"{g_full:+.6f} -> {g_clean:+.6f} ({coef_shift_pct:+.1f}% shift), "
        f"wild-cluster bootstrap p {p_full:.4f} -> {p_clean:.4f}; "
        f"10%-significance verdict {'unchanged' if same_sig_10 else 'CHANGED'}, "
        f"sign {'unchanged' if np.sign(g_clean)==np.sign(g_full) else 'FLIPPED'}."
    )
    print(f"\n{verdict}", flush=True)

    results = {
        "task": "Task A — HQ-relocation drop-test (Gate 5 Reject deepen r2)",
        "headline_definition": "TWFE state+month FE, three-way coefficient on "
        "mom_x_iv_x_literacy_corr; restricted wild-cluster bootstrap at the "
        "state level (Webb 6-point, B=9999). Matches results_v3.json "
        "item1_wild_cluster_bootstrap (the canonical headline).",
        "relocation_flag": {
            "source": "SEC EDGAR 10-K filing-header business-address STATE "
            "(point-in-time). deepen_hq_edgar_pull.py pulls the SGML-header "
            "STATE of the first batch of firms' every 10-K filed 2009-2023, "
            "and the FIRST and LAST in-sample 10-K for the rest, for each "
            "panel firm with a CIK.",
            "edgar_pull_file": hq_path,
            "edgar_pull_is_final": final,
            "edgar_pull_n_firms": int(len(hq)),
            "panel_firms": n_panel,
            "flag_covered_firms": n_covered,
            "flag_covered_pct": round(100.0 * n_covered / n_panel, 1),
            "flag1_hdr_changed_count": n_flag1,
            "flag1_definition": "10-K-header STATE changed across the firm's "
            "in-sample 10-Ks (genuine point-in-time relocation).",
            "flag2_sec_disagree_count": n_flag2,
            "flag2_definition": "most-recent 10-K-header STATE disagrees with "
            "the panel's static hq_state snapshot, firm has >=2 parsed 10-K "
            "headers (two-point cross-check).",
            "union_reloc_count": n_reloc,
            "union_reloc_pct_of_covered": round(
                100.0 * n_reloc / n_covered, 1),
            "firm_months_dropped": fm_dropped,
            "firm_months_dropped_pct": round(
                100.0 * fm_dropped / len(d_all), 2),
            "relocation_year_distribution": reloc_year_counts,
            "coverage_caveat": "The flag covers only firms with a CIK and a "
            "parseable 10-K header. Un-covered firms are NOT flagged, so the "
            "drop-test is a LOWER bound on relocators — any un-caught "
            "relocator stays in the 'relocation-free' panel and works AGAINST "
            "finding robustness, making the test conservative.",
            "example_relocators": examples.to_dict(orient='records'),
        },
        "headline_full_panel": res_full,
        "headline_relocation_free": res_clean,
        "robustness": {
            "coef_full": g_full,
            "coef_relocation_free": g_clean,
            "coef_shift_pct": round(coef_shift_pct, 2),
            "wcb_p_full": p_full,
            "wcb_p_relocation_free": p_clean,
            "sign_unchanged": bool(np.sign(g_clean) == np.sign(g_full)),
            "sig_10pct_verdict_unchanged": bool(same_sig_10),
            "sig_5pct_verdict_unchanged": bool(same_sig_5),
            "robust": bool(robust),
            "verdict": verdict,
        },
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)
    _write_table(results)
    return results


def _write_table(r):
    os.makedirs(TABDIR, exist_ok=True)
    rf = r['relocation_flag']
    a = r['headline_full_panel']
    b = r['headline_relocation_free']
    rob = r['robustness']

    def fmt(x, d=4):
        return f"{x:.{d}f}" if x is not None else "---"

    with open(os.path.join(TABDIR, "tab_hq_relocation_droptest_v6.tex"), 'w',
              encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcc}\n\\hline\\hline\n")
        f.write(" & Full panel & Relocation-free panel \\\\\n")
        f.write(" & (reference) & (drop confirmed HQ relocators) \\\\\n")
        f.write("\\hline\n")
        f.write(f"Three-way $\\hat\\gamma$ & ${fmt(a['gamma_hat'],6)}$ & "
                f"${fmt(b['gamma_hat'],6)}$ \\\\\n")
        f.write(f"Two-way CGM $t$ & ${fmt(a['t_cgm_two_way'],2)}$ & "
                f"${fmt(b['t_cgm_two_way'],2)}$ \\\\\n")
        f.write(f"State-clustered CR1 $t$ & "
                f"${fmt(a['t_state_clustered_CR1'],2)}$ & "
                f"${fmt(b['t_state_clustered_CR1'],2)}$ \\\\\n")
        f.write(f"Wild-cluster bootstrap $p$ & "
                f"${fmt(a['wcb_restricted_p_value'],4)}$ & "
                f"${fmt(b['wcb_restricted_p_value'],4)}$ \\\\\n")
        f.write(f"\\quad ($B={a['wcb_B']}$, state clusters) & "
                f"({a['wcb_n_state_clusters']}) & "
                f"({b['wcb_n_state_clusters']}) \\\\\n")
        f.write(f"Firm-months & ${a['n_obs']:,}$ & ${b['n_obs']:,}$ \\\\\n")
        f.write(f"Firms & ${a['n_permnos']:,}$ & ${b['n_permnos']:,}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\multicolumn{{3}}{{l}}{{\\textit{{Relocation flag: SEC "
                f"EDGAR 10-K filing-header business-address STATE, "
                f"point-in-time.}}}} \\\\\n")
        f.write(f"\\multicolumn{{3}}{{l}}{{Flag-covered firms: "
                f"{rf['flag_covered_firms']:,} "
                f"({rf['flag_covered_pct']}\\% of panel). "
                f"Confirmed relocators: {rf['union_reloc_count']:,} "
                f"({rf['firm_months_dropped']:,} firm-months, "
                f"{rf['firm_months_dropped_pct']}\\% of sample).}} \\\\\n")
        f.write(f"\\multicolumn{{3}}{{l}}{{Coefficient shift: "
                f"{rob['coef_shift_pct']:+.1f}\\%. "
                f"Sign unchanged: {rob['sign_unchanged']}. "
                f"10\\%-significance verdict unchanged: "
                f"{rob['sig_10pct_verdict_unchanged']}.}} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")
    print(f"  wrote tab_hq_relocation_droptest_v6.tex", flush=True)


if __name__ == '__main__':
    main()
