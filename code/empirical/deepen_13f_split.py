# =====================================================================
# deepen_13f_split.py
# Runs the round-0 13F retail/institutional-ownership split: three-way TWFE within EDGAR-13F IO terciles (Split A) and within firm market-equity terciles (Split B), with the high-retail-minus-high-institutional difference.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet; code/empirical/_edgar_13f_*_ticker.parquet (round-0 EDGAR 13F aggregates)
# Outputs:   output/stage3a/deepen_13f_split.json; deepen_13f_split.md; output/stage3a/tables/tab_13f_split.tex
# Paper:     SUPERSEDED by deepen_13f_split_v6.py — kept for provenance (round-0 EDGAR-proxy split; feeds the IA mid-IO / EDGAR-proxy cross-check material)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive Item 6 (part 2/2) — 13F retail/institutional ownership split.

Both referees name this as the single most important deepening move: the
mechanism (sophistication-modulated LOCAL RETAIL demand) predicts the three-way
coefficient is PRESENT in high-retail-ownership / low-institutional-ownership
stocks and ABSENT in institutionally-held stocks. The within-stock contrast has
a built-in placebo.

Data route (deepen directive item 6, WRDS s34 blocked):
  - OSAP has no clean raw institutional-ownership-SHARE signal (its IO signals
    are conditional/residualized constructs: RIO_*, Activism*; DelBreadth is a
    flow; IO_ShortInterest covers only 1,737 high-short-interest NYSE permnos).
  - We built a PERSISTENT institutional-ownership measure from SEC EDGAR 13F-HR
    filings — see deepen_13f_download.py. The download targeted three quarters
    (2011Q2, 2015Q2, 2019Q2); 2011Q2 and 2015Q2 completed fully (10,333 and
    10,804 unique tickers), 2019Q2 was killed by a runtime cap at ~1,000/5,513
    filings. Two quarters spanning 2011 and 2015 still give a defensible
    persistent IO measure (institutional ownership is highly persistent), so we
    proceed with the completed quarters rather than restart the long pull.
  - Institutional ownership share per permno-quarter = (sum of institutional
    shares held, from 13F infotables, Shares-type rows, options excluded) /
    (shares outstanding, from the panel). Matched to the panel by ticker. The
    multi-quarter average is the time-invariant split variable.

Two splits are run:
  (A) EDGAR-13F institutional-ownership terciles — the mechanism-discriminating
      test. Three-way TWFE within IO tercile 1 (high-retail / low-institutional)
      vs IO tercile 3 (high-institutional). Difference + SE.
  (B) Size proxy (companion robustness, computable from the panel alone): firm
      market-equity terciles. Institutional ownership is mechanically strongly
      increasing in size, and Ivkovic-Weisbenner (2005) place the largest
      local-retail-bias effects in small / non-S&P-500 stocks. Small-cap = the
      high-retail proxy; large-cap = the high-institutional proxy.

Output: output/stage3a/deepen_13f_split.{json,md}
      + output/stage3a/tables/tab_13f_split.tex
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "deepen_13f_split.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "deepen_13f_split.md")
TABLE = os.path.join(ROOT, "output", "stage3a", "tables", "tab_13f_split.tex")

FOCAL = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
         'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']

# quarter label -> a representative panel month for shrout
Q_TO_MONTH = {"2011Q2": "2011-06", "2015Q2": "2015-06", "2019Q2": "2019-06"}


def twfe_three_way_clustered(d):
    """TWFE state+month FE, two-way state x month CGM clustering; one-way
    state-clustered CR1 fallback if the CGM variance is not PSD. Returns dict
    with coef, se, t, n_obs, se_kind, var (se^2)."""
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    x3 = focal[:, 6]
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(focal[:, :6])
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, W6s, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    x3t = x3 - W @ np.linalg.solve(WtW, W.T @ x3)
    yt = y - W @ np.linalg.solve(WtW, W.T @ y)
    Sxx = x3t @ x3t
    coef = float((x3t @ yt) / Sxx)
    e = yt - coef * x3t
    score = x3t * e

    def cl_sum_sq(g):
        ug = np.unique(g)
        s = np.zeros(len(ug))
        np.add.at(s, np.searchsorted(ug, g), score)
        return (s ** 2).sum()

    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    meat_cgm = cl_sum_sq(sc) + cl_sum_sq(mc) - cl_sum_sq(inter)
    G = len(np.unique(sc))
    if meat_cgm > 0:
        se = float(np.sqrt(meat_cgm) / Sxx)
        se_kind = "cgm_two_way"
    else:
        kp = W.shape[1] + 1
        c = (G / (G - 1.0)) * ((n - 1.0) / (n - kp))
        se = float(np.sqrt(c * cl_sum_sq(sc)) / Sxx)
        se_kind = "state_clustered_CR1_fallback"
    return {"coef": coef, "se": se, "t": coef / se if se > 0 else np.nan,
            "n_obs": int(n), "se_kind": se_kind, "var": se ** 2}


def run_split(d, group_col, group_labels, split_name):
    """Three-way TWFE within each group; report group estimates and the
    high-retail-minus-high-institutional difference with its SE (groups are
    disjoint firm sets, so the difference variance = sum of group variances)."""
    out = {}
    for lab in group_labels:
        sub = d[d[group_col] == lab]
        r = twfe_three_way_clustered(sub)
        out[lab] = r
        print(f"  [{split_name}] {lab}: coef={r['coef']:.6f} se={r['se']:.6f} "
              f"t={r['t']:.3f} n={r['n_obs']:,} [{r['se_kind']}]")
    hr, hi = group_labels[0], group_labels[-1]
    diff = out[hr]['coef'] - out[hi]['coef']
    diff_se = np.sqrt(out[hr]['var'] + out[hi]['var'])
    diff_t = diff / diff_se if diff_se > 0 else np.nan
    out['_difference'] = {
        "high_retail_group": hr, "high_institutional_group": hi,
        "diff_coef": float(diff), "diff_se": float(diff_se),
        "diff_t": float(diff_t),
        "mechanism_prediction": "diff < 0 — three-way more negative in the "
                                "high-retail group than the high-institutional "
                                "group",
        "mechanism_supported": bool(diff < 0 and abs(diff_t) > 1.64),
    }
    print(f"  [{split_name}] DIFF (high-retail minus high-institutional): "
          f"{diff:+.6f} (se {diff_se:.6f}, t {diff_t:.3f})")
    return out


def main():
    print("=== loading panel ===")
    df = pd.read_parquet(PANEL)
    d = df.dropna(subset=FOCAL + ['ret', 'hq_state', 'date']).copy()
    d['ym'] = d['date'].dt.to_period('M').astype(str)
    print(f"panel: {len(d):,} firm-months, {d['permno'].nunique()} permnos")

    results = {"splits": {}}

    # ===== SPLIT B (companion): firm-size terciles ==========================
    print("\n=== SPLIT B: firm-size terciles (high-retail proxy = small-cap) ===")
    permno_me = d.groupby('permno')['me'].mean()
    me_terciles = pd.qcut(permno_me, 3, labels=['T1_small', 'T2_mid', 'T3_large'])
    d['size_grp'] = d['permno'].map(me_terciles).astype(str)
    size_out = run_split(d, 'size_grp',
                         ['T1_small', 'T2_mid', 'T3_large'], "size")
    results['splits']['size_proxy'] = size_out
    results['size_split_note'] = (
        "Companion robustness split, computable from the panel alone. Tercile "
        "assignment from each permno's time-mean market equity. Small-cap = the "
        "high-retail / low-institutional proxy (institutional ownership rises "
        "mechanically with size; Ivkovic-Weisbenner 2005 place the largest "
        "local-retail-bias effects in small / non-S&P-500 stocks). Caveat: firm "
        "size is a coarse proxy and also correlates with IV, liquidity, and "
        "arbitrage costs."
    )

    # ===== SPLIT A: EDGAR-13F institutional-ownership terciles ==============
    # Assemble from whatever per-quarter ticker files completed (2019Q2 download
    # was killed by a runtime cap; 2011Q2 and 2015Q2 completed fully).
    qfiles = sorted(glob.glob(os.path.join(
        ROOT, "code", "empirical", "_edgar_13f_*_ticker.parquet")))
    if not qfiles:
        print("\n=== SPLIT A: no EDGAR-13F per-quarter files found ===")
        results['splits']['edgar_13f'] = None
        results['edgar_13f_status'] = (
            "INFEASIBLE THIS SESSION — EDGAR 13F download produced no "
            "per-quarter file. Item 6 becomes a disclosed limitation."
        )
    else:
        print("\n=== SPLIT A: EDGAR-13F institutional-ownership terciles ===")
        parts = []
        for qf in qfiles:
            qlab = os.path.basename(qf).replace("_edgar_13f_", "").replace(
                "_ticker.parquet", "")
            part = pd.read_parquet(qf)
            if 'quarter' not in part.columns:
                part['quarter'] = qlab
            parts.append(part)
            print(f"  loaded {qlab}: {len(part):,} ticker rows")
        io = pd.concat(parts, ignore_index=True)
        quarters_used = sorted(io['quarter'].unique())
        print(f"13F data: {len(io):,} ticker-quarter rows, "
              f"quarters {quarters_used}")

        io_shares = []
        for q in quarters_used:
            mlabel = Q_TO_MONTH.get(q)
            if mlabel is None:
                continue
            iq = io[io['quarter'] == q]
            pq = d[d['ym'] == mlabel][['permno', 'ticker', 'shrout']
                                      ].drop_duplicates('permno')
            mrg = pq.merge(iq[['Ticker', 'inst_shares', 'n_managers']],
                           left_on='ticker', right_on='Ticker', how='inner')
            # shrout is in thousands of shares; inst_shares is raw shares
            mrg['io_share'] = mrg['inst_shares'] / (mrg['shrout'] * 1000.0)
            # 13F can slightly exceed shrout (reporting lags / share classes);
            # cap at 1.0
            mrg['io_share'] = mrg['io_share'].clip(upper=1.0)
            mrg['quarter'] = q
            io_shares.append(mrg[['permno', 'io_share', 'n_managers',
                                  'quarter']])
            print(f"  {q}: {len(mrg):,} permnos matched, "
                  f"median io_share = {mrg['io_share'].median():.3f}")

        if not io_shares:
            results['splits']['edgar_13f'] = None
            results['edgar_13f_status'] = "13F files present but empty after merge"
        else:
            io_all = pd.concat(io_shares, ignore_index=True)
            io_perm = io_all.groupby('permno').agg(
                io_share=('io_share', 'mean'),
                n_managers=('n_managers', 'mean'),
                n_quarters=('quarter', 'nunique')).reset_index()
            print(f"  persistent IO measure: {len(io_perm):,} permnos "
                  f"(matched on >=1 of {len(quarters_used)} quarters); "
                  f"median io_share = {io_perm['io_share'].median():.3f}")
            d_io = d.merge(io_perm, on='permno', how='inner')
            cov = d_io['permno'].nunique()
            print(f"  panel coverage with IO: {cov} permnos, "
                  f"{len(d_io):,} firm-months "
                  f"({len(d_io)/len(d)*100:.1f}% of panel)")
            io_terc = pd.qcut(io_perm.set_index('permno')['io_share'], 3,
                              labels=['IO1_low', 'IO2_mid', 'IO3_high'])
            d_io['io_grp'] = d_io['permno'].map(io_terc).astype(str)
            terc_means = d_io.groupby('io_grp')['io_share'].mean().to_dict()
            print(f"  IO tercile mean io_share: {terc_means}")
            io_out = run_split(d_io, 'io_grp',
                               ['IO1_low', 'IO2_mid', 'IO3_high'], "13F-IO")
            io_out['_coverage'] = {
                "n_permnos_with_io": int(cov),
                "n_firm_months": int(len(d_io)),
                "pct_of_panel": float(len(d_io) / len(d) * 100),
                "io_tercile_mean_share": {k: float(v)
                                          for k, v in terc_means.items()},
                "quarters_used": quarters_used,
                "n_permnos_in_io_measure": int(len(io_perm)),
            }
            results['splits']['edgar_13f'] = io_out
            results['edgar_13f_status'] = (
                f"EXECUTED on {len(quarters_used)} EDGAR-13F quarter(s) "
                f"({', '.join(quarters_used)}). The third targeted quarter "
                f"(2019Q2) download was killed by a runtime cap; two quarters "
                f"spanning 2011 and 2015 still give a persistent IO measure."
            )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ---- LaTeX table ----
    with open(TABLE, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        f.write("Group & three-way $\\hat\\gamma$ & SE & $t$ & $N$ \\\\\n")
        f.write("\\hline\n")
        if results['splits'].get('edgar_13f'):
            io_out = results['splits']['edgar_13f']
            f.write("\\multicolumn{5}{l}{\\textit{Panel A: EDGAR-13F "
                    "institutional-ownership terciles}} \\\\\n")
            for lab, disp in [('IO1_low', 'Low inst. own. (high retail)'),
                              ('IO2_mid', 'Mid inst. own.'),
                              ('IO3_high', 'High inst. own.')]:
                r = io_out[lab]
                f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                        f"${r['t']:.2f}$ & ${r['n_obs']:,}$ \\\\\n")
            idf = io_out['_difference']
            f.write(f"\\quad Difference (low IO $-$ high IO) & "
                    f"${idf['diff_coef']:.4f}$ & ${idf['diff_se']:.4f}$ & "
                    f"${idf['diff_t']:.2f}$ & --- \\\\\n")
            f.write("\\hline\n")
        f.write("\\multicolumn{5}{l}{\\textit{Panel B: Size terciles "
                "(small-cap = high-retail proxy)}} \\\\\n")
        for lab, disp in [('T1_small', 'Small-cap (high retail)'),
                          ('T2_mid', 'Mid-cap'),
                          ('T3_large', 'Large-cap (high institutional)')]:
            r = size_out[lab]
            f.write(f"{disp} & ${r['coef']:.4f}$ & ${r['se']:.4f}$ & "
                    f"${r['t']:.2f}$ & ${r['n_obs']:,}$ \\\\\n")
        sd = size_out['_difference']
        f.write(f"\\quad Difference (small $-$ large) & ${sd['diff_coef']:.4f}$ "
                f"& ${sd['diff_se']:.4f}$ & ${sd['diff_t']:.2f}$ & --- \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    # ---- markdown ----
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Deepen Item 6 — 13F Retail/Institutional Ownership Split\n\n")
        f.write("The mechanism (sophistication-modulated **local retail** "
                "demand) predicts the three-way coefficient is present in "
                "high-retail / low-institutional stocks and absent in "
                "institutionally-held stocks.\n\n")
        if results['splits'].get('edgar_13f'):
            io_out = results['splits']['edgar_13f']
            cov = io_out['_coverage']
            f.write("## Split A — EDGAR-13F institutional-ownership terciles "
                    "(the mechanism-discriminating test)\n\n")
            f.write(f"- {results['edgar_13f_status']}\n")
            f.write(f"- Persistent institutional-ownership measure from SEC "
                    f"EDGAR 13F-HR filings, quarters "
                    f"{', '.join(cov['quarters_used'])} "
                    f"(WRDS Thomson s34 unavailable; OSAP has no raw IO-share "
                    f"signal).\n")
            f.write(f"- io_share = (institutional shares held, summed across "
                    f"13F filers, Shares-type, options excluded) / (shares "
                    f"outstanding); matched to the panel by ticker; averaged "
                    f"across the available quarters.\n")
            f.write(f"- IO measure built for {cov['n_permnos_in_io_measure']:,} "
                    f"permnos; panel coverage = {cov['n_permnos_with_io']} "
                    f"permnos, {cov['n_firm_months']:,} firm-months "
                    f"({cov['pct_of_panel']:.1f}% of the panel).\n")
            f.write(f"- IO tercile mean institutional-ownership share: "
                    f"{cov['io_tercile_mean_share']}.\n\n")
            f.write("| Group | three-way coef | SE | t | N | SE kind |\n")
            f.write("|---|---|---|---|---|---|\n")
            for lab, disp in [('IO1_low', 'Low IO (high retail)'),
                              ('IO2_mid', 'Mid IO'),
                              ('IO3_high', 'High IO (institutional)')]:
                r = io_out[lab]
                f.write(f"| {disp} | {r['coef']:.6f} | {r['se']:.6f} | "
                        f"{r['t']:.3f} | {r['n_obs']:,} | {r['se_kind']} |\n")
            idf = io_out['_difference']
            f.write(f"\n**Difference (low-IO minus high-IO):** "
                    f"{idf['diff_coef']:+.6f} (SE {idf['diff_se']:.6f}, "
                    f"t {idf['diff_t']:.3f}). Mechanism predicts a negative "
                    f"difference (three-way more negative where retail "
                    f"dominates). "
                    f"{'SUPPORTED at 10%' if idf['mechanism_supported'] else 'NOT supported at 10%'}.\n\n")
        else:
            f.write("## Split A — EDGAR-13F institutional-ownership terciles\n\n")
            f.write(f"**{results.get('edgar_13f_status','not run')}**\n\n")
        f.write("## Split B — firm-size terciles (companion, panel-only)\n\n")
        f.write(results['size_split_note'] + "\n\n")
        f.write("| Group | three-way coef | SE | t | N | SE kind |\n")
        f.write("|---|---|---|---|---|---|\n")
        for lab, disp in [('T1_small', 'Small-cap (high-retail proxy)'),
                          ('T2_mid', 'Mid-cap'),
                          ('T3_large', 'Large-cap (high-institutional proxy)')]:
            r = size_out[lab]
            f.write(f"| {disp} | {r['coef']:.6f} | {r['se']:.6f} | "
                    f"{r['t']:.3f} | {r['n_obs']:,} | {r['se_kind']} |\n")
        sd = size_out['_difference']
        f.write(f"\n**Difference (small-cap minus large-cap):** "
                f"{sd['diff_coef']:+.6f} (SE {sd['diff_se']:.6f}, "
                f"t {sd['diff_t']:.3f}). Mechanism predicts a negative "
                f"difference. "
                f"{'SUPPORTED at 10%' if sd['mechanism_supported'] else 'NOT supported at 10%'}.\n\n")
        f.write("## Files\n\n")
        f.write("- `output/stage3a/deepen_13f_split.json`\n")
        f.write("- `code/empirical/deepen_13f_download.py` (EDGAR 13F download)\n")
        f.write("- `code/empirical/deepen_13f_split.py` (this analysis)\n")
        f.write("- `code/empirical/_edgar_13f_*_ticker.parquet` (per-quarter "
                "13F ticker aggregates)\n")
    print(f"\njson -> {OUT_JSON}\nmd   -> {OUT_MD}\ntex  -> {TABLE}")
    return results


if __name__ == '__main__':
    main()
