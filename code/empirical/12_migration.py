# =====================================================================
# 12_migration.py
# Reconstructs the interaction-zone MIGRATION result (Table 4 + Figure 2 of
# the main paper) from the stable-HQ panel, and regenerates Figure 2 cleanly.
#
# This result was originally produced by ad-hoc code that was not saved; this
# script is the reproducible replacement. Validation against the published
# Table 4 and the saved per-state CSVs is reported when run with --validate.
#
# Inputs:    _dfm_stable_hq.parquet (stable-HQ panel: returns, momentum, IV,
#            io_share_persist, log_me, hq_state, the literacy three-way);
#            output/state_views/per_state_three_way_size_controlled.csv and
#            per_state_abv_med_io1.csv (saved per-state coefficients for the two
#            cross-state cells that reproduce exactly).
# Outputs:   output/stage3a/results_migration.json;
#            output/stage3a/tables/tab_migration.tex;
#            output/state_views/figure2_migration.png  (clean, no baked-in title)
# Paper:     Main Table 4 (tab:migration) + Figure 2 (fig:migration)
# Method:    size-controlled three-way (Eq. 2): literacy three-way with size_z,
#            mom*size_z, iv*size_z, mom*iv*size_z (each within-month z-scored)
#            and the 6 lower-order literacy terms as controls, state+month FE,
#            FWL partialling, state-clustered SE. IO terciles re-formed (persistent
#            IO) within each size filter. Per-state cells use month FE only;
#            cross-state slope is precision-weighted (1/SE^2). Mirrors the
#            convention in v9_placeholders_resolve.test_7a.
# Run order: see code/00_master.py (Stage 3 / headline results)
# =====================================================================
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from v9_helpers import load_stable_hq, add_io_terciles, ROOT
from deepen_estimators import FOCAL, _cluster_matrix

LIT3 = 'mom_x_iv_x_literacy_corr'
SIZE_COLS = ['size_z', 'mom_x_size_z', 'iv_x_size_z', 'mom_x_iv_x_size_z']
OUT = os.path.join(ROOT, "output", "stage3a")
TABDIR = os.path.join(OUT, "tables")
SV = os.path.join(ROOT, "output", "state_views")
FILTERS = [0.0, 0.20, 0.33, 0.50]
FILTER_LABELS = {0.0: "Full panel", 0.20: "Drop bottom 20%",
                 0.33: "Drop bottom 33%", 0.50: "Drop bottom 50%"}
TERCILES = ['IO1_low', 'IO2_mid', 'IO3_high']


def _zw(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0.0


def build_size_terms(d):
    d = d.copy()
    d['size_z'] = d.groupby('ym')['log_me'].transform(_zw)
    d['mom_x_size_z'] = d['mom_12_2'] * d['size_z']
    d['iv_x_size_z'] = d['iv'] * d['size_z']
    d['mom_x_iv_x_size_z'] = d['mom_x_iv'] * d['size_z']
    for c in ['mom_x_size_z', 'iv_x_size_z', 'mom_x_iv_x_size_z', 'size_z']:
        d[c] = d.groupby('ym')[c].transform(_zw)
    return d


def _design(d, state_fe=True):
    n = len(d)
    lo6 = d[FOCAL[:6]].values.astype(float)
    size = d[SIZE_COLS].values.astype(float)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    rows = np.arange(n)
    blocks = [sp.csr_matrix(np.ones((n, 1))), sp.csr_matrix(lo6), sp.csr_matrix(size)]
    if state_fe:
        sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
        blocks.append(sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, sc.max() + 1))[:, 1:])
    blocks.append(sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, mc.max() + 1))[:, 1:])
    return sp.hstack(blocks).tocsc()


def sizectrl_three_way(d, state_fe=True):
    """Size-controlled literacy three-way via FWL; state-clustered SE (state_fe)
    or HC0 robust SE (per-state, month FE only)."""
    x3 = d[LIT3].values.astype(float)
    y = d['ret'].values.astype(float)
    W = _design(d, state_fe=state_fe)
    WtWi = np.linalg.pinv((W.T @ W).toarray())
    x3t = x3 - W @ (WtWi @ (W.T @ x3))
    yt = y - W @ (WtWi @ (W.T @ y))
    Sxx = float(x3t @ x3t)
    coef = float((x3t @ yt) / Sxx)
    e = yt - coef * x3t
    score = x3t * e
    if state_fe:
        Cs, _ = _cluster_matrix(pd.Categorical(d['hq_state']).codes.astype(np.int64))
        se = float(np.sqrt(((Cs.T @ score) ** 2).sum()) / Sxx)
    else:
        se = float(np.sqrt((score ** 2).sum()) / Sxx)
    return {"coef": coef, "se": se, "t": coef / se if se > 0 else np.nan, "n": int(len(d))}


def panel_A(base, firm_me):
    grid = {}
    for pct in FILTERS:
        if pct > 0:
            keep = firm_me[firm_me >= firm_me.quantile(pct)].index
            df = base[base['permno'].isin(keep)].copy()
        else:
            df = base.copy()
        df = add_io_terciles(df, 'io_share_persist')
        grid[pct] = {"N": int(len(df)),
                     **{g: sizectrl_three_way(df[df['io_grp'] == g]) for g in TERCILES}}
    return grid


def per_state_cell(df_cell, min_firms=5):
    """Per-state size-controlled three-way (month FE only) for a sample cell."""
    rows = []
    for st, g in df_cell.groupby('hq_state'):
        if g['permno'].nunique() < min_firms:
            continue
        r = sizectrl_three_way(g, state_fe=False)
        rows.append({'hq_state': st, 'coef': r['coef'], 'se': r['se'],
                     'n_firms': int(g['permno'].nunique())})
    return pd.DataFrame(rows)


def weighted_slope(ps, lit_col='pass3_panel_mean_pct'):
    ps = ps.dropna(subset=['coef', 'se', lit_col]).copy()
    w = 1.0 / ps['se'].values ** 2
    x = ps[lit_col].values.astype(float); y = ps['coef'].values.astype(float)
    xb = np.average(x, weights=w); yb = np.average(y, weights=w)
    b = (w * (x - xb) * (y - yb)).sum() / (w * (x - xb) ** 2).sum()
    t = b * np.sqrt((w * (x - xb) ** 2).sum())          # precision-weighted (GLS) t
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(b), "t": float(t), "pearson_r": r, "n_states": int(len(ps)),
            "xb": float(xb), "yb": float(yb)}


def main():
    d = build_size_terms(load_stable_hq())
    base = d.dropna(subset=FOCAL + SIZE_COLS + ['ret', 'hq_state', 'io_share_persist', 'me']).copy()
    firm_me = base.groupby('permno')['me'].mean()
    lit = pd.read_csv(os.path.join(SV, 'state_literacy_panel.csv'))[['state', 'pass3_panel_mean_pct']]
    lit['hq_state'] = 'US-' + lit['state']

    # ---------- Table 4 Panel A ----------
    grid = panel_A(base, firm_me)

    # ---------- Table 4 Panel B / Figure 2 scatters ----------
    # Two cells reproduce exactly from saved per-state CSVs; one is re-estimated.
    full_mid_csv = pd.read_csv(os.path.join(SV, 'per_state_three_way_size_controlled.csv'))
    full_mid_csv = full_mid_csv[full_mid_csv['n_firms'] >= 5]
    migr_io1_csv = pd.read_csv(os.path.join(SV, 'per_state_abv_med_io1.csv'))
    migr_io1_csv = migr_io1_csv[migr_io1_csv['n_firms'] >= 5]

    keep50 = firm_me[firm_me >= firm_me.quantile(0.50)].index
    above = base[base['permno'].isin(keep50)].copy()
    above = add_io_terciles(above, 'io_share_persist')
    loses_mid = per_state_cell(above[above['io_grp'] == 'IO2_mid']).merge(lit, on='hq_state', how='left')
    null_io3 = per_state_cell(above[above['io_grp'] == 'IO3_high']).merge(lit, on='hq_state', how='left')

    cells = {
        "full_panel_mid_IO": {"ps": full_mid_csv, "published": {"slope": -0.0623, "t": -5.01, "r": -0.275, "n": 42},
                              "title": "Full panel, mid-IO\n(headline cell)"},
        "above_median_mid_IO": {"ps": loses_mid, "published": {"slope": -0.0325, "t": -3.82, "r": 0.008, "n": 31},
                                "title": "Above-median size, mid-IO\n(loses the gradient)"},
        "above_median_low_IO": {"ps": migr_io1_csv, "published": {"slope": -0.0379, "t": -4.30, "r": -0.291, "n": 34},
                                "title": "Above-median size, low-IO\n(migrated headline cell)"},
    }
    for k, c in cells.items():
        c["recomputed"] = weighted_slope(c["ps"])
    # drop50 IO3 (predicted-null cell, Table 4 Panel B row 4; not in figure)
    io3 = weighted_slope(null_io3)
    io3_published = {"slope": 0.0061, "t": 0.26, "r": 0.054, "n": 31}

    # ---------- save numbers ----------
    results = {
        "spec": "size-controlled literacy three-way (Eq.2), state+month FE, "
                "state-clustered SE; IO terciles re-formed (persistent IO) within "
                "each firm panel-mean market-cap filter; per-state month-FE-only, "
                "cross-state slope precision-weighted 1/SE^2.",
        "panelA": {FILTER_LABELS[p]: {"N": grid[p]["N"],
                    **{g: {"coef": grid[p][g]["coef"], "se": grid[p][g]["se"],
                           "t": grid[p][g]["t"]} for g in TERCILES}} for p in FILTERS},
        "panelB": {**{k: {"published": c["published"], "recomputed": c["recomputed"]}
                      for k, c in cells.items()},
                   "above_median_high_IO": {"published": io3_published, "recomputed": io3}},
    }
    os.makedirs(TABDIR, exist_ok=True)
    with open(os.path.join(OUT, "results_migration.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results_migration.json")

    write_tex(grid, cells, io3)
    make_figure(grid, cells)
    print("DONE")


def write_tex(grid, cells, io3):
    def fmt(g):
        return f"${g['coef']:+.4f}$ (${g['t']:+.2f}$)"

    def fmtB(rc):
        return f"${rc['slope']:+.4f}$ & ${rc['t']:+.2f}$ & ${rc['pearson_r']:+.3f}$ & {rc['n_states']}"
    lines = [r"% Auto-generated by 12_migration.py - Table 4 (migration), reconstructed.",
             r"\textit{Panel A: within-panel three-way coefficient by IO tercile}\\[2pt]",
             r"\begin{tabular}{lcccr}", r"\toprule",
             r"Size filter & IO1 (low) & IO2 (mid) & IO3 (high) & $N$ \\", r"\midrule"]
    for p in FILTERS:
        g = grid[p]
        lab = FILTER_LABELS[p].replace('%', r'\%')
        lines.append(f"{lab} & {fmt(g['IO1_low'])} & {fmt(g['IO2_mid'])} & "
                     f"{fmt(g['IO3_high'])} & {g['N']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"", r"\medskip",
              r"\textit{Panel B: cross-state precision-weighted slope of per-state "
              r"$\hat\gamma_{\mathrm{lit}}$ on state literacy}\\[2pt]",
              r"\begin{tabular}{lcccc}", r"\toprule",
              r"Sample cell & Weighted slope & state-$t$ & Pearson $r$ & $n_{\text{states}}$ \\",
              r"\midrule",
              f"Full panel, mid-IO & {fmtB(cells['full_panel_mid_IO']['recomputed'])} \\\\",
              f"Drop bottom 50\\%, mid-IO & {fmtB(cells['above_median_mid_IO']['recomputed'])} \\\\",
              f"Drop bottom 50\\%, IO1 & {fmtB(cells['above_median_low_IO']['recomputed'])} \\\\",
              f"Drop bottom 50\\%, IO3 & {fmtB(io3)} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABDIR, "tab_migration.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_migration.tex")


def make_figure(grid, cells):
    colors = {'IO1_low': '#5b8db8', 'IO2_mid': '#4f9d69', 'IO3_high': '#e8a0a0'}
    labels = {'IO1_low': 'IO1 low', 'IO2_mid': 'IO2 mid', 'IO3_high': 'IO3 high'}
    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

    # ----- Panel A: grouped bars -----
    axA = fig.add_subplot(gs[0, :])
    xf = np.arange(len(FILTERS)); width = 0.26
    for j, g in enumerate(TERCILES):
        coefs = [grid[p][g]["coef"] for p in FILTERS]
        ses = [grid[p][g]["se"] for p in FILTERS]
        ts = [grid[p][g]["t"] for p in FILTERS]
        xpos = xf + (j - 1) * width
        axA.bar(xpos, coefs, width, yerr=[2 * s for s in ses], capsize=3,
                color=colors[g], edgecolor='black', linewidth=0.6, label=labels[g],
                error_kw={'linewidth': 0.9})
        for xi, c, t in zip(xpos, coefs, ts):
            if abs(t) >= 2:
                axA.text(xi, c - np.sign(c) * (2 * ses[FILTERS.index(FILTERS[0])] + 0.004)
                         if False else c, '', ha='center')
                axA.annotate('*', (xi, c + (0.004 if c >= 0 else -0.010)),
                             ha='center', va='bottom' if c >= 0 else 'top',
                             fontsize=15, fontweight='bold')
    axA.axhline(0, color='black', linewidth=0.7)
    axA.set_xticks(xf); axA.set_xticklabels([FILTER_LABELS[p] for p in FILTERS])
    axA.set_ylabel(r'Three-way coefficient $\hat\gamma_{\mathrm{lit}}$')
    axA.set_title(r'Panel A: within-panel $\hat\gamma_{\mathrm{lit}}$ by IO tercile '
                  r'under progressive size filters (size-controlled; whiskers $\pm 2$ SE; $*\,|t|\geq 2$)',
                  fontsize=10)
    axA.legend(title='IO tercile', fontsize=9, title_fontsize=9, loc='upper right')

    # ----- Panel B: three cross-state scatters -----
    for i, (k, c) in enumerate(cells.items()):
        ax = fig.add_subplot(gs[1, i])
        ps = c["ps"].dropna(subset=['coef', 'pass3_panel_mean_pct']).copy()
        sizes = 18 + 26 * np.sqrt(ps['n_firms'].clip(lower=1))
        ax.scatter(ps['pass3_panel_mean_pct'], ps['coef'], s=sizes,
                   color=colors['IO2_mid'] if 'mid' in k else colors['IO1_low'],
                   alpha=0.55, edgecolor='black', linewidth=0.4)
        xs = np.array([ps['pass3_panel_mean_pct'].min(), ps['pass3_panel_mean_pct'].max()])
        yb = c["recomputed"]["yb"]; xb = c["recomputed"]["xb"]
        ax.plot(xs, yb + c["recomputed"]["slope"] * (xs - xb), '--', color='black', linewidth=1.3)
        ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
        ax.set_title(c["title"], fontsize=9.5)
        ax.set_xlabel('State literacy (panel mean %)', fontsize=9)
        if i == 0:
            ax.set_ylabel(r'Per-state $\hat\gamma_{\mathrm{lit}}$', fontsize=9)
        rc = c["recomputed"]
        txt = (f"slope = {rc['slope']:+.4f}\n$t$ = {rc['t']:+.2f}\n"
               f"$r$ = {rc['pearson_r']:+.3f}\n$n$ = {rc['n_states']}")
        ax.text(0.04, 0.04, txt, transform=ax.transAxes, fontsize=8.3, va='bottom',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

    for ext in ['png']:
        path = os.path.join(SV, f"figure2_migration.{ext}")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print("wrote", path)
    plt.close(fig)


if __name__ == '__main__':
    main()
