# =====================================================================
# deepen_io_panel.py
# Shared module that builds the firm-quarter institutional-ownership panel from EDGAR 13F quarters (cusip9 or ticker path) and merges it to the firm-month panel with no look-ahead (time-varying + persistent IO measures).
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet; code/empirical/_edgar_13f_*_cusip9.parquet + *_xwalk.parquet (preferred) or *_ticker.parquet (fallback)
# Outputs:   in-memory firm-quarter IO table + merged panel (no files written); printed diagnostics when run as __main__
# Paper:     SUPERSEDED by deepen_thomson_io_panel_v6.py as the live IO panel builder — kept for provenance (EDGAR-proxy; feeds the IA EDGAR-proxy cross-check)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 — IO panel builder (shared module for items a-d).

Builds the firm-quarter institutional-ownership panel from EDGAR 13F quarters and
merges it to the firm-month panel with NO look-ahead (each firm-month gets the most
recent available 13F quarter's IO share).

Two construction paths, auto-selected by what is on disk:
  PATH 1 (preferred — v4 source-deduplicated): _edgar_13f_{Q}_cusip9.parquet +
    _edgar_13f_{Q}_xwalk.parquet. IO measured at cusip9 (security/share-class),
    matched to the panel via the per-quarter ticker->cusip6 crosswalk, with the
    shares-outstanding denominator aligned to the 13F report quarter-end month.
  PATH 2 (fallback — round-0 ticker aggregates): _edgar_13f_{Q}_ticker.parquet,
    matched by ticker only (the round-0 path; kept so items b-d still run if the
    v4 download produces no cusip9 files).

Returns the firm-month panel with columns io_share (time-varying, no look-ahead),
io_share_persist (time-mean across all available quarters — the round-0-style
persistent measure, for the 2-quarter-comparable split), n_io_quarters, and the
firm-quarter IO table itself.

The share-above-one problem is addressed at the SOURCE in PATH 1 (per-filer max by
cusip9, sum across filers, cusip9-level denominator). Any residual io_share > 1.0
is reported and capped at 1.0 ONLY for the final measure, with the rate disclosed.
"""

import os
import glob
import re
import numpy as np
import pandas as pd

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
EMP = os.path.join(ROOT, "code", "empirical")
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed",
                     "panel_corrected_standardized.parquet")


def _q_to_month(qlabel):
    """13F-HR filed in Qk reports holdings as of the prior quarter-end. We map a
    QYYYYQk label to the report month = end of Qk's calendar quarter (the panel
    month whose shrout best matches the 13F position date)."""
    m = re.match(r"(\d{4})Q(\d)", qlabel)
    yr, q = int(m.group(1)), int(m.group(2))
    month = {1: '03', 2: '06', 3: '09', 4: '12'}[q]
    return f"{yr}-{month}"


def load_panel():
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    return df


def build_io_firmquarter(df, verbose=True):
    """Return a firm-quarter IO table: columns permno, quarter, ym_report,
    io_share, n_filers, path. Auto-selects PATH 1 (cusip9) or PATH 2 (ticker)."""
    c9_files = sorted(glob.glob(os.path.join(EMP, "_edgar_13f_*_cusip9.parquet")))
    if c9_files:
        return _build_path1_cusip9(df, c9_files, verbose)
    tk_files = sorted(glob.glob(os.path.join(EMP, "_edgar_13f_*_ticker.parquet")))
    if tk_files:
        return _build_path2_ticker(df, tk_files, verbose)
    raise FileNotFoundError("no EDGAR 13F quarter files on disk")


def _build_path1_cusip9(df, c9_files, verbose):
    rows = []
    diag = []
    for c9p in c9_files:
        qlabel = os.path.basename(c9p).split('_')[3]
        xwp = os.path.join(EMP, f"_edgar_13f_{qlabel}_xwalk.parquet")
        if not os.path.exists(xwp):
            continue
        c9 = pd.read_parquet(c9p)
        xw = pd.read_parquet(xwp)
        ym_rep = _q_to_month(qlabel)
        pq = df[df['ym'] == ym_rep][['permno', 'ticker', 'shrout']].copy()
        pq = pq.dropna(subset=['ticker', 'shrout'])
        pq = pq[pq['ticker'] != '']
        pq = pq.drop_duplicates('permno')
        xw = xw.dropna(subset=['cusip6'])
        pq = pq.merge(xw[['Ticker', 'cusip6']], left_on='ticker',
                      right_on='Ticker', how='inner')
        c9['Ticker'] = c9['Ticker'].astype(str).str.strip().str.upper()
        # per ticker pick the cusip9 with the most inst shares (dominant class)
        c9_tk = c9[c9['Ticker'] != ''].copy()
        if len(c9_tk):
            idx = c9_tk.groupby('Ticker')['inst_shares'].idxmax()
            c9_dom = c9_tk.loc[idx, ['Ticker', 'cusip9', 'cusip6',
                                     'inst_shares', 'n_filers']]
        else:
            c9_dom = pd.DataFrame(columns=['Ticker', 'cusip9', 'cusip6',
                                           'inst_shares', 'n_filers'])
        mrg = pq.merge(c9_dom, left_on='ticker', right_on='Ticker',
                       how='inner', suffixes=('', '_c9'))
        denom = mrg['shrout'] * 1000.0  # panel shrout in 1000s
        mrg['io_share_raw'] = mrg['inst_shares'] / denom
        n_match = len(mrg)
        n_over = int((mrg['io_share_raw'] > 1.0).sum())
        diag.append({'quarter': qlabel, 'ym_report': ym_rep,
                     'n_matched': n_match,
                     'pct_over_one': 100.0 * n_over / max(n_match, 1),
                     'median_io': float(mrg['io_share_raw'].median())})
        if verbose:
            print(f"  [{qlabel}] matched {n_match} permnos | "
                  f"io>1.0: {100.0*n_over/max(n_match,1):.1f}% | "
                  f"median io {mrg['io_share_raw'].median():.3f}")
        mrg['io_share'] = mrg['io_share_raw'].clip(upper=1.0)
        mrg['quarter'] = qlabel
        mrg['ym_report'] = ym_rep
        rows.append(mrg[['permno', 'quarter', 'ym_report', 'io_share',
                         'io_share_raw', 'n_filers']])
    if not rows:
        raise RuntimeError("PATH 1: no cusip9 quarter merged")
    fq = pd.concat(rows, ignore_index=True)
    fq['path'] = 'cusip9'
    fq.attrs['diag'] = pd.DataFrame(diag)
    return fq


def _build_path2_ticker(df, tk_files, verbose):
    rows = []
    diag = []
    for tkp in tk_files:
        qlabel = os.path.basename(tkp).split('_')[3]
        t = pd.read_parquet(tkp)
        t['Ticker'] = t['Ticker'].astype(str).str.strip().str.upper()
        ym_rep = _q_to_month(qlabel)
        pq = df[df['ym'] == ym_rep][['permno', 'ticker', 'shrout']].copy()
        pq = pq.dropna(subset=['ticker', 'shrout'])
        pq = pq[pq['ticker'] != ''].drop_duplicates('permno')
        mrg = pq.merge(t[['Ticker', 'inst_shares', 'n_managers']],
                       left_on='ticker', right_on='Ticker', how='inner')
        mrg['io_share_raw'] = mrg['inst_shares'] / (mrg['shrout'] * 1000.0)
        n_match = len(mrg)
        n_over = int((mrg['io_share_raw'] > 1.0).sum())
        diag.append({'quarter': qlabel, 'ym_report': ym_rep,
                     'n_matched': n_match,
                     'pct_over_one': 100.0 * n_over / max(n_match, 1),
                     'median_io': float(mrg['io_share_raw'].median())})
        if verbose:
            print(f"  [{qlabel}] (PATH2 ticker) matched {n_match} | "
                  f"io>1.0: {100.0*n_over/max(n_match,1):.1f}%")
        mrg['io_share'] = mrg['io_share_raw'].clip(upper=1.0)
        mrg['quarter'] = qlabel
        mrg['ym_report'] = ym_rep
        mrg = mrg.rename(columns={'n_managers': 'n_filers'})
        rows.append(mrg[['permno', 'quarter', 'ym_report', 'io_share',
                         'io_share_raw', 'n_filers']])
    fq = pd.concat(rows, ignore_index=True)
    fq['path'] = 'ticker'
    fq.attrs['diag'] = pd.DataFrame(diag)
    return fq


def merge_io_to_panel(df, fq, verbose=True):
    """Merge the firm-quarter IO table to the firm-month panel.

    Produces two IO columns:
      io_share        — TIME-VARYING, no look-ahead: each firm-month gets the IO
                        share from the most recent 13F quarter at or before it.
      io_share_persist— time-mean of io_share across all the firm's available
                        13F quarters (the round-0-style persistent split var).
    """
    fq = fq.copy()
    fq['report_date'] = pd.to_datetime(fq['ym_report'] + '-01')
    df = df.copy()
    df['month_date'] = pd.to_datetime(df['ym'] + '-01')
    # merge_asof: BOTH frames sorted on the asof key globally.
    left = df[['permno', 'month_date']].drop_duplicates().sort_values(
        ['month_date', 'permno']).reset_index(drop=True)
    right = fq[['permno', 'report_date', 'io_share']].dropna(
        subset=['io_share']).sort_values(
        ['report_date', 'permno']).reset_index(drop=True)
    tv = pd.merge_asof(left, right, left_on='month_date',
                       right_on='report_date', by='permno',
                       direction='backward')
    tv = tv.rename(columns={'io_share': 'io_share_tv'})
    df = df.merge(tv[['permno', 'month_date', 'io_share_tv']],
                  on=['permno', 'month_date'], how='left')
    df = df.rename(columns={'io_share_tv': 'io_share'})
    # persistent measure
    persist = fq.groupby('permno').agg(
        io_share_persist=('io_share', 'mean'),
        n_io_quarters=('quarter', 'nunique')).reset_index()
    df = df.merge(persist, on='permno', how='left')
    if verbose:
        cov_tv = df['io_share'].notna().sum()
        cov_p = df['io_share_persist'].notna().sum()
        print(f"  time-varying IO coverage: {cov_tv:,} firm-months "
              f"({100.0*cov_tv/len(df):.1f}%)")
        print(f"  persistent IO coverage:   {cov_p:,} firm-months "
              f"({100.0*cov_p/len(df):.1f}%), "
              f"{df.loc[df['io_share_persist'].notna(),'permno'].nunique()} permnos")
        med_q = df.groupby('permno')['n_io_quarters'].first().median()
        print(f"  median n_io_quarters per covered permno: {med_q}")
    return df


if __name__ == '__main__':
    df = load_panel()
    print(f"panel: {len(df):,} firm-months, {df['permno'].nunique()} permnos")
    fq = build_io_firmquarter(df)
    print(f"\nfirm-quarter IO table: {len(fq):,} rows, "
          f"{fq['permno'].nunique()} permnos, "
          f"quarters {sorted(fq['quarter'].unique())}, path={fq['path'].iloc[0]}")
    print("\nper-quarter diagnostics:")
    print(fq.attrs['diag'].to_string(index=False))
    dfm = merge_io_to_panel(df, fq)


def build_io_firmquarter_ticker_only(df, verbose=True):
    """Force PATH 2 (round-0 ticker aggregates). Used by the v4 analysis to
    reproduce the round-0 ticker-matched IO measure and re-run the round-0
    difference under the proper stacked clustering — isolating whether the
    round-0 t = -3.56 dies from (a) proper clustering alone or (b) the
    source-level CUSIP9 dedup. Returns the same firm-quarter table shape as
    build_io_firmquarter, path='ticker'."""
    import glob as _glob
    tk_files = sorted(_glob.glob(os.path.join(EMP, "_edgar_13f_*_ticker.parquet")))
    if not tk_files:
        raise FileNotFoundError("no round-0 ticker files on disk")
    return _build_path2_ticker(df, tk_files, verbose)
