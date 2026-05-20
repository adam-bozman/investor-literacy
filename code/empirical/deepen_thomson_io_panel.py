# =====================================================================
# deepen_thomson_io_panel.py
# Shared module that loads the real Thomson s34 firm-quarter IO panel and merges it to the firm-month panel with a conservative no-look-ahead lag (LAG_MONTHS=4), producing time-varying and persistent IO measures.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet; code/empirical/_thomson_s34_firmquarter.parquet (WRDS Thomson Reuters s34)
# Outputs:   in-memory merged panel + diagnostics (no files written); printed diagnostics when run as __main__
# Paper:     SUPERSEDED by deepen_thomson_io_panel_v6.py — kept for provenance (pre-denominator-correction; feeds the IA Thomson s34 panel construction section)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 (WRDS-recovered re-fire) — IO panel builder on the REAL
Thomson Reuters s34 13F panel.

Consumes code/empirical/_thomson_s34_firmquarter.parquet (built by
deepen_thomson_s34_download.py — the genuine 60-quarter 2009Q1-2023Q4 quarterly
institutional-ownership panel) and merges it to the firm-month panel.

Two IO measures, mirroring deepen_io_panel.py so the v4 estimators run unchanged:
  io_share         — TIME-VARYING, no look-ahead. Each firm-month gets the IO
                     share from the most recent 13F quarter whose report date
                     (rdate) is STRICTLY BEFORE the firm-month, i.e. at least one
                     full quarter stale. The 13F filing for quarter rdate is
                     public ~45 days after rdate (fdate); to be strictly
                     conservative we require rdate < firm-month-start AND we lag
                     one extra quarter so the most recent usable quarter is the
                     one whose holdings were already public. Concretely: a
                     firm-month in calendar month t uses the IO_share of the most
                     recent rdate <= (t minus ~4 months) — backward merge_asof on
                     a 4-month-lagged report date.
  io_share_persist — time-mean of io_share_raw (capped) across all the firm's
                     available 13F quarters — the persistent / between-firm split
                     variable, comparable to the round-0/v4 persistent measure.

The share-above-one problem: Thomson s34 summed across managers can exceed
shares outstanding (multi-class share confusion, cusip-history edge cases,
manager double-reporting of jointly-held blocks). We report the rate and cap at
1.0 for the final measure (no post-hoc cap beyond that). The real Thomson panel
is expected to be far cleaner than the 21.7% EDGAR-proxy rate.
"""

import os
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed",
                     "panel_corrected_standardized.parquet")
S34_FQ = os.path.join(EMP, "_thomson_s34_firmquarter.parquet")

# conservative no-look-ahead lag: a firm-month uses the most recent 13F quarter
# whose report date is >= LAG_MONTHS before the firm-month. 4 months guarantees
# the 13F (public ~45 days after rdate) was already filed AND adds a full extra
# quarter of staleness so the contemporaneous-information critique cannot bite.
LAG_MONTHS = 4


def load_panel():
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    return df


def load_s34_firmquarter():
    if not os.path.exists(S34_FQ):
        raise FileNotFoundError(f"Thomson s34 firm-quarter panel not found: "
                                f"{S34_FQ}. Run deepen_thomson_s34_download.py.")
    fq = pd.read_parquet(S34_FQ)
    fq['io_share'] = fq['io_share_raw'].clip(upper=1.0)
    fq['rdate'] = pd.to_datetime(fq['rdate'])
    return fq


def io_share_distribution(fq):
    """Diagnostic table: per-quarter coverage and io>1.0 rate."""
    g = fq.groupby('quarter')
    diag = g.agg(
        n_permnos=('permno', 'nunique'),
        median_io_raw=('io_share_raw', 'median'),
        mean_io_raw=('io_share_raw', 'mean'),
    ).reset_index()
    over = g.apply(lambda x: 100.0 * (x['io_share_raw'] > 1.0).mean(),
                   include_groups=False).rename('pct_over_one')
    diag = diag.merge(over.reset_index(), on='quarter')
    return diag


def merge_io_to_panel(df, fq, verbose=True):
    """Merge firm-quarter Thomson IO to the firm-month panel.

    io_share         — time-varying, no look-ahead (rdate lagged LAG_MONTHS).
    io_share_persist — time-mean io_share across the firm's available quarters.
    n_io_quarters    — count of distinct 13F quarters available for the firm.
    """
    fq = fq.copy()
    # backward merge_asof on a LAG_MONTHS-shifted report date
    fq['report_date'] = fq['rdate'] + pd.DateOffset(months=LAG_MONTHS)
    df = df.copy()
    df['month_date'] = pd.to_datetime(df['ym'] + '-01')

    left = (df[['permno', 'month_date']].drop_duplicates()
            .sort_values(['month_date', 'permno']).reset_index(drop=True))
    right = (fq[['permno', 'report_date', 'io_share']].dropna(subset=['io_share'])
             .sort_values(['report_date', 'permno']).reset_index(drop=True))
    tv = pd.merge_asof(left, right, left_on='month_date',
                       right_on='report_date', by='permno',
                       direction='backward')
    tv = tv.rename(columns={'io_share': 'io_share_tv'})
    df = df.merge(tv[['permno', 'month_date', 'io_share_tv']],
                  on=['permno', 'month_date'], how='left')
    df = df.rename(columns={'io_share_tv': 'io_share'})

    persist = fq.groupby('permno').agg(
        io_share_persist=('io_share', 'mean'),
        n_io_quarters=('quarter', 'nunique')).reset_index()
    df = df.merge(persist, on='permno', how='left')

    if verbose:
        cov_tv = df['io_share'].notna().sum()
        cov_p = df['io_share_persist'].notna().sum()
        print(f"  time-varying IO coverage: {cov_tv:,} firm-months "
              f"({100.0*cov_tv/len(df):.1f}% of {len(df):,})")
        print(f"  persistent IO coverage:   {cov_p:,} firm-months "
              f"({100.0*cov_p/len(df):.1f}%), "
              f"{df.loc[df['io_share_persist'].notna(),'permno'].nunique()} "
              f"permnos")
        med_q = df.groupby('permno')['n_io_quarters'].first().median()
        print(f"  median n_io_quarters per covered permno: {med_q}")
        # temporal coverage of the time-varying measure
        tvcov = df[df['io_share'].notna()].groupby(
            df['date'].dt.year).size()
        print(f"  time-varying coverage by year (first/last): "
              f"{tvcov.index.min()}={tvcov.iloc[0]:,} ... "
              f"{tvcov.index.max()}={tvcov.iloc[-1]:,}")
    return df


if __name__ == '__main__':
    df = load_panel()
    print(f"panel: {len(df):,} firm-months, {df['permno'].nunique()} permnos")
    fq = load_s34_firmquarter()
    print(f"Thomson s34 firm-quarter: {len(fq):,} rows, "
          f"{fq['permno'].nunique()} permnos, "
          f"{fq['quarter'].nunique()} quarters "
          f"({sorted(fq['quarter'].unique())[0]}.."
          f"{sorted(fq['quarter'].unique())[-1]})")
    print("\nio_share_raw distribution:")
    print(fq['io_share_raw'].describe().to_string())
    n_over = int((fq['io_share_raw'] > 1.0).sum())
    print(f"io_share_raw > 1.0: {n_over:,} "
          f"({100.0*n_over/len(fq):.2f}% of firm-quarters)")
    print("\nper-quarter diagnostics (head/tail):")
    diag = io_share_distribution(fq)
    print(diag.head(8).to_string(index=False))
    print("...")
    print(diag.tail(8).to_string(index=False))
    print()
    dfm = merge_io_to_panel(df, fq)
