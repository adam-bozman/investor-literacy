# =====================================================================
# deepen_thomson_io_panel_v6.py
# Live IO panel builder: loads the denominator-corrected Thomson s34 firm-quarter panel and merges it to the firm-month panel (no-look-ahead LAG_MONTHS=4), building corrected and v5-method time-varying + persistent IO measures.
#
# Inputs:    output/seed/data/processed/panel_corrected_standardized.parquet; code/empirical/_thomson_s34_v6_firmquarter.parquet (corrected WRDS Thomson s34)
# Outputs:   in-memory merged panel + diagnostics (no files written); printed diagnostics when run as __main__
# Paper:     Feeds T2 tab:headline / T3 tab:size_ctrl / T4 tab:migration (via deepen_13f_split_v6); IA Thomson s34 panel construction section
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r2 (Gate 5 Reject) — Item 3: IO panel builder on the
DENOMINATOR-CORRECTED Thomson Reuters s34 13F panel.

Consumes code/empirical/_thomson_s34_v6_firmquarter.parquet (built by
deepen_thomson_s34_download_v6.py — amendment-deduped, institutional shares =
sole+shared) and merges it to the firm-month panel.

Identical structure to deepen_thomson_io_panel.py (v5), so deepen_estimators.py
and the v5 split logic run unchanged. Only the IO data source is swapped:
v5's `_thomson_s34_firmquarter.parquet` (io_share_raw from `shares`, summed over
all amendment rows; 6.49% >1.0) -> v6's corrected panel (io_share_raw from
sole+shared, amendment-deduped; ~0.2% >1.0).

Two IO measures (unchanged definitions from v5):
  io_share         — TIME-VARYING, no look-ahead. merge_asof backward, rdate
                     lagged LAG_MONTHS=4 (the most recent 13F quarter already
                     public, plus a full extra quarter of staleness).
  io_share_persist — time-mean of io_share_raw (capped at 1.0) across all the
                     firm's available 13F quarters.

ALSO builds the v5-method measures (io_share_v5method / io_share_persist_v5method
from io_share_raw_v5method) so the split can be re-run on the OLD numerator for
the insensitivity comparison the referee asked for.

The residual io_share_raw>1.0 (~0.2%, the irreducible cross-manager
shared-block measurement-noise floor) is still capped at 1.0 — but it is now a
"few tenths of a percent" residual, exactly the level the referee says is
acceptable, not the 6.49% that was capped in v5.
"""

import os
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed",
                     "panel_corrected_standardized.parquet")
S34_FQ = os.path.join(EMP, "_thomson_s34_v6_firmquarter.parquet")

LAG_MONTHS = 4


def load_panel():
    df = pd.read_parquet(PANEL)
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    return df


def load_s34_firmquarter():
    if not os.path.exists(S34_FQ):
        raise FileNotFoundError(
            f"Corrected Thomson s34 firm-quarter panel not found: {S34_FQ}. "
            f"Run deepen_thomson_s34_download_v6.py.")
    fq = pd.read_parquet(S34_FQ)
    fq['io_share'] = fq['io_share_raw'].clip(upper=1.0)
    fq['io_share_v5method'] = fq['io_share_raw_v5method'].clip(upper=1.0)
    fq['rdate'] = pd.to_datetime(fq['rdate'])
    return fq


def io_share_distribution(fq):
    """Per-quarter coverage and io>1.0 rate for both the corrected (sole+shared)
    and the v5-method (shares, amendment-deduped) numerators."""
    g = fq.groupby('quarter')
    diag = g.agg(
        n_permnos=('permno', 'nunique'),
        median_io_raw=('io_share_raw', 'median'),
        mean_io_raw=('io_share_raw', 'mean'),
    ).reset_index()
    over = g.apply(lambda x: 100.0 * (x['io_share_raw'] > 1.0).mean(),
                   include_groups=False).rename('pct_over_one')
    over_v5 = g.apply(
        lambda x: 100.0 * (x['io_share_raw_v5method'] > 1.0).mean(),
        include_groups=False).rename('pct_over_one_v5method')
    diag = diag.merge(over.reset_index(), on='quarter')
    diag = diag.merge(over_v5.reset_index(), on='quarter')
    return diag


def merge_io_to_panel(df, fq, verbose=True):
    """Merge firm-quarter Thomson IO to the firm-month panel — corrected
    (sole+shared) AND v5-method (shares) numerators, time-varying + persistent.

    io_share / io_share_persist                — corrected numerator
    io_share_v5method / io_share_persist_v5m   — v5 numerator (insensitivity)
    n_io_quarters                              — count of 13F quarters / firm
    """
    fq = fq.copy()
    fq['report_date'] = fq['rdate'] + pd.DateOffset(months=LAG_MONTHS)
    df = df.copy()
    df['month_date'] = pd.to_datetime(df['ym'] + '-01')

    left = (df[['permno', 'month_date']].drop_duplicates()
            .sort_values(['month_date', 'permno']).reset_index(drop=True))

    # --- time-varying, corrected numerator ---
    right = (fq[['permno', 'report_date', 'io_share']].dropna(subset=['io_share'])
             .sort_values(['report_date', 'permno']).reset_index(drop=True))
    tv = pd.merge_asof(left, right, left_on='month_date',
                       right_on='report_date', by='permno',
                       direction='backward')
    tv = tv.rename(columns={'io_share': 'io_share_tv'})

    # --- time-varying, v5-method numerator ---
    right_v5 = (fq[['permno', 'report_date', 'io_share_v5method']]
                .dropna(subset=['io_share_v5method'])
                .sort_values(['report_date', 'permno']).reset_index(drop=True))
    tv_v5 = pd.merge_asof(left, right_v5, left_on='month_date',
                          right_on='report_date', by='permno',
                          direction='backward')
    tv_v5 = tv_v5.rename(columns={'io_share_v5method': 'io_share_tv_v5method'})

    df = df.merge(tv[['permno', 'month_date', 'io_share_tv']],
                  on=['permno', 'month_date'], how='left')
    df = df.merge(tv_v5[['permno', 'month_date', 'io_share_tv_v5method']],
                  on=['permno', 'month_date'], how='left')
    df = df.rename(columns={'io_share_tv': 'io_share',
                            'io_share_tv_v5method': 'io_share_v5method'})

    persist = fq.groupby('permno').agg(
        io_share_persist=('io_share', 'mean'),
        io_share_persist_v5method=('io_share_v5method', 'mean'),
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
        tvcov = df[df['io_share'].notna()].groupby(df['date'].dt.year).size()
        print(f"  time-varying coverage by year (first/last): "
              f"{tvcov.index.min()}={tvcov.iloc[0]:,} ... "
              f"{tvcov.index.max()}={tvcov.iloc[-1]:,}")
    return df


if __name__ == '__main__':
    df = load_panel()
    print(f"panel: {len(df):,} firm-months, {df['permno'].nunique()} permnos")
    fq = load_s34_firmquarter()
    print(f"corrected Thomson s34 firm-quarter: {len(fq):,} rows, "
          f"{fq['permno'].nunique()} permnos, "
          f"{fq['quarter'].nunique()} quarters "
          f"({sorted(fq['quarter'].unique())[0]}.."
          f"{sorted(fq['quarter'].unique())[-1]})")
    print("\nio_share_raw (sole+shared, corrected) distribution:")
    print(fq['io_share_raw'].describe().to_string())
    n_over = int((fq['io_share_raw'] > 1.0).sum())
    n_over_v5 = int((fq['io_share_raw_v5method'] > 1.0).sum())
    print(f"io_share_raw (corrected) > 1.0: {n_over:,} "
          f"({100.0*n_over/len(fq):.3f}% of firm-quarters)")
    print(f"io_share_raw_v5method (shares, amendment-deduped) > 1.0: "
          f"{n_over_v5:,} ({100.0*n_over_v5/len(fq):.3f}%)")
    print("[v5 ORIGINAL panel had 6.485% — see empirical_analysis_v5.md]")
    print("\nper-quarter diagnostics (head/tail):")
    diag = io_share_distribution(fq)
    print(diag.head(6).to_string(index=False))
    print("...")
    print(diag.tail(6).to_string(index=False))
    print()
    dfm = merge_io_to_panel(df, fq)
