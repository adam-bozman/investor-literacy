# =====================================================================
# deepen_13f_download.py
# Downloads SEC EDGAR 13F-HR filings for three quarters (2011Q2/2015Q2/2019Q2), parses infotables, and aggregates institutional shares held per ticker/CUSIP into a persistent IO proxy.
#
# Inputs:    SEC EDGAR 13F-HR filings (via edgartools/get_filings); SEC identity from .env (SEC_EDGAR_NAME/EMAIL)
# Outputs:   code/empirical/_edgar_13f_cache.parquet; per-quarter _edgar_13f_{Q}_ticker.parquet and _edgar_13f_{Q}_cusip.parquet; _edgar_13f_download.log
# Paper:     SUPERSEDED by deepen_13f_download_v4.py — kept for provenance (round-0 EDGAR-proxy; feeds the IA EDGAR-proxy cross-check section)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive Item 6 (part 1/2) — EDGAR 13F download.

The canonical 13F source (WRDS Thomson Reuters s34) is unavailable this session.
OSAP has no clean raw institutional-ownership-SHARE signal (its IO signals are all
conditional/residualized constructs: RIO_*, Activism*, DelBreadth is a flow, etc).
So we route through SEC EDGAR (edgartools), per deepen directive item 6a.

A full 2009-2023 13F pull is ~20 hours of downloading (4,000+ 13F-HR filings per
quarter x 60 quarters) — not feasible to time-box this session. Instead we build a
PERSISTENT institutional-ownership measure from THREE quarters spread across the
sample (2011Q2, 2015Q2, 2019Q2): institutional ownership is one of the most
persistent firm characteristics, so a 3-quarter average is a defensible split
variable for the full panel. Each quarter is ~23 min; ~70 min total. Cached.

For each quarter:
  - download all 13F-HR filings, parse the infotable (CUSIP, Ticker, shares),
  - keep Shares-type rows, exclude option positions (PutCall non-null),
  - aggregate institutional shares held by Ticker,
  - the firm-quarter institutional share is computed in the analysis script
    (part 2/2) as inst_shares / (shrout * 1000), matched to the panel by ticker.

Output: code/empirical/_edgar_13f_cache.parquet
        (columns: quarter, Ticker, inst_shares, n_managers)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
np.random.seed(42)

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
CACHE = os.path.join(ROOT, "code", "empirical", "_edgar_13f_cache.parquet")
LOG = os.path.join(ROOT, "code", "empirical", "_edgar_13f_download.log")

# three quarters spread across 2009-2023; holdings dated end of the filing
# quarter. 13F-HR filed in Q2 reports holdings as of end of Q1/Q2.
QUARTERS = [(2011, 2), (2015, 2), (2019, 2)]


def main():
    from edgar import set_identity, get_filings
    name = os.getenv('SEC_EDGAR_NAME', 'Research')
    email = os.getenv('SEC_EDGAR_EMAIL', 'research@university.edu')
    set_identity(f"{name} {email}")
    print(f"edgar identity: {name} {email}")

    if os.path.exists(CACHE):
        cached = pd.read_parquet(CACHE)
        have = set(cached['quarter'].unique())
        print(f"cache exists with quarters {sorted(have)}")
    else:
        cached = pd.DataFrame()
        have = set()

    logf = open(LOG, 'a', encoding='utf-8')

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    all_new = []
    for (yr, q) in QUARTERS:
        qlabel = f"{yr}Q{q}"
        if qlabel in have:
            log(f"[{qlabel}] already cached, skipping")
            continue
        log(f"[{qlabel}] downloading 13F-HR filings ...")
        t0 = time.time()
        filings = get_filings(form='13F-HR', year=yr, quarter=q)
        nfil = len(filings)
        log(f"[{qlabel}] {nfil} 13F-HR filings found")
        rows = []
        n_ok = 0
        n_err = 0
        for i in range(nfil):
            try:
                obj = filings[i].obj()
                it = getattr(obj, 'infotable', None)
                if it is None or len(it) == 0:
                    continue
                sub = it[['Ticker', 'Cusip', 'SharesPrnAmount', 'Type',
                          'PutCall']].copy()
                # shares only, exclude option positions
                sub = sub[(sub['Type'] == 'Shares')
                          & (sub['PutCall'].isna() | (sub['PutCall'] == ''))]
                sub['SharesPrnAmount'] = pd.to_numeric(
                    sub['SharesPrnAmount'], errors='coerce')
                sub = sub.dropna(subset=['SharesPrnAmount'])
                sub = sub[sub['SharesPrnAmount'] > 0]
                if len(sub):
                    sub['_manager'] = i
                    rows.append(sub[['Ticker', 'Cusip', 'SharesPrnAmount',
                                     '_manager']])
                    n_ok += 1
            except Exception as e:
                n_err += 1
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                log(f"[{qlabel}]   {i+1}/{nfil} parsed "
                    f"({n_ok} ok, {n_err} err), {el:.0f}s elapsed")
        if not rows:
            log(f"[{qlabel}] NO holdings parsed — skipping quarter")
            continue
        allh = pd.concat(rows, ignore_index=True)
        # aggregate institutional shares and manager count by ticker
        agg = allh.groupby('Ticker').agg(
            inst_shares=('SharesPrnAmount', 'sum'),
            n_managers=('_manager', 'nunique')).reset_index()
        agg['quarter'] = qlabel
        # also aggregate by CUSIP (8-digit, for a robustness match path)
        allh['cusip8'] = allh['Cusip'].astype(str).str[:8]
        agg_c = allh.groupby('cusip8').agg(
            inst_shares_cusip=('SharesPrnAmount', 'sum')).reset_index()
        agg_c['quarter'] = qlabel
        el = time.time() - t0
        log(f"[{qlabel}] DONE: {n_ok} filings with holdings, "
            f"{len(allh):,} share-rows, {len(agg):,} unique tickers, "
            f"{len(agg_c):,} unique cusip8, {el:.0f}s")
        # save cusip aggregation alongside
        agg.to_parquet(os.path.join(
            ROOT, "code", "empirical", f"_edgar_13f_{qlabel}_ticker.parquet"))
        agg_c.to_parquet(os.path.join(
            ROOT, "code", "empirical", f"_edgar_13f_{qlabel}_cusip.parquet"))
        all_new.append(agg)

    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        out = pd.concat([cached, new_df], ignore_index=True) if len(cached) \
            else new_df
        out.to_parquet(CACHE)
        log(f"cache written: {CACHE} ({len(out):,} rows, "
            f"quarters {sorted(out['quarter'].unique())})")
    else:
        log("no new quarters downloaded")
    logf.close()


if __name__ == '__main__':
    main()
