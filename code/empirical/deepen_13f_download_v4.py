# =====================================================================
# deepen_13f_download_v4.py
# Extends the EDGAR 13F-HR download to one quarter per calendar year 2009-2023, with source-level CUSIP9 dedup, ticker->CUSIP6 crosswalks, conservative SEC rate-limiting, and per-quarter time caps.
#
# Inputs:    SEC EDGAR 13F-HR filings (via edgartools/get_filings); SEC identity from .env (SEC_EDGAR_NAME/EMAIL)
# Outputs:   per-quarter code/empirical/_edgar_13f_{YYYYQQ}_cusip9.parquet and _edgar_13f_{YYYYQQ}_xwalk.parquet; _edgar_13f_v4_download.log
# Paper:     SUPERSEDED by the Thomson s34 panel (deepen_thomson_s34_download_v6.py) as the live IO source — kept for provenance; feeds the IA EDGAR-proxy cross-check section
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 Item (a) — EDGAR 13F panel extension (v4).

Builds a TIME-VARYING firm-quarter institutional-ownership panel from SEC EDGAR
13F-HR filings. Round-0 completed 2011Q2 and 2015Q2 (ticker+cusip8 aggregates only);
this script extends coverage targeting one quarter per calendar year 2009-2023.

Key fixes vs round-0 (deepen directive item a):
  - Aggregate at the SOURCE by CUSIP9 (issuer+issue): per-filer MAX SharesPrnAmount
    per cusip9 (joint-filing OtherManager rows repeat the lot, they do not add),
    then SUM across distinct filers. This kills the multi-class double-count that
    ticker-aggregation produced (round-0: 21.7% of io_share > 1.0).
  - Keep a Ticker->CUSIP6 crosswalk per quarter (share-weighted dominant cusip6
    per ticker) so the panel (ticker, no cusip) can be matched via the crosswalk.
  - Track distinct filer count per security.
  - Exclude option positions (PutCall non-null), keep Type == 'Shares' only,
    SharesPrnAmount > 0.

Caching: each completed quarter is written immediately to
  code/empirical/_edgar_13f_{YYYYQQ}_cusip9.parquet
  code/empirical/_edgar_13f_{YYYYQQ}_xwalk.parquet
A PER-QUARTER time cap (PER_QUARTER_CAP) means a slow quarter saves whatever it
parsed (partial) rather than blocking the rest — partial quarters are still a
valid IO snapshot (institutional ownership is highly persistent and the sample is
random across filers). A runtime cap on the whole loop (TIME_BUDGET) preserves
completed quarters.

Download order: MODERN-FIRST. Post-2013 quarters parse 5-10x faster (the 13F XML
mandate phased in ~2013; pre-2013 filings hit slow SGML/homepage fallbacks). We do
2013,2015,...,2023 then 2011, then the even years, then 2010/2012 — so even a
partial completion spans the sample well. 2009Q2 is already cached from the
first run; this script skips cached quarters.
"""

import os
import sys
import time
import gc
# SEC rate-limit: edgartools defaults to 9 req/sec which (with parallel runs)
# triggered an SEC block in the first pass. Be conservative — single-threaded,
# 3 req/sec — and set this BEFORE edgar is imported anywhere.
os.environ.setdefault("EDGAR_RATE_LIMIT_PER_SEC", "3")
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
np.random.seed(42)

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
EMP = os.path.join(ROOT, "code", "empirical")
LOG = os.path.join(EMP, "_edgar_13f_v4_download.log")

# MODERN-FIRST order: fast quarters first so a partial run still spans the sample.
QUARTERS = [
    # already cached: 2009Q2, 2010Q2, 2013Q2(partial). Fill the temporal gaps,
    # widest-spread-first so a partial run still spans 2009-2023.
    (2017, 2), (2021, 2), (2015, 2), (2019, 2), (2023, 2), (2011, 2),
    (2012, 2), (2014, 2), (2016, 2), (2018, 2), (2020, 2), (2022, 2),
]

# whole-loop wall-clock budget (seconds). leave headroom for items (b)-(d).
TIME_BUDGET = 9000
# per-quarter cap: if a quarter takes longer than this, save the partial parse
# and move on. modern quarters finish well under this; slow early years get
# capped with a partial (still a valid random sample of filers).
PER_QUARTER_CAP = 1500


def _aggregate(rows, qlabel, n_ok, log, t0, partial=False):
    """Aggregate parsed filer rows -> (cusip9_agg, xwalk)."""
    allh = pd.concat(rows, ignore_index=True)
    allh['Cusip'] = allh['Cusip'].astype(str).str.strip().str.upper()
    allh['cusip9'] = allh['Cusip'].str[:9]
    allh['cusip6'] = allh['Cusip'].str[:6]
    allh['Ticker'] = allh['Ticker'].astype(str).str.strip().str.upper()
    # SOURCE-LEVEL DEDUP: per filer, MAX shares per cusip9 (joint-filing repeats);
    # then SUM across distinct filers.
    per_filer = allh.groupby(['_filer', 'cusip9', 'cusip6'], as_index=False).agg(
        shares=('SharesPrnAmount', 'max'),
        Ticker=('Ticker', 'first'))
    cusip9_agg = per_filer.groupby('cusip9', as_index=False).agg(
        inst_shares=('shares', 'sum'),
        n_filers=('_filer', 'nunique'),
        cusip6=('cusip6', 'first'),
        Ticker=('Ticker', lambda s: s.mode().iloc[0] if len(s.mode()) else ''))
    cusip9_agg['quarter'] = qlabel
    cusip9_agg['partial'] = partial
    cusip6_agg = per_filer.groupby('cusip6', as_index=False).agg(
        inst_shares_c6=('shares', 'sum'),
        n_filers_c6=('_filer', 'nunique'))
    tk = per_filer[per_filer['Ticker'] != ''].copy()
    if len(tk):
        tk_w = tk.groupby(['Ticker', 'cusip6'], as_index=False)['shares'].sum()
        idx = tk_w.groupby('Ticker')['shares'].idxmax()
        xwalk = tk_w.loc[idx, ['Ticker', 'cusip6']].reset_index(drop=True)
    else:
        xwalk = pd.DataFrame(columns=['Ticker', 'cusip6'])
    xwalk['quarter'] = qlabel
    cusip9_agg = cusip9_agg.merge(
        cusip6_agg[['cusip6', 'inst_shares_c6', 'n_filers_c6']],
        on='cusip6', how='left')
    el = time.time() - t0
    tag = "PARTIAL" if partial else "DONE"
    log(f"[{qlabel}] {tag}: {n_ok} filings w/ holdings, {len(allh):,} share-rows, "
        f"{len(cusip9_agg):,} cusip9, {len(xwalk):,} tickers, {el:.0f}s")
    del allh, per_filer
    gc.collect()
    return cusip9_agg, xwalk


def parse_quarter(yr, q, log):
    """Download + parse one 13F-HR quarter, with a per-quarter time cap. Returns
    (cusip9_agg, xwalk, is_partial) or None."""
    from edgar import get_filings
    qlabel = f"{yr}Q{q}"
    t0 = time.time()
    try:
        filings = get_filings(form='13F-HR', year=yr, quarter=q)
    except Exception as e:
        log(f"[{qlabel}] get_filings FAILED: {e}")
        return None
    nfil = len(filings)
    log(f"[{qlabel}] {nfil} 13F-HR filings found")
    rows = []
    n_ok = 0
    n_err = 0
    capped = False
    for i in range(nfil):
        try:
            obj = filings[i].obj()
            it = getattr(obj, 'infotable', None)
            if it is None or len(it) == 0:
                continue
            cols = it.columns
            keep = ['Cusip', 'SharesPrnAmount', 'Type', 'PutCall']
            if 'Ticker' in cols:
                keep.append('Ticker')
            sub = it[keep].copy()
            sub = sub[(sub['Type'] == 'Shares')
                      & (sub['PutCall'].isna() | (sub['PutCall'] == ''))]
            sub['SharesPrnAmount'] = pd.to_numeric(
                sub['SharesPrnAmount'], errors='coerce')
            sub = sub.dropna(subset=['SharesPrnAmount'])
            sub = sub[sub['SharesPrnAmount'] > 0]
            if len(sub):
                sub['_filer'] = i
                if 'Ticker' not in sub.columns:
                    sub['Ticker'] = ''
                rows.append(sub[['Cusip', 'Ticker', 'SharesPrnAmount',
                                 '_filer']])
                n_ok += 1
        except Exception:
            n_err += 1
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            log(f"[{qlabel}]   {i+1}/{nfil} parsed "
                f"({n_ok} ok, {n_err} err), {el:.0f}s")
            if el > PER_QUARTER_CAP:
                log(f"[{qlabel}] per-quarter cap ({PER_QUARTER_CAP}s) hit at "
                    f"{i+1}/{nfil} — saving partial.")
                capped = True
                break
    if not rows:
        log(f"[{qlabel}] NO holdings parsed")
        return None
    cusip9_agg, xwalk = _aggregate(rows, qlabel, n_ok, log, t0, partial=capped)
    return cusip9_agg, xwalk, capped


def main():
    from edgar import set_identity
    name = os.getenv('SEC_EDGAR_NAME', 'Research')
    email = os.getenv('SEC_EDGAR_EMAIL', 'research@university.edu')
    set_identity(f"{name} {email}")

    logf = open(LOG, 'a', encoding='utf-8')

    def log(msg):
        ts = time.strftime('%H:%M:%S')
        line = f"{ts} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== v4 13F download (conservative 3 req/s) start; identity={name} ===")
    # initial cooldown: the first pass tripped an SEC rate-limit block; the SEC
    # penalises continued requests during the block. Wait it out before resuming.
    COOLDOWN = int(os.environ.get("EDGAR_COOLDOWN_SEC", "720"))
    log(f"initial cooldown {COOLDOWN}s to clear any SEC rate-limit block ...")
    time.sleep(COOLDOWN)
    log("cooldown done; resuming downloads")
    t_start = time.time()
    done, partial, skipped, failed = [], [], [], []

    for (yr, q) in QUARTERS:
        qlabel = f"{yr}Q{q}"
        c9_path = os.path.join(EMP, f"_edgar_13f_{qlabel}_cusip9.parquet")
        xw_path = os.path.join(EMP, f"_edgar_13f_{qlabel}_xwalk.parquet")
        if os.path.exists(c9_path) and os.path.exists(xw_path):
            log(f"[{qlabel}] already cached, skipping")
            skipped.append(qlabel)
            continue
        if time.time() - t_start > TIME_BUDGET:
            log(f"[{qlabel}] TIME BUDGET ({TIME_BUDGET}s) exceeded — stopping.")
            break
        res = parse_quarter(yr, q, log)
        if res is None:
            failed.append(qlabel)
            continue
        cusip9_agg, xwalk, is_partial = res
        cusip9_agg.to_parquet(c9_path)
        xwalk.to_parquet(xw_path)
        log(f"[{qlabel}] cached -> {os.path.basename(c9_path)}"
            f"{' (PARTIAL)' if is_partial else ''}")
        (partial if is_partial else done).append(qlabel)

    log(f"=== v4 download finished: {len(done)} full, {len(partial)} partial, "
        f"{len(skipped)} cached, {len(failed)} failed ===")
    log(f"full: {done}  partial: {partial}  cached: {skipped}  failed: {failed}")
    import glob
    have = sorted(os.path.basename(p).split('_')[3]
                  for p in glob.glob(
                      os.path.join(EMP, "_edgar_13f_*_cusip9.parquet")))
    log(f"ALL cusip9 quarters on disk: {have} ({len(have)} total)")
    logf.close()


if __name__ == '__main__':
    main()
