# =====================================================================
# deepen_thomson_s34_download_v6.py
# Builds the denominator-corrected 60-quarter Thomson s34 13F panel from WRDS: amendment-deduped (latest fdate per mgr-cusip), per-manager numerator LEAST(LEAST(sole+shared, shares), shrout), crsp.msf_v2 denominator; drops the io>1.0 rate from 6.49% to ~0.2%.
#
# Inputs:    WRDS tr_13f.s34, crsp.stocknames, crsp.msf_v2 (via utils.wrds_client)
# Outputs:   per-quarter code/empirical/_thomson_s34_v6_{YYYYQQ}.parquet; concatenated code/empirical/_thomson_s34_v6_firmquarter.parquet
# Paper:     Live IO source feeding T2 tab:headline / T3 tab:size_ctrl / T4 tab:migration (via deepen_thomson_io_panel_v6 + deepen_13f_split_v6); IA Thomson s34 panel construction section
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r2 (Gate 5 Reject) — Item 3: resolve the Thomson s34
io_share>1.0 denominator problem AT THE SOURCE.

================================ DIAGNOSIS ================================
The v5 download (deepen_thomson_s34_download.py) had io_share>1.0 for 6.49% of
firm-quarters, capped at 1.0 rather than resolved. Diagnosed on 5 quarters
(2009Q1, 2013Q2, 2016Q2, 2019Q3, 2023Q4) via code/tmp/diag_s34_*.py.

Ruled OUT: multi-share-class permnos and cusip->permno fan-out. Every permno
maps to exactly one valid ncusip at rdate; no cusip maps to >1 permno.

Three real root causes, in order of contribution:

  (1) AMENDMENT DOUBLE-COUNTING. tr_13f.s34 stacks original AND amended 13F
      filings: a manager that files an original then an amendment for the same
      rdate appears as 2-4 rows for the same (mgrno,cusip,rdate), each with its
      own fdate (5k-28k such pairs per quarter). v5 did SUM(shares) over ALL
      rows -> amended managers counted 2-4x. FIX: DISTINCT ON (mgrno,cusip)
      ORDER BY fdate DESC -- keep only the latest filing per manager-stock.

  (2) THE `shares` FIELD OVERSTATES INSTITUTIONAL HOLDINGS. The s34 `shares`
      field, summed across managers, exceeds shares outstanding for 3-8% of
      firm-quarters even after amendment dedup -- cross-manager double-counting
      of jointly-held / block positions. The authority decomposition
      sole+shared (the literature-standard institutional-ownership numerator;
      Lewellen 2011; Ben-David, Franzoni, Moussawi, Sedunov 2021) deflates the
      bulk of this: it drops the >1.0 rate to ~0.1-1.0% per quarter.

  (3) BUT `sole+shared` IS ITSELF CORRUPT for ~200-900 manager-stock rows
      (a units bug -- sole/shared run 10-75x the `shares` value, producing
      stock-level io up to 343, concentrated in high-priced names and in 2023).
      FIX: take MIN(sole+shared, shares) per manager -- this uses sole+shared
      where it correctly deflates block-sharing, and falls back to `shares`
      where sole+shared is corrupt-large. min() is conservative on both
      failure modes.

  Plus: (4) a handful of stale crsp.msf shrout values -> use crsp.msf_v2 (the
  current CRSP monthly product; identical to crsp.msf shrout for 99.99%+ of
  firm-months, fixes the stale cases). (5) Per-manager cap at shrout (no single
  manager can hold >100%) catches any residual single-manager data-entry error.

================================== THE FIX ==================================
Per manager-stock-quarter (amendment-deduped):
    h = LEAST( LEAST(sole+shared, shares), shrout )
institutional shares held = SUM_managers h
io_share_raw = institutional shares held / (shrout * 1000)
Residual io_share_raw > 1.0 (now ~0.1-0.7% per quarter, the irreducible
cross-manager measurement-noise floor) is capped at 1.0 for the final measure.

Verified on the 5 diagnostic quarters: io>1.0 rate 0.09% / 0.31% / 0.28% /
0.46% / 0.71% -- the "few tenths of a percent" the r2 referee says is
acceptable, vs the 6.49% capped in v5.

Also carries io_share_raw_v5method (the v5 `shares` numerator, amendment-
deduped) so the 13F split can be re-run on the OLD numerator for the
insensitivity comparison the referee asked for.

Output: code/empirical/_thomson_s34_v6_{YYYYQQ}.parquet (per quarter) and
        code/empirical/_thomson_s34_v6_firmquarter.parquet (concatenated).
Self-healing on WRDS connection drop. Incremental per-quarter cache.
"""

import os
import sys
import glob
import time
import signal
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "code"))
from utils.wrds_client import wrds_query, wrds_start, wrds_ping

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
UTILS = os.path.join(ROOT, "code", "utils")
PID_FILE = os.path.join(UTILS, ".wrds_server.pid")
SERVER = os.path.join(UTILS, "wrds_server.py")

MAX_RETRIES = 12
SERVER_LAUNCH_RETRIES = 8

QUARTER_ENDS = []
for yr in range(2009, 2024):
    for mm, q in [('03-31', 'Q1'), ('06-30', 'Q2'),
                  ('09-30', 'Q3'), ('12-31', 'Q4')]:
        QUARTER_ENDS.append((f"{yr}-{mm}", f"{yr}{q}"))


def quarter_sql(rdate):
    """Amendment-deduped s34 holdings; per-manager numerator
    LEAST(LEAST(sole+shared, shares), shrout); aggregated by permno; shrout from
    crsp.msf_v2 at the rdate quarter-end month."""
    yr, mm, _ = rdate.split('-')
    return f"""
    WITH raw AS (
        SELECT mgrno, cusip, fdate, shares,
               COALESCE(sole,0) + COALESCE(shared,0) AS solesh
        FROM tr_13f.s34
        WHERE rdate = DATE '{rdate}'
          AND cusip IS NOT NULL
          AND shares > 0
    ),
    -- amendment dedup: keep only the latest filing per (mgrno, cusip)
    latest AS (
        SELECT DISTINCT ON (mgrno, cusip)
               mgrno, cusip, shares, solesh
        FROM raw
        ORDER BY mgrno, cusip, fdate DESC
    ),
    link AS (
        SELECT l.mgrno, l.cusip, l.shares, l.solesh,
               sn.permno, sn.shrcd, sn.exchcd
        FROM latest l
        JOIN crsp.stocknames sn
          ON l.cusip = sn.ncusip
         AND DATE '{rdate}' BETWEEN sn.namedt AND sn.nameenddt
        WHERE sn.shrcd IN (10, 11)
    ),
    shr AS (
        SELECT permno, MAX(shrout) * 1000.0 AS shrout_sh
        FROM crsp.msf_v2
        WHERE mthcaldt BETWEEN DATE '{yr}-{mm}-01' AND DATE '{rdate}'
          AND shrout > 0
        GROUP BY permno
    ),
    mgr AS (
        SELECT link.permno, link.mgrno, shr.shrout_sh, link.exchcd,
               -- FINAL per-manager numerator: min(sole+shared, shares),
               -- then cap at shares outstanding
               LEAST(LEAST(link.solesh, link.shares), shr.shrout_sh)
                   AS h_final,
               -- v5-method per-manager numerator: just `shares`
               link.shares AS h_shares_v5
        FROM link
        JOIN shr ON link.permno = shr.permno
    )
    SELECT permno,
           SUM(h_final)        AS inst_final,
           SUM(h_shares_v5)    AS inst_shares_v5,
           MAX(shrout_sh)      AS shrout_sh,
           MAX(exchcd)         AS exchcd,
           COUNT(*)            AS n_mgr
    FROM mgr
    GROUP BY permno
    """


def _kill_old_server():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
            except Exception:
                pass
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        except Exception:
            pass
        try:
            os.remove(PID_FILE)
        except Exception:
            pass


def restart_server():
    print("  [server] restarting wrds_server...", flush=True)
    env = {**os.environ}
    wp = os.getenv("WRDS_PASS")
    if wp:
        env["PGPASSWORD"] = wp
        env["WRDS_PASS"] = wp
    wu = os.getenv("WRDS_USER")
    if wu:
        env["WRDS_USER"] = wu
    for attempt in range(1, SERVER_LAUNCH_RETRIES + 1):
        _kill_old_server()
        time.sleep(2)
        proc = subprocess.Popen(
            [sys.executable, SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True, env=env)
        for _ in range(40):
            time.sleep(1.5)
            if wrds_ping():
                print(f"  [server] back up (launch attempt {attempt}).",
                      flush=True)
                return True
            if proc.poll() is not None:
                break
        print(f"  [server] launch attempt {attempt} failed; retrying...",
              flush=True)
    print("  [server] FAILED to come back after "
          f"{SERVER_LAUNCH_RETRIES} launch attempts", flush=True)
    return False


def query_with_heal(sql, label):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return wrds_query(sql, timeout=300)
        except Exception as e:
            msg = str(e)
            transient = ("server closed the connection" in msg
                         or "invalid transaction" in msg
                         or "Connection refused" in msg
                         or "actively refused" in msg
                         or "OperationalError" in msg
                         or "EOF detected" in msg
                         or "Broken pipe" in msg
                         or "ConnectionResetError" in msg
                         or "WinError 10061" in msg)
            print(f"  [{label}] attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{msg[:120]}", flush=True)
            if not transient or attempt == MAX_RETRIES:
                raise
            if not restart_server():
                raise RuntimeError(f"{label}: server restart failed")
    raise RuntimeError(f"{label}: exhausted retries")


def main():
    if not wrds_ping():
        wrds_start()
    print(f"=== Thomson s34 13F panel build V6 (denominator fix): "
          f"{len(QUARTER_ENDS)} quarters 2009Q1-2023Q4 ===", flush=True)
    print("  FIX: amendment-dedup + per-mgr numerator "
          "min(sole+shared, shares) capped at shrout; crsp.msf_v2 denominator",
          flush=True)
    done = set()
    for cached in glob.glob(os.path.join(EMP, "_thomson_s34_v6_*Q*.parquet")):
        base = os.path.basename(cached)
        if base == "_thomson_s34_v6_firmquarter.parquet":
            continue
        done.add(base.replace("_thomson_s34_v6_", "").replace(".parquet", ""))
    print(f"  already cached: {len(done)} quarters", flush=True)

    for rdate, qlabel in QUARTER_ENDS:
        if qlabel in done:
            continue
        path = os.path.join(EMP, f"_thomson_s34_v6_{qlabel}.parquet")
        t0 = time.time()
        try:
            agg = query_with_heal(quarter_sql(rdate), qlabel)
        except Exception as e:
            print(f"  [{qlabel}] GAVE UP after retries: {e}", flush=True)
            print("  -- continuing to next quarter (cache preserves progress)",
                  flush=True)
            continue
        if len(agg) == 0:
            print(f"  [{qlabel}] EMPTY — skipping", flush=True)
            continue
        agg = agg.dropna(subset=['permno', 'inst_final', 'shrout_sh'])
        agg = agg[agg['shrout_sh'] > 0]
        # corrected measure
        agg['io_share_raw'] = agg['inst_final'] / agg['shrout_sh']
        # v5-method measure (shares numerator; amendment-deduped here)
        agg['io_share_raw_v5method'] = agg['inst_shares_v5'] / agg['shrout_sh']
        agg['rdate'] = rdate
        agg['quarter'] = qlabel
        n_over = int((agg['io_share_raw'] > 1.0).sum())
        n_over_v5 = int((agg['io_share_raw_v5method'] > 1.0).sum())
        agg.to_parquet(path, index=False)
        print(f"  [{qlabel}] {len(agg):,} permnos | "
              f"io>1.0 (FINAL): {n_over} ({100.0*n_over/len(agg):.2f}%) | "
              f"io>1.0 (v5 shares): {n_over_v5} "
              f"({100.0*n_over_v5/len(agg):.2f}%) | "
              f"median io {agg['io_share_raw'].median():.3f} | "
              f"max {agg['io_share_raw'].max():.2f} | "
              f"{time.time()-t0:.1f}s", flush=True)

    files = sorted(glob.glob(os.path.join(EMP, "_thomson_s34_v6_*Q*.parquet")))
    files = [f for f in files
             if os.path.basename(f) != "_thomson_s34_v6_firmquarter.parquet"]
    if len(files) < len(QUARTER_ENDS):
        print(f"  WARNING: only {len(files)}/{len(QUARTER_ENDS)} quarters "
              f"cached — re-run to fill gaps before concatenating", flush=True)
        return
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = os.path.join(EMP, "_thomson_s34_v6_firmquarter.parquet")
    panel.to_parquet(out, index=False)
    print(f"\n=== firm-quarter panel V6: {len(panel):,} rows, "
          f"{panel['permno'].nunique()} permnos, "
          f"{panel['quarter'].nunique()} quarters ===", flush=True)
    print("  io_share_raw (FINAL) distribution:", flush=True)
    print(panel['io_share_raw'].describe().to_string(), flush=True)
    n_over = int((panel['io_share_raw'] > 1.0).sum())
    n_over_v5 = int((panel['io_share_raw_v5method'] > 1.0).sum())
    print(f"  io_share_raw (FINAL) > 1.0: {n_over:,} "
          f"({100.0*n_over/len(panel):.3f}%)", flush=True)
    print(f"  io_share_raw_v5method (shares, amendment-deduped) > 1.0: "
          f"{n_over_v5:,} ({100.0*n_over_v5/len(panel):.3f}%)", flush=True)
    print(f"  [v5 ORIGINAL for reference: 6.485%]", flush=True)
    print(f"  wrote {out}", flush=True)


if __name__ == '__main__':
    main()
