# =====================================================================
# deepen_thomson_s34_download.py
# Builds the real 60-quarter (2009Q1-2023Q4) Thomson s34 13F institutional-ownership panel from WRDS: sums shares across managers by cusip, links cusip->permno via crsp.stocknames, and divides by crsp.msf shares outstanding; self-healing on WRDS connection drops.
#
# Inputs:    WRDS tr_13f.s34, crsp.stocknames, crsp.msf (via utils.wrds_client)
# Outputs:   per-quarter code/empirical/_thomson_s34_{YYYYQQ}.parquet; concatenated code/empirical/_thomson_s34_firmquarter.parquet
# Paper:     SUPERSEDED by deepen_thomson_s34_download_v6.py — kept for provenance (pre-denominator-correction; feeds the IA Thomson s34 panel construction section)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 (WRDS-recovered re-fire) — item (a).

Build the REAL Thomson Reuters s34 quarterly 13F institutional-ownership panel,
the canonical data source the round-1 referees demanded in place of the EDGAR
proxy.

For each rdate quarter 2009Q1-2023Q4 (60 quarters):
  1. Aggregate tr_13f.s34 by cusip: sum `shares` across all managers -> total
     institutional shares held for that (cusip, rdate). Filter shares > 0.
  2. Join cusip -> permno via crsp.stocknames on the 8-digit historical cusip
     (ncusip) valid at rdate. Restrict to common equity (shrcd in 10,11).
  3. Join shares-outstanding denominator from crsp.msf at the rdate month
     (month-end shrout in 1000s, contemporaneous with the holdings date).
  4. IO_share = institutional shares held / (shrout * 1000).

Each completed quarter is cached incrementally to
code/empirical/_thomson_s34_{YYYYQQ}.parquet so a connection drop loses at most
one quarter.

SELF-HEALING: the WRDS connection drops periodically and the server process
itself can fail to reconnect on the first try (transient WRDS-side rejection,
which makes wrds.Connection fall back to an interactive prompt and EOF-crash).
On any query failure this script kills the dead wrds_server and restarts it,
RETRYING the launch up to SERVER_LAUNCH_RETRIES times, then retries the same
quarter. The incremental cache means a restart never loses completed work.

Output: code/empirical/_thomson_s34_{YYYYQQ}.parquet (per quarter) and
        code/empirical/_thomson_s34_firmquarter.parquet (concatenated panel).
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

MAX_RETRIES = 8
SERVER_LAUNCH_RETRIES = 6

# 60 quarter-end report dates 2009Q1-2023Q4
QUARTER_ENDS = []
for yr in range(2009, 2024):
    for mm, q in [('03-31', 'Q1'), ('06-30', 'Q2'),
                  ('09-30', 'Q3'), ('12-31', 'Q4')]:
        QUARTER_ENDS.append((f"{yr}-{mm}", f"{yr}{q}"))


def quarter_sql(rdate):
    """s34 holdings aggregated by cusip, joined to permno and shrout."""
    yr, mm, _ = rdate.split('-')
    return f"""
    WITH s AS (
        SELECT cusip,
               SUM(shares)          AS inst_shares,
               COUNT(DISTINCT mgrno) AS n_mgr
        FROM tr_13f.s34
        WHERE rdate = DATE '{rdate}'
          AND cusip IS NOT NULL
          AND shares > 0
        GROUP BY cusip
    ),
    link AS (
        SELECT s.cusip, s.inst_shares, s.n_mgr,
               sn.permno, sn.shrcd, sn.exchcd
        FROM s
        JOIN crsp.stocknames sn
          ON s.cusip = sn.ncusip
         AND DATE '{rdate}' BETWEEN sn.namedt AND sn.nameenddt
        WHERE sn.shrcd IN (10, 11)
    )
    SELECT link.permno, link.cusip, link.inst_shares, link.n_mgr,
           link.shrcd, link.exchcd,
           m.shrout, ABS(m.prc) AS prc
    FROM link
    JOIN crsp.msf m
      ON link.permno = m.permno
     AND m.date BETWEEN DATE '{yr}-{mm}-01' AND DATE '{rdate}'
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
    """Kill the dead wrds_server and start a fresh one; retry the launch
    SERVER_LAUNCH_RETRIES times because wrds.Connection transiently falls back
    to an interactive prompt (and EOF-crashes) on a WRDS-side rejection."""
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
                break  # process died — retry the launch
        print(f"  [server] launch attempt {attempt} failed; retrying...",
              flush=True)
    print("  [server] FAILED to come back after "
          f"{SERVER_LAUNCH_RETRIES} launch attempts", flush=True)
    return False


def query_with_heal(sql, label):
    """Run a query; on connection-drop failure, restart the server and retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return wrds_query(sql, timeout=300)
        except Exception as e:
            msg = str(e)
            transient = ("server closed the connection" in msg
                         or "invalid transaction" in msg
                         or "Connection refused" in msg
                         or "OperationalError" in msg
                         or "EOF detected" in msg
                         or "Broken pipe" in msg
                         or "ConnectionResetError" in msg)
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
    print(f"=== Thomson s34 13F panel build: {len(QUARTER_ENDS)} quarters "
          f"2009Q1-2023Q4 ===", flush=True)
    done = set()
    for cached in glob.glob(os.path.join(EMP, "_thomson_s34_*Q*.parquet")):
        base = os.path.basename(cached)
        if base == "_thomson_s34_firmquarter.parquet":
            continue
        done.add(base.replace("_thomson_s34_", "").replace(".parquet", ""))
    print(f"  already cached: {len(done)} quarters", flush=True)

    for rdate, qlabel in QUARTER_ENDS:
        if qlabel in done:
            continue
        path = os.path.join(EMP, f"_thomson_s34_{qlabel}.parquet")
        t0 = time.time()
        try:
            df = query_with_heal(quarter_sql(rdate), qlabel)
        except Exception as e:
            print(f"  [{qlabel}] GAVE UP after retries: {e}", flush=True)
            print("  -- continuing to next quarter (cache preserves progress)",
                  flush=True)
            continue
        if len(df) == 0:
            print(f"  [{qlabel}] EMPTY — skipping", flush=True)
            continue
        df = df.dropna(subset=['permno', 'inst_shares', 'shrout'])
        df = df[df['shrout'] > 0]
        agg = df.groupby('permno').agg(
            inst_shares=('inst_shares', 'sum'),
            n_mgr=('n_mgr', 'max'),
            n_cusip=('cusip', 'nunique'),
            shrout=('shrout', 'first'),
            prc=('prc', 'first')).reset_index()
        agg['io_share_raw'] = agg['inst_shares'] / (agg['shrout'] * 1000.0)
        agg['rdate'] = rdate
        agg['quarter'] = qlabel
        n_over = int((agg['io_share_raw'] > 1.0).sum())
        agg.to_parquet(path, index=False)
        print(f"  [{qlabel}] {len(agg):,} permnos | "
              f"io>1.0: {n_over} ({100.0*n_over/len(agg):.2f}%) | "
              f"median io {agg['io_share_raw'].median():.3f} | "
              f"{time.time()-t0:.1f}s", flush=True)

    files = sorted(glob.glob(os.path.join(EMP, "_thomson_s34_*Q*.parquet")))
    files = [f for f in files
             if os.path.basename(f) != "_thomson_s34_firmquarter.parquet"]
    if not files:
        print("  NO quarters cached — aborting", flush=True)
        return
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = os.path.join(EMP, "_thomson_s34_firmquarter.parquet")
    panel.to_parquet(out, index=False)
    print(f"\n=== firm-quarter panel: {len(panel):,} rows, "
          f"{panel['permno'].nunique()} permnos, "
          f"{panel['quarter'].nunique()} quarters ===", flush=True)
    print("  io_share_raw distribution:", flush=True)
    print(panel['io_share_raw'].describe().to_string(), flush=True)
    n_over = int((panel['io_share_raw'] > 1.0).sum())
    print(f"  io_share_raw > 1.0: {n_over:,} ({100.0*n_over/len(panel):.2f}%)",
          flush=True)
    print(f"  wrote {out}", flush=True)


if __name__ == '__main__':
    main()
