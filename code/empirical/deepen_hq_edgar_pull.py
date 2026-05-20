# =====================================================================
# deepen_hq_edgar_pull.py
# Pulls point-in-time HQ state for each panel firm from SEC EDGAR: the current business-address state (submissions JSON) and the business-address STATE in the SGML header of the first and last in-sample 10-K, to flag firms that relocated HQ during 2009-2023.
#
# Inputs:    code/empirical/_panel_cik_v6.parquet (permno->CIK); SEC EDGAR submissions JSON + 10-K filing-header text; SEC identity from .env (SEC_EDGAR_NAME/EMAIL)
# Outputs:   code/empirical/_hq_edgar_state_v6.parquet; checkpoint _hq_edgar_state_v6_ckpt.parquet
# Paper:     Intermediate data-build step feeding the IA PIT HQ reassignment / stable-HQ construction diagnostics (via deepen_hq_relocation_droptest.py)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r2 (Gate 5 Reject) — Task A support: pull point-in-time HQ
state from SEC EDGAR for the panel firms.

The seed panel's hq_state is a STATIC Compustat snapshot. The referee asks: which
firms relocated HQ during 2009-2023? This script pulls, for every panel firm with
a CIK:
  (1) the current SEC business-address state (submissions JSON — free, one
      request per firm);
  (2) the STATE in the SGML header of the FIRST and LAST 10-K filed 2009-2023.

The SEC filing-header business-address STATE is the canonical point-in-time HQ
location used in the HQ-bias literature (Pirinsky-Wang 2006; Coval-Moskowitz
1999). 10-K headers are the genuine PIT source: each carries the address as of
the filing date. A firm whose FIRST-vs-LAST 10-K-header STATE differs relocated
during the sample; comparing the LAST header STATE to the panel's static
snapshot catches firms whose move predates the first sample 10-K or whose
snapshot is simply stale.

PERFORMANCE NOTE. The first version of this script fetched EVERY 10-K header per
firm (~9-11 per firm) — ~5s/firm, ~10h for the full panel. This version fetches
only the FIRST and LAST in-sample 10-K header (2 fetches/firm) — ~1.3s/firm,
~2.7h. The first 300 firms were pulled by the original script with FULL header
sequences; those records are preserved as-is (richer data — every state change
datable), and the fast path is used for the remaining firms. The flag logic in
deepen_hq_relocation_droptest.py handles both: `hdr_changed` = any state change
across whatever headers a firm has; `reloc_year` is datable only where the full
sequence (or a first/last difference) pins it.

Output: code/empirical/_hq_edgar_state_v6.parquet
  columns: permno, cik, sec_state_current, n_10k, hdr_states (JSON list of
           [date, state]), hdr_first, hdr_last, hdr_changed (bool), reloc_year
"""
import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = ROOT + "/code/empirical"
NAME = os.getenv('SEC_EDGAR_NAME', 'Research')
EMAIL = os.getenv('SEC_EDGAR_EMAIL', 'research@university.edu')
HDR = {"User-Agent": f"{NAME} {EMAIL}"}
OUT = EMP + "/_hq_edgar_state_v6.parquet"
CKPT = EMP + "/_hq_edgar_state_v6_ckpt.parquet"

HDR_STATE_RE = re.compile(
    r"BUSINESS ADDRESS:.*?STATE:\s*([A-Z]{2})", re.DOTALL)


def get_submissions(cik10):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HDR, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.5 + attempt)
    return None


def get_10k_header_state(cik_int, accno):
    """Fetch only the SGML header of a filing; return business-address STATE."""
    acc_nodash = accno.replace('-', '')
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
           f"{acc_nodash}/{accno}.txt")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HDR, timeout=30, stream=True)
            if r.status_code != 200:
                time.sleep(0.5 + attempt)
                continue
            buf = ""
            for chunk in r.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk is None:
                    continue
                buf += chunk
                if "</SEC-HEADER>" in buf or len(buf) > 60000:
                    break
            r.close()
            m = HDR_STATE_RE.search(buf)
            return m.group(1) if m else None
        except Exception:
            time.sleep(0.6 + attempt)
    return None


def main():
    panel_cik = pd.read_parquet(EMP + "/_panel_cik_v6.parquet")
    panel_cik = panel_cik[panel_cik['cik'].notna()].copy()
    panel_cik['cik_int'] = panel_cik['cik'].astype(int)
    panel_cik['cik10'] = panel_cik['cik_int'].astype(str).str.zfill(10)
    print(f"panel firms with CIK: {len(panel_cik)}", flush=True)

    done = {}
    if os.path.exists(CKPT):
        prev = pd.read_parquet(CKPT)
        done = {int(r['permno']): dict(r) for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already done "
              f"(first batch with FULL header sequences)", flush=True)

    rows = list(done.values())
    todo = panel_cik[~panel_cik['permno'].isin(done.keys())]
    print(f"to fetch (fast path, first+last 10-K header): {len(todo)}",
          flush=True)

    t0 = time.time()
    for i, rec in enumerate(todo.itertuples(index=False)):
        permno = int(rec.permno)
        cik_int = rec.cik_int
        cik10 = rec.cik10
        subm = get_submissions(cik10)
        time.sleep(0.10)
        if subm is None:
            rows.append(dict(permno=permno, cik=cik_int,
                             sec_state_current=None, n_10k=0,
                             hdr_states=None, hdr_first=None, hdr_last=None,
                             hdr_changed=False, reloc_year=None))
            continue
        sec_state = (subm.get('addresses', {}).get('business', {})
                     .get('stateOrCountry'))
        fr = subm.get('filings', {}).get('recent', {})
        forms = fr.get('form', [])
        accs = fr.get('accessionNumber', [])
        dates = fr.get('filingDate', [])
        tk = [(accs[j], dates[j]) for j in range(len(forms))
              if forms[j] == '10-K'
              and '2009-01-01' <= dates[j] <= '2024-06-30']
        tk = sorted(tk, key=lambda x: x[1])
        # fast path: fetch only the FIRST and LAST in-sample 10-K header
        fetch = []
        if len(tk) >= 1:
            fetch.append(tk[0])
        if len(tk) >= 2:
            fetch.append(tk[-1])
        hdr_states = []
        for accno, fdate in fetch:
            st = get_10k_header_state(cik_int, accno)
            time.sleep(0.10)
            hdr_states.append((fdate[:10], st))
        valid = [(d, s) for d, s in hdr_states if s is not None]
        states_seq = [s for d, s in valid]
        changed = len(set(states_seq)) > 1
        # reloc_year: with first+last only, the change is bracketed by the two
        # filing dates — record the LAST filing's year as the upper bound
        reloc_year = None
        if changed and len(valid) == 2:
            reloc_year = int(valid[1][0][:4])
        rows.append(dict(
            permno=permno, cik=cik_int, sec_state_current=sec_state,
            n_10k=len(valid),
            hdr_states=json.dumps(valid) if valid else None,
            hdr_first=valid[0][1] if valid else None,
            hdr_last=valid[-1][1] if valid else None,
            hdr_changed=changed, reloc_year=reloc_year))
        if (i + 1) % 200 == 0:
            pd.DataFrame(rows).to_parquet(CKPT)
            rate = (time.time() - t0) / (i + 1)
            eta = rate * (len(todo) - i - 1) / 60.0
            print(f"  {i+1}/{len(todo)} done (last permno {permno}, "
                  f"n_10k={len(valid)}); {rate:.2f}s/firm, ETA {eta:.0f}min",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    out.to_parquet(CKPT)
    print(f"\nwrote {OUT}: {len(out)} firms", flush=True)
    print(f"  firms with >=1 parsed 10-K header: "
          f"{(out['n_10k']>0).sum()}", flush=True)
    print(f"  firms with a 10-K-header STATE change (first vs last, or "
          f"full-seq for first batch): {out['hdr_changed'].sum()}", flush=True)


if __name__ == '__main__':
    main()
