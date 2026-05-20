# =====================================================================
# v9_state_pension_control.py
# Builds a home-state public-pension holder share from 13F and adds it as a control/interaction in the headline mid-IO three-way (home-bias robustness).
#
# Inputs:    WRDS tr_13f.s34names/s34, crsp.stocknames; _dfm_stable_hq.parquet, _thomson_s34_v6_firmquarter.parquet; cached _v9_state_pension_*.parquet
# Outputs:   _v9_state_pension_mgrs.parquet, _v9_state_pension_holdings.parquet, _v9_state_pension_shares_by_cusip_rdate_state.parquet, _v9_pension_cusip_xwalk.parquet, output/stage3a/results_v9_state_pension_control.json
# Paper:     DiD / T9 tab:fs_iv supporting robustness + IA DiD section
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 9 — State-pension home-bias control.

Triage [FIX] Item 7 (High). Construct a state-pension-fund holder share
from the canonical 13F panel: for each (stock, quarter), what fraction
of total 13F holdings is held by state-resident public pension funds?

Approach:
  1. Pull tr_13f.s34names where mgrname matches state-pension keywords
     ('STATE OF', 'TEACHERS', 'EMPLOYEES RETIREMENT', 'PUBLIC EMP',
      'STATE RETIREMENT', 'COMMON PENSION', 'TREASURER', etc.).
     Manually map each managed entity to a US state abbreviation (US-XX).
  2. Pull tr_13f.s34 rows for these managers across 2009Q1-2023Q4
     (quarterly).
  3. For each (cusip, rdate), compute state_pension_shares = sum of
     shares held by each state's pension funds, normalize by shrout
     to get state_pension_share_of_float.
  4. Sum across all states to compute "any state pension share"; also
     compute home-state pension share = state pensions of the same state
     as the firm's HQ.
  5. Add as control in headline three-way regression: ret ~ FOCAL +
     home_state_pension_share * three-way + state FE + month FE.

Output: output/stage3a/results_v9_state_pension_control.json
"""
import os
import sys
import json
import time
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import (load_stable_hq, add_io_terciles, save_json, OUT)
from deepen_estimators import twfe_three_way, FOCAL
sys.path.insert(0, os.path.join(ROOT, "code"))
from utils.wrds_client import wrds_query, wrds_start

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_state_pension_control.json")

# Cached state-pension manager mapping
CACHE_MGR = os.path.join(EMP, "_v9_state_pension_mgrs.parquet")
CACHE_HOLDINGS = os.path.join(EMP, "_v9_state_pension_holdings.parquet")

# US state name -> abbreviation
STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC",
}
# Also common alternate keywords
STATE_ABBR_KEYWORDS = {
    "CALPERS": "CA", "CALSTRS": "CA", "NYSCRF": "NY", "TRS NY": "NY",
    "TRSNYC": "NY", "OHPERS": "OH", "STRS OHIO": "OH",
    "OREGON PERS": "OR", "WSIB": "WA",
}


def pull_state_pension_managers():
    """Pull all state-pension managers from tr_13f.s34names and assign each a
    US state of residence."""
    print("  pulling state-pension managers from tr_13f.s34names ...",
          flush=True)
    df = wrds_query("""
        SELECT mgrno, mgrname, typecode, country
        FROM tr_13f.s34names
        WHERE (
              LOWER(mgrname) LIKE '%retirement system%'
              OR LOWER(mgrname) LIKE '%state of %'
              OR LOWER(mgrname) LIKE '%public emp%'
              OR LOWER(mgrname) LIKE '%teachers retirement%'
              OR LOWER(mgrname) LIKE '%state teach%'
              OR LOWER(mgrname) LIKE '%public retirement%'
              OR LOWER(mgrname) LIKE '%municipal employ%'
              OR LOWER(mgrname) LIKE '%state pension%'
              OR LOWER(mgrname) LIKE '%common pension%'
              OR LOWER(mgrname) LIKE '%state treasurer%'
              OR LOWER(mgrname) LIKE '%calpers%'
              OR LOWER(mgrname) LIKE '%calstrs%'
              OR LOWER(mgrname) LIKE '%nyscrf%'
              OR LOWER(mgrname) LIKE '%trsny%'
              OR LOWER(mgrname) LIKE '%state board%'
              OR LOWER(mgrname) LIKE '%state employees%'
              OR LOWER(mgrname) LIKE '%police & fire%'
              OR LOWER(mgrname) LIKE '%police and fire%'
              OR LOWER(mgrname) LIKE '%state univ%retire%'
              OR LOWER(mgrname) LIKE '%judicial retire%'
              OR LOWER(mgrname) LIKE '%wsib%'
        )
        AND (country IS NULL OR country='UNITED STATES' OR country='USA')
    """, timeout=300)
    print(f"    raw candidates: {len(df)}", flush=True)

    # Assign each manager to a state if name contains a state name keyword
    def assign_state(name):
        if not isinstance(name, str):
            return None
        u = name.upper()
        # First look for exact state names (longer first to avoid e.g. "DAKOTA")
        for sname, abbr in sorted(STATE_NAMES.items(),
                                  key=lambda x: -len(x[0])):
            if re.search(rf"\b{re.escape(sname)}\b", u):
                return abbr
        # Common short keywords
        for kw, abbr in STATE_ABBR_KEYWORDS.items():
            if kw in u:
                return abbr
        # Also try common abbreviations like "NJ" or "CA"
        # only as part of explicit constructions: "STATE NJ", "TRS NY"
        m = re.search(r"\b(STATE|RETIRE\w*|TRS|EMP\w*)\s+([A-Z]{2})\b", u)
        if m:
            cand = m.group(2)
            if cand in STATE_NAMES.values():
                return cand
        return None

    df['state'] = df['mgrname'].apply(assign_state)
    print(f"    assigned to a state: {df['state'].notna().sum()}",
          flush=True)
    # Keep only assigned ones
    df = df[df['state'].notna()].copy()
    # De-dup at the mgrno level — manager-name-version variants
    df = df.sort_values(['mgrno', 'mgrname']).drop_duplicates(
        subset='mgrno', keep='first')
    print(f"    distinct mgrno with state assignment: {len(df)}",
          flush=True)
    print(f"    by state distribution: "
          f"{df['state'].value_counts().head(15).to_dict()}", flush=True)
    return df[['mgrno', 'mgrname', 'state', 'typecode']].copy()


def pull_pension_holdings(mgrno_list):
    """Pull tr_13f.s34 holdings for the state-pension managers across the
    panel period."""
    print(f"  pulling holdings for {len(mgrno_list)} state-pension mgrnos "
          "...", flush=True)
    # Use a parameterized query via inline value list
    in_str = ','.join(str(int(m)) for m in mgrno_list)
    sql = f"""
        SELECT mgrno, cusip, shares, rdate, fdate
        FROM tr_13f.s34
        WHERE mgrno IN ({in_str})
          AND rdate BETWEEN DATE '2009-01-01' AND DATE '2024-01-01'
    """
    df = wrds_query(sql, timeout=600)
    print(f"    raw holdings rows: {len(df)}", flush=True)
    # Dedup amendments: keep most-recent fdate per (mgrno, cusip, rdate)
    df = df.sort_values(['mgrno', 'cusip', 'rdate', 'fdate'])
    df = df.drop_duplicates(subset=['mgrno', 'cusip', 'rdate'], keep='last')
    print(f"    after amendment-dedup: {len(df)}", flush=True)
    return df


def build_state_pension_panel():
    """Build (cusip, rdate, state) state-pension shares held."""
    if os.path.exists(CACHE_HOLDINGS):
        print(f"  loading cached state-pension holdings "
              f"{CACHE_HOLDINGS}", flush=True)
        return pd.read_parquet(CACHE_HOLDINGS)

    wrds_start()
    if os.path.exists(CACHE_MGR):
        mgr = pd.read_parquet(CACHE_MGR)
        print(f"  loaded cached state-pension manager map: {len(mgr)} mgrs",
              flush=True)
    else:
        mgr = pull_state_pension_managers()
        mgr.to_parquet(CACHE_MGR)
        print(f"  cached state-pension manager map to {CACHE_MGR}",
              flush=True)
    h = pull_pension_holdings(mgr['mgrno'].tolist())
    h = h.merge(mgr[['mgrno', 'state']], on='mgrno', how='left')
    h.to_parquet(CACHE_HOLDINGS)
    print(f"  cached state-pension holdings to {CACHE_HOLDINGS}",
          flush=True)
    return h


def main():
    t0 = time.time()
    print("=== v9 Test 9: State-pension home-bias control ===", flush=True)

    try:
        h = build_state_pension_panel()
    except Exception as e:
        print(f"  ERROR pulling state-pension data: {e}", flush=True)
        save_json({
            "test": "v9 Test 9: state pension control",
            "verdict": f"NOT_RUN_WRDS_ERROR: {e}",
            "elapsed_s": round(time.time() - t0, 2),
        }, OUT_JSON)
        return

    print(f"  state-pension holdings panel: {len(h):,} rows", flush=True)
    print(f"    states: {h['state'].value_counts().head(10).to_dict()}",
          flush=True)
    print(f"    quarters: rdate range "
          f"{h['rdate'].min()} to {h['rdate'].max()}", flush=True)

    # Aggregate state-pension shares per (cusip, rdate, state)
    # The holding cusip in tr_13f is 8-digit cusip without check digit; the
    # canonical Thomson panel built earlier uses similar key. We can link
    # holdings -> permno via crsp.stocknames cusip history.
    # Simpler: sum shares by (cusip, rdate, state); convert to share of
    # institutional total later via inst_final from the canonical panel.
    h['cusip8'] = h['cusip'].str[:8].str.upper()
    pen_panel = (h.groupby(['cusip8', 'rdate', 'state'], as_index=False)
                 ['shares'].sum())
    print(f"  aggregated state-pension panel: {len(pen_panel):,} rows",
          flush=True)

    # Save the aggregated panel
    out_path = os.path.join(EMP,
                            "_v9_state_pension_shares_by_cusip_rdate_state.parquet")
    pen_panel.to_parquet(out_path)
    print(f"  saved to {out_path}", flush=True)

    # Now: link cusip -> permno using crsp.stocknames.
    # Inline pull because we want to use cached values if possible.
    cusip_xwalk_path = os.path.join(EMP, "_v9_pension_cusip_xwalk.parquet")
    if os.path.exists(cusip_xwalk_path):
        xw = pd.read_parquet(cusip_xwalk_path)
        print(f"  loaded cached cusip->permno xwalk: {len(xw)}",
              flush=True)
    else:
        print("  pulling cusip->permno crosswalk from CRSP ...", flush=True)
        unique_cusips = pen_panel['cusip8'].dropna().unique()
        # batch
        xw_parts = []
        batch_n = 500
        for i in range(0, len(unique_cusips), batch_n):
            batch = unique_cusips[i:i+batch_n]
            in_str = ','.join("'" + c + "'" for c in batch)
            sql = f"""
                SELECT DISTINCT ncusip AS cusip8, permno, namedt, nameendt
                FROM crsp.stocknames
                WHERE ncusip IN ({in_str})
            """
            try:
                p = wrds_query(sql, timeout=120)
                xw_parts.append(p)
            except Exception as e:
                print(f"    error on batch {i}: {e}", flush=True)
        xw = pd.concat(xw_parts, ignore_index=True)
        xw.to_parquet(cusip_xwalk_path)
        print(f"  cached xwalk: {len(xw)} rows", flush=True)

    # Merge holdings -> permno by cusip, then filter by rdate within namedt..nameendt
    print("  joining pension holdings to permno ...", flush=True)
    pen_panel['rdate'] = pd.to_datetime(pen_panel['rdate'])
    xw['namedt'] = pd.to_datetime(xw['namedt'])
    xw['nameendt'] = pd.to_datetime(xw['nameendt'])
    merged = pen_panel.merge(xw, on='cusip8', how='inner')
    merged = merged[(merged['rdate'] >= merged['namedt'])
                    & (merged['rdate'] <= merged['nameendt'])]
    print(f"  after xwalk+time filter: {len(merged):,} rows", flush=True)

    # Sum to (permno, rdate, state) — i.e., state-pension shares from each
    # state for each stock-quarter.
    p_state = (merged.groupby(['permno', 'rdate', 'state'], as_index=False)
               ['shares'].sum())
    # Pivot: state -> column
    pivot = p_state.pivot_table(index=['permno', 'rdate'], columns='state',
                                values='shares', fill_value=0).reset_index()
    print(f"  pivot table shape: {pivot.shape}", flush=True)

    # ============================================================
    # Build home-state pension share per stock-month for the stable-HQ panel
    # ============================================================
    print("\n  --- Loading stable-HQ panel ---", flush=True)
    d_stable = load_stable_hq()
    d_stable['rdate_q'] = (d_stable['date'].dt.to_period('Q')
                            .dt.start_time + pd.offsets.QuarterEnd(0))

    # melt pivot back for home-state lookup
    p_state_long = (p_state
                    .rename(columns={'shares': 'pension_shares_held'}))
    p_state_long['rdate_q'] = pd.to_datetime(p_state_long['rdate'])
    p_state_long['hq_state'] = 'US-' + p_state_long['state']

    # Get total institutional shares per (permno, rdate) from canonical
    # Thomson panel for normalization
    from v9_helpers import THOMSON
    inst = pd.read_parquet(THOMSON)
    inst['rdate'] = pd.to_datetime(inst['rdate'])
    inst = inst[['permno', 'rdate', 'inst_final', 'shrout_sh']].copy()
    inst['rdate_q'] = inst['rdate']

    # Compute total state-pension-resident share = sum across all states
    total_pen = (merged.groupby(['permno', 'rdate'], as_index=False)
                 ['shares'].sum().rename(
                     columns={'shares': 'total_state_pension_shares'}))
    total_pen['rdate_q'] = pd.to_datetime(total_pen['rdate'])

    # Merge into stable-HQ
    d = d_stable.merge(
        p_state_long[['permno', 'rdate_q', 'hq_state',
                      'pension_shares_held']],
        on=['permno', 'rdate_q', 'hq_state'], how='left')
    d['home_state_pension_shares'] = (d['pension_shares_held']
                                       .fillna(0))
    d = d.merge(total_pen[['permno', 'rdate_q',
                            'total_state_pension_shares']],
                on=['permno', 'rdate_q'], how='left')
    d['total_state_pension_shares'] = (d['total_state_pension_shares']
                                        .fillna(0))
    d = d.merge(inst[['permno', 'rdate_q', 'shrout_sh', 'inst_final']],
                on=['permno', 'rdate_q'], how='left')
    # share of float
    d['home_state_pen_pct_float'] = (
        d['home_state_pension_shares'] /
        d['shrout_sh'].replace(0, np.nan))
    d['total_state_pen_pct_float'] = (
        d['total_state_pension_shares'] /
        d['shrout_sh'].replace(0, np.nan))
    d['home_state_pen_pct_inst'] = (
        d['home_state_pension_shares'] /
        d['inst_final'].replace(0, np.nan))

    d = d.dropna(subset=FOCAL + ['ret', 'hq_state', 'io_share_persist',
                                  'home_state_pen_pct_float']).copy()
    print(f"  estimation panel: {len(d):,} firm-months "
          f"({d['permno'].nunique()} permnos)", flush=True)
    print(f"  home_state_pen_pct_float distribution: "
          f"mean={d['home_state_pen_pct_float'].mean():.4f}, "
          f"median={d['home_state_pen_pct_float'].median():.4f}, "
          f"max={d['home_state_pen_pct_float'].max():.4f}", flush=True)

    # ============================================================
    # Run headline regression with home-state pension control
    # ============================================================
    print("\n  --- Headline three-way + home_state_pen interactions ---",
          flush=True)
    # Standardize the control
    d['hsp_z'] = d.groupby('ym')['home_state_pen_pct_float'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    # mid-IO subsample
    d_io = add_io_terciles(d, 'io_share_persist')
    d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
    print(f"  mid-IO sample: {len(d_mid):,}", flush=True)

    # Baseline three-way (no control)
    r_base = twfe_three_way(d_mid)
    print(f"    baseline mid-IO: gamma={r_base['coef']:+.6f} "
          f"state_t={r_base['t_state']:.2f}", flush=True)

    # Augmented spec: add home_state_pen_z as control + interaction with
    # mom*iv (to allow the three-way coefficient to differ at different
    # pension penetrations). Add the 4 extra interactions:
    #   hsp_z, mom*hsp_z, iv*hsp_z, mom*iv*hsp_z
    d_mid['mom_x_hsp'] = d_mid['mom_12_2'] * d_mid['hsp_z']
    d_mid['iv_x_hsp'] = d_mid['iv'] * d_mid['hsp_z']
    d_mid['mom_x_iv_x_hsp'] = d_mid['mom_x_iv'] * d_mid['hsp_z']

    # Add these 4 columns to a manual OLS spec with state+month FE
    n = len(d_mid)
    base_focal = d_mid[FOCAL].values.astype(float)
    pen_extra = d_mid[['hsp_z', 'mom_x_hsp', 'iv_x_hsp',
                        'mom_x_iv_x_hsp']].values.astype(float)
    y = d_mid['ret'].values.astype(float)
    sc = pd.Categorical(d_mid['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d_mid['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Lit = sp.csr_matrix(base_focal)  # 7 cols
    Pen = sp.csr_matrix(pen_extra)   # 4 cols
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, Lit, Pen, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.pinv(WtW)
    beta = WtW_inv @ (W.T @ y)
    resid = y - W @ beta
    ug = np.unique(sc)
    Cs = sp.csr_matrix((np.ones(n),
                        (np.arange(n), np.searchsorted(ug, sc))),
                       shape=(n, len(ug)))
    scores = (W.multiply(resid[:, None])).toarray()
    G_scores = Cs.T @ scores
    meat = G_scores.T @ G_scores
    V_state = WtW_inv @ meat @ WtW_inv
    # Index of lit three-way: 1 (intercept) + 6 (lower-order) = 7
    idx_lit = 1 + 6
    # Index of home-state pension three-way: 1 + 7 + 3 = 11
    idx_pen = 1 + 7 + 3
    coef_lit = float(beta[idx_lit])
    coef_pen = float(beta[idx_pen])
    se_lit = float(np.sqrt(max(V_state[idx_lit, idx_lit], 0)))
    se_pen = float(np.sqrt(max(V_state[idx_pen, idx_pen], 0)))
    t_lit = coef_lit / se_lit if se_lit > 0 else np.nan
    t_pen = coef_pen / se_pen if se_pen > 0 else np.nan
    print(f"    augmented mid-IO:", flush=True)
    print(f"      gamma_lit (three-way) = {coef_lit:+.6f} "
          f"(state-t={t_lit:.2f})", flush=True)
    print(f"      gamma_pen (home-state pension three-way) = "
          f"{coef_pen:+.6f} (state-t={t_pen:.2f})", flush=True)

    results = {
        "test": "v9 Test 9: State-pension home-bias control",
        "triage_fix": "Item 7 (High)",
        "sample": {
            "n_estimation_panel": int(len(d)),
            "mid_io_n": int(n),
            "n_state_pension_mgrs": int(len(p_state['state'].unique())),
        },
        "baseline_mid_io_no_control": {
            "coef": float(r_base['coef']),
            "se_state": float(r_base['se_state']),
            "t_state": float(r_base['t_state']),
            "n_obs": int(r_base['n_obs']),
        },
        "augmented_mid_io_with_home_state_pen": {
            "gamma_lit_three_way": coef_lit,
            "se_state_lit": se_lit,
            "t_state_lit": float(t_lit),
            "gamma_pen_three_way": coef_pen,
            "se_state_pen": se_pen,
            "t_state_pen": float(t_pen),
            "n_obs": int(n),
        },
        "home_state_pension_distribution": {
            "mean_pct_float": float(d['home_state_pen_pct_float'].mean()),
            "median_pct_float": float(d['home_state_pen_pct_float'].median()),
            "max_pct_float": float(d['home_state_pen_pct_float'].max()),
        },
    }

    # Verdict
    ratio = coef_lit / r_base['coef'] if r_base['coef'] != 0 else None
    notes = []
    if ratio is not None:
        notes.append(f"Augmented lit-three-way / baseline = {ratio:.3f}.")
    if (ratio is not None and ratio > 0.6 and ratio < 1.4):
        verdict = "SURVIVES_CONTROL"
    elif ratio is not None and ratio > 0.3:
        verdict = "PARTIALLY_ATTENUATED"
    else:
        verdict = "ATTENUATED_HEAVILY"
    results['verdict'] = verdict
    results['verdict_notes'] = notes
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2), "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
