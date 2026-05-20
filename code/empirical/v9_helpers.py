# =====================================================================
# v9_helpers.py
# Shared loaders/utilities (panel I/O, relocator set, IO terciles, JSON save) for the v9 [FIX] battery.
#
# Inputs:    _dfm_v7.parquet, _dfm_stable_hq.parquet, _hq_edgar_state_v6.parquet, _reloc_flag_v6.parquet, _thomson_s34_v6_firmquarter.parquet
# Outputs:   none directly (provides save_json + paths used by callers)
# Paper:     Shared library module — imported by v9_* scripts (no direct table)
# Run order: see code/00_master.py
# =====================================================================

"""v9 shared helpers for the round-5 NEW EMPIRICAL [FIX] battery.

Reuses _dfm_v7.parquet (full panel) and _dfm_stable_hq.parquet (the
6,081-firm stable-HQ subsample = headline population).
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
OUT = os.path.join(ROOT, "output", "stage3a")

DFM_V7 = os.path.join(EMP, "_dfm_v7.parquet")
DFM_STABLE = os.path.join(EMP, "_dfm_stable_hq.parquet")
HQ_EDGAR = os.path.join(EMP, "_hq_edgar_state_v6.parquet")
RELOC_FLAG = os.path.join(EMP, "_reloc_flag_v6.parquet")
THOMSON = os.path.join(EMP, "_thomson_s34_v6_firmquarter.parquet")


def load_full_panel():
    d = pd.read_parquet(DFM_V7)
    d['date'] = pd.to_datetime(d['date'])
    return d


def load_stable_hq():
    d = pd.read_parquet(DFM_STABLE)
    d['date'] = pd.to_datetime(d['date'])
    return d


def last_hdr_state(s):
    if not isinstance(s, str):
        return None
    try:
        seq = json.loads(s)
    except Exception:
        return None
    return seq[-1][1] if seq else None


def build_relocator_set():
    """Reproduce the v6 Task A 1,366-firm relocator set.
    Returns set of permnos."""
    hq = pd.read_parquet(HQ_EDGAR)
    dfm = pd.read_parquet(DFM_V7)
    panel_hq = (dfm.groupby('permno')
                .agg(hq_state=('hq_state', 'first'),
                     n_months=('date', 'size'))
                .reset_index())
    panel_hq['hq_state_2l'] = panel_hq['hq_state'].str.replace(
        'US-', '', regex=False)
    m = panel_hq.merge(hq, on='permno', how='left')
    m['flag_hdr_changed'] = (m['hdr_changed'] == True)
    m['hdr_state_last'] = m['hdr_states'].apply(last_hdr_state)
    m['n_10k'] = m['n_10k'].fillna(0).astype(int)
    m['flag_sec_disagree'] = (
        (m['n_10k'] >= 2)
        & m['hdr_state_last'].notna()
        & m['hq_state_2l'].notna()
        & (m['hdr_state_last'] != m['hq_state_2l'])
    )
    m['reloc'] = m['flag_hdr_changed'] | m['flag_sec_disagree']
    return set(m.loc[m['reloc'], 'permno'].astype(int))


def add_io_terciles(d, io_col, group_name='io_grp'):
    """Time-varying tercile assignment from firm-mean IO. Same as the
    canonical pattern used elsewhere."""
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d[group_name] = d['permno'].map(terc).astype('object')
    return d


def add_io_terciles_2009(d, io_col, group_name='io_grp_2009',
                         start="2009-01-01", end="2009-12-31"):
    """Fixed-vintage IO terciles: firm's tercile assignment is determined by
    its 2009 (default: full year) mean IO, held constant across the panel.
    Mid-IO etc. defined by 2009 cross-section, not time-varying.
    """
    d2009 = d[(d['date'] >= pd.Timestamp(start))
              & (d['date'] <= pd.Timestamp(end))]
    perm_io_2009 = d2009.groupby('permno')[io_col].mean().dropna()
    if len(perm_io_2009) < 30:
        # not enough firms to form terciles
        d = d.copy()
        d[group_name] = None
        return d
    terc = pd.qcut(perm_io_2009, 3,
                   labels=['IO1_low', 'IO2_mid', 'IO3_high'],
                   duplicates='drop')
    d = d.copy()
    d[group_name] = d['permno'].map(terc).astype('object')
    return d


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  wrote {path}", flush=True)


if __name__ == '__main__':
    print("v9_helpers loaded.", flush=True)
    print(f"  DFM_V7: {os.path.exists(DFM_V7)}")
    print(f"  DFM_STABLE: {os.path.exists(DFM_STABLE)}")
    print(f"  HQ_EDGAR: {os.path.exists(HQ_EDGAR)}")
    print(f"  THOMSON: {os.path.exists(THOMSON)}")
