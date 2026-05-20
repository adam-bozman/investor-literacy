# =====================================================================
# v7_four_way_continuous.py
# Estimates the continuous-margin four-way interaction mom*IV*literacy_z*IO (level
# and within-month IO-rank specs) with state-clustered SE and wild-cluster
# bootstrap inference on the four-way coefficient.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_four_way.json (no tables written)
# Paper:     Main Table T7 tab:form_grid (quadratic / four-way continuous region)
#            + Internet Appendix 20-bin quadratic section
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Q1 (referee structured Q1; triager row 25): continuous-margin
four-way term `mom x IV x literacy_z x IO`.

Tests the continuous-margin version of the tercile contrast:
    r_{i,s,t} = gamma_1 * mom * IV * literacy_z
              + gamma_2 * mom * IV * literacy_z * IO_{i,t}
              + (all lower-order terms)
              + state FE + month FE
The coefficient gamma_2 is the continuous-margin version of the within-stock
contrast. Under the mechanism (low-IO three-way more negative), gamma_2 > 0
(literacy-modulated three-way attenuates as IO rises).

Two specifications:
  (A) Continuous IO: use io_share_persist or io_share (time-varying) directly
      as the IO variable in the four-way interaction.
  (B) IO RANK: use the within-month IO rank (uniform on [0,1]) — robust to
      the skewed marginal distribution of io_share.

The four-way term sits at the top of a 16-term saturated structure:
    1, mom, IV, lit, IO,
    mom*IV, mom*lit, mom*IO, IV*lit, IV*IO, lit*IO,
    mom*IV*lit, mom*IV*IO, mom*lit*IO, IV*lit*IO,
    mom*IV*lit*IO
(after the 11 lower-order focal + IO terms are partialled out, the four-way
is identified). For the FE block we add state + month FE; clustering and
inference are state-clustered CR1 + wcb on the four-way coefficient.

This is the v7 power-test the referee asked for in Q1.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from deepen_estimators import FOCAL  # noqa: E402

DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_four_way.json")

# bootstrap reps for the wild-cluster bootstrap on the four-way coefficient
B_WCB = 4999
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                 np.sqrt(0.5), 1.0, np.sqrt(1.5)])
CHUNK = 200


def load_merged_panel():
    return pd.read_parquet(DFM_CACHE)


def _cluster_matrix(codes):
    n = len(codes)
    ug = np.unique(codes)
    idx = np.searchsorted(ug, codes)
    return sp.csr_matrix((np.ones(n), (np.arange(n), idx)),
                         shape=(n, len(ug))), ug


def four_way_with_inference(d, io_col, io_label):
    """Run the four-way continuous regression. Returns dict with gamma_2 +
    clustered SEs + wcb p."""
    d = d[d[io_col].notna()].copy()
    print(f"  [{io_label}] N={len(d):,}, "
          f"states={d['hq_state'].nunique()}, "
          f"months={d['ym'].nunique()}", flush=True)

    n = len(d)
    mom = d['mom_12_2'].values.astype(float)
    iv = d['iv'].values.astype(float)
    lit = d['literacy_score_corrected'].values.astype(float)
    IO = d[io_col].values.astype(float)
    y = d['ret'].values.astype(float)

    # 15 lower-order / partial terms (all 4-1 = 14 lower + intercept) plus
    # state FE + month FE. The four-way is the 16th.
    # Lower-order: mom, iv, lit, IO,
    #              mom*iv, mom*lit, mom*IO, iv*lit, iv*IO, lit*IO,
    #              mom*iv*lit, mom*iv*IO, mom*lit*IO, iv*lit*IO
    lo = np.column_stack([
        mom, iv, lit, IO,
        mom * iv, mom * lit, mom * IO, iv * lit, iv * IO, lit * IO,
        mom * iv * lit, mom * iv * IO, mom * lit * IO, iv * lit * IO,
    ])
    four = mom * iv * lit * IO

    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Lo = sp.csr_matrix(lo)
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, Lo, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)

    def partial(v):
        return v - W @ (WtW_inv @ (W.T @ v))

    four_t = partial(four)
    yt = partial(y)
    Sxx = float(four_t @ four_t)
    coef = float(four_t @ yt) / Sxx
    e = yt - coef * four_t
    score = four_t * e

    Cs, states = _cluster_matrix(sc)
    Cm, _ = _cluster_matrix(mc)
    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    Ci, _ = _cluster_matrix(inter)
    ss_s = float(((Cs.T @ score) ** 2).sum())
    ss_m = float(((Cm.T @ score) ** 2).sum())
    ss_i = float(((Ci.T @ score) ** 2).sum())
    meat_cgm = ss_s + ss_m - ss_i
    se_state = float(np.sqrt(ss_s) / Sxx)
    if meat_cgm > 0:
        se_cgm = float(np.sqrt(meat_cgm) / Sxx)
    else:
        se_cgm = None

    # wild-cluster bootstrap on the four-way (restricted: H0: gamma_4 = 0)
    rng = np.random.RandomState(42)
    e_r = yt.copy()  # under H0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    t_obs = coef / se_state if se_state > 0 else np.nan
    t_boot = np.empty(B_WCB)
    done = 0
    print(f"    [{io_label}] running wcb B={B_WCB} ...", flush=True)
    while done < B_WCB:
        b = min(CHUNK, B_WCB - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, b))]
        w = w_state[state_idx, :]
        Y = e_r[:, None] * w
        coef_b = (four_t @ Y) / Sxx
        E_b = Y - four_t[:, None] * coef_b[None, :]
        Score_b = four_t[:, None] * E_b
        cl_b = Cs.T @ Score_b
        se_b = np.sqrt((cl_b ** 2).sum(axis=0)) / Sxx
        t_boot[done:done + b] = np.where(se_b > 0, coef_b / se_b, 0.0)
        done += b
    p_wcb = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))
    q = np.percentile(t_boot, [2.5, 97.5])

    out = {
        'io_label': io_label,
        'n_obs': int(n),
        'n_state_clusters': int(G),
        'gamma_four_way': coef,
        'se_state_clustered': se_state,
        't_state_clustered': float(t_obs),
        'se_cgm_two_way': se_cgm,
        't_cgm_two_way': float(coef / se_cgm) if se_cgm else None,
        'wcb_p_value': p_wcb,
        'wcb_t_q025': float(q[0]),
        'wcb_t_q975': float(q[1]),
        'wcb_B': B_WCB,
        'spec': 'r ~ {15 lower-order incl IO main and lower-order Ks } + '
                'state FE + month FE + mom*IV*lit*IO (the four-way)',
    }
    print(f"    [{io_label}] gamma_4 = {coef:+.6f} | state-t={t_obs:.2f} | "
          f"wcb p={p_wcb:.4f}", flush=True)
    return out


def main():
    print("=== Q1 four-way continuous-margin term (v7) ===", flush=True)
    dfm = load_merged_panel()
    print(f"merged panel: {len(dfm):,} firm-months", flush=True)

    # Add IO-rank columns
    dfm = dfm.copy()
    # within-month rank for time-varying io_share (uniform on [0,1])
    dfm['io_share_rank'] = (
        dfm.groupby('ym')['io_share'].rank(pct=True, method='average'))
    # firm-mean rank for persistent
    perm_io = dfm.groupby('permno')['io_share_persist'].first()
    perm_rank = perm_io.rank(pct=True, method='average')
    dfm['io_share_persist_rank'] = dfm['permno'].map(perm_rank)

    d_focal = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    results = {
        'task': 'Q1 (referee structured Q1; triager row 25): continuous-'
                'margin four-way term mom*IV*literacy_z*IO. Tests if the '
                'continuous-margin form is more powerful than the stacked '
                'tercile contrast.',
        'mechanism_sign_prediction': 'gamma_4 > 0 (literacy-modulated three-'
                                     'way attenuates as IO rises; i.e., the '
                                     'three-way is more negative in low-IO '
                                     'firms).',
        'reference_v6_stacked': {
            'persistent_diff': -0.0211, 'persistent_wcb_p': 0.144,
            'time_varying_diff': -0.0181, 'time_varying_wcb_p': 0.264,
        },
    }

    print("\n--- continuous IO: io_share_persist (between-firm, level) ---",
          flush=True)
    results['persistent_level'] = four_way_with_inference(
        d_focal, 'io_share_persist', 'persistent_level')

    print("\n--- continuous IO: io_share (time-varying, level) ---",
          flush=True)
    results['time_varying_level'] = four_way_with_inference(
        d_focal, 'io_share', 'time_varying_level')

    print("\n--- IO rank: io_share_persist_rank (between-firm, uniform [0,1]) "
          "---", flush=True)
    results['persistent_rank'] = four_way_with_inference(
        d_focal, 'io_share_persist_rank', 'persistent_rank')

    print("\n--- IO rank: io_share_rank (time-varying, uniform [0,1]) ---",
          flush=True)
    results['time_varying_rank'] = four_way_with_inference(
        d_focal, 'io_share_rank', 'time_varying_rank')

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
