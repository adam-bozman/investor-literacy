# =====================================================================
# v7_permutation_stress_test.py
# Permutation/placebo inference for the within-stock low-IO minus high-IO three-way
# difference: builds null distributions by reshuffling firm IO-tercile assignment
# (within state-month, within month, within-permno) plus a median-split robustness.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_permutation.json; saved null arrays
#            _v7_perm_null_persist_arm{1,2,3}.npy and _v7_perm_null_tv_arm{1,2}.npy
# Paper:     Internet Appendix full inference battery section (permutation inference)
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Stage 3a re-fire (Gate 5 Major Revision r3) — Row 1 CRITICAL PATH.

Permutation / placebo stress-test battery for the within-stock contrast on the
denominator-corrected Thomson s34 panel (v6 cache). The structured referee
(Major 1) asks for three permutation arms; the editor's canonical list ranks
this as the load-bearing test deciding centerpiece framing.

The HEADLINE within-stock difference (v6 corrected panel, stacked fully-
interacted design):
    persistent      DIFF = -0.0211, wcb p = 0.144
    time-varying    DIFF = -0.0181, wcb p = 0.264

We construct three null distributions of THIS SAME ESTIMATOR by re-shuffling
firm->IO-tercile assignment, holding everything else fixed (panel, focal
covariates, returns, FE structure). The observed difference is then compared
to the simulated null distribution.

The mechanism predicts a NEGATIVE difference (low-IO three-way more negative
than high-IO). We report:
    one-sided directional left-tail   P(DIFF_null <= DIFF_obs)
    two-tailed                        P(|DIFF_null| >= |DIFF_obs|)
across N_PERM permutation draws.

CRITICAL: the estimator under permutation MUST be the same fully-interacted
stacked design used to compute the headline difference — otherwise omitted-
variable bias from the (low_IO x lower-order focal) interactions distorts the
comparison. We therefore rebuild the full stacked design (with the PERMUTED
low_IO indicator) per draw, partial out the WHOLE control matrix per draw, and
fit the 2x2 system on (three_way, low_IO_perm x three_way) per draw. This is
exact (no shortcut) and is the reference distribution the referee actually
wants.

THREE PERMUTATION ARMS
======================
Arm 1 (REFEREE Major 1 (i)): RANDOMIZE IO-TERCILE ASSIGNMENT WITHIN STATE-
    MONTH. Each (state, month) cell's firms are randomly re-assigned to
    terciles preserving the marginal tercile sizes in that cell. STRONGEST
    test: discards the firm-IO link while preserving state-month composition.

Arm 2 (REFEREE Major 1 (iii)): WITHIN MONTH ONLY. Randomly assign IO-tercile
    across firms within each month, NOT within state-month. Coarser null.

Arm 3 (additional, for the persistent measure): WITHIN-PERMNO shuffle.
    Each permno gets a random tercile (constant across the firm's history),
    preserving the marginal tercile sizes. Same shape as the persistent IO
    measure (one tercile per firm); the corresponding null for the persistent
    contrast.

The referee's second branch (Major 1 (ii): two-tercile median split) is a
DIFFERENT point estimate, not a permutation null. It is reported alongside
as a robustness number.

Performance
-----------
Per-draw cost is dominated by partialling out the FE block W from x3 and from
x_int_perm. We exploit that W stays *the same* across draws (W = intercept +
6 lower-order focal + state FE + month FE only — the low_IO main effect and
(low_IO x lower-order focal) terms are absorbed via partialling-out of a
LARGER control matrix W_full that varies by draw). To make this tractable we
use the *concentrated* FWL design:

  1. Build W_BASE = intercept + 6 lower-order focal + state FE + month FE.
     Pre-compute (W_BASE' W_BASE)^{-1} W_BASE'. (Once.)
  2. Pre-partial three := residual of three_way after W_BASE. (Once.)
  3. Pre-partial y := residual of y after W_BASE. (Once.)
  4. For each draw with permuted low_IO indicator L:
       L_t          := residual of L after W_BASE
       L_three      := L * three (element-wise)
       L_three_t    := residual of (L*three) after W_BASE
       L_lo6        := L * (6 lower-order focal)  -- 6 new controls
       L_lo6_t      := residual of (L*lo6) after W_BASE
       AUX          := [L_t, L_lo6_t]   (concatenated; n x 7)
       Then partial three_t, L_three_t, yt against AUX (using AUX' AUX)^{-1}.
       Fit the 2x2 system on (three_t_aux, L_three_t_aux), report coef on
       L_three_t_aux. This recovers the exact same difference coefficient as
       the full stacked_io_difference but in O(n * 7) per draw rather than
       O(n * (n_states + n_months + 7 + 7)) per draw.

This gives the EXACT fully-interacted difference estimator under each draw,
matching the headline numbers byte-for-byte at the observed assignment.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

np.random.seed(42)

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from deepen_estimators import FOCAL, stacked_io_difference   # noqa: E402

DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a",
                        "results_v7_permutation.json")
TABDIR = os.path.join(ROOT, "output", "stage3a", "tables")
os.makedirs(TABDIR, exist_ok=True)

N_PERM = 1000


def load_merged_panel():
    if not os.path.exists(DFM_CACHE):
        raise FileNotFoundError(
            f"Merged v7 panel not found ({DFM_CACHE}). Run probe_v7.py first.")
    return pd.read_parquet(DFM_CACHE)


def stack_low_high(dfm, io_col):
    """Build the stacked low+high subsample, with io_grp_bin in {low, high}.

    Matches deepen_13f_split_v6.py add_io_terciles: tercile assignment is PER
    FIRM (cross-sectional, on the firm-mean of io_col), even for the
    time-varying io_col. The time-varying label refers to the io_share VALUES
    (the within-month, no look-ahead lagged values used as a regressor), not
    to a within-month tercile assignment. The v6 convention is preserved so
    the permutation null is directly comparable to the v6 headline.
    """
    d = dfm[dfm[io_col].notna()].copy()
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d['io_grp'] = d['permno'].map(terc).astype('object')
    d = d[d['io_grp'].isin(['IO1_low', 'IO3_high'])].copy()
    d['io_grp_bin'] = np.where(d['io_grp'] == 'IO1_low', 'low', 'high')
    return d


def precompute_base(d):
    """Build W_BASE = intercept + 6 lower-order focal + state FE + month FE.
    Returns (W_BASE_csc, projection_function P that partials W_BASE out,
    state_codes, month_codes, ym_codes, lo6, three, y)."""
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    lo6 = focal[:, :6]
    three = focal[:, 6]
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(lo6)
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, W6s, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)
    # cache (W * WtW_inv) once for fast partial: partial(v) = v - W*(WtW_inv*(W^T v))

    def partial(v):
        # v: n-array or n x k matrix
        if v.ndim == 1:
            return v - W @ (WtW_inv @ (W.T @ v))
        else:
            return v - W @ (WtW_inv @ (W.T @ v))

    return partial, sc, mc, lo6, three, y, nS, nM


def diff_for_assignment(low_ind, partial, three, lo6, y_t, three_t):
    """Compute the fully-interacted stacked DIFF for one assignment of low_ind.

    The stacked design is:
      ret ~ intercept + 6 lower-order focal + state FE + month FE
            + low_IO + (low_IO x 6 lower-order focal)
            + three_way + (low_IO x three_way)
    We partial out W_BASE (everything except the low_IO and interaction terms)
    once. The remaining residual regression is y_t ~ {L_t, L_lo6_t, three_t,
    L_three_t}; the coefficient on L_three_t is the difference. By
    Frisch-Waugh-Lovell, partial L_t / L_lo6_t / three_t / L_three_t through
    each other gives the same OLS coefficient.

    Implementation: build AUX = [L_t, L_lo6_t] (7 cols), partial three_t and
    L_three_t against AUX, then fit a 2x2 system on (three_t_aux,
    L_three_t_aux).
    """
    L = low_ind.astype(float)
    L_t = partial(L)
    L_lo6 = lo6 * L[:, None]
    L_lo6_t = partial(L_lo6)
    AUX = np.column_stack([L_t, L_lo6_t])  # n x 7
    AUX_tA = AUX.T @ AUX
    AUX_tA_inv = np.linalg.inv(AUX_tA)
    # partial three_t and L_three_t through AUX (FWL within partialled space)
    L_three = L * three
    L_three_t = partial(L_three)
    AUX_tx3 = AUX.T @ three_t
    AUX_txL3 = AUX.T @ L_three_t
    three_t_aux = three_t - AUX @ (AUX_tA_inv @ AUX_tx3)
    L_three_t_aux = L_three_t - AUX @ (AUX_tA_inv @ AUX_txL3)
    AUX_ty = AUX.T @ y_t
    y_t_aux = y_t - AUX @ (AUX_tA_inv @ AUX_ty)
    # fit 2x2 on (three_t_aux, L_three_t_aux)
    M = np.array([[float(three_t_aux @ three_t_aux),
                   float(three_t_aux @ L_three_t_aux)],
                  [float(three_t_aux @ L_three_t_aux),
                   float(L_three_t_aux @ L_three_t_aux)]])
    rhs = np.array([float(three_t_aux @ y_t_aux),
                    float(L_three_t_aux @ y_t_aux)])
    beta = np.linalg.solve(M, rhs)
    return float(beta[1]), float(beta[0])


def gen_permutation(sc, ym_codes, low_obs, kind, rng,
                    cell_rows_sm=None, cell_lows_sm=None,
                    cell_rows_m=None, cell_lows_m=None,
                    permno_pos=None, row_firm_pos=None,
                    firm_labels=None):
    """Return a permuted low indicator of length n."""
    n = len(low_obs)
    low_perm = np.zeros(n, dtype=np.int8)
    if kind == 'within_state_month':
        for k, rows in cell_rows_sm.items():
            k_low = cell_lows_sm[k]
            if k_low == 0 or k_low == len(rows):
                low_perm[rows] = low_obs[rows]
            else:
                idx = rng.choice(len(rows), size=k_low, replace=False)
                low_perm[rows[idx]] = 1
    elif kind == 'within_month_only':
        for k, rows in cell_rows_m.items():
            k_low = cell_lows_m[k]
            if k_low == 0 or k_low == len(rows):
                low_perm[rows] = low_obs[rows]
            else:
                idx = rng.choice(len(rows), size=k_low, replace=False)
                low_perm[rows[idx]] = 1
    elif kind == 'within_permno':
        shuffled = rng.permutation(firm_labels)
        row_labels = shuffled[row_firm_pos]
        low_perm = (row_labels == 'low').astype(np.int8)
    else:
        raise ValueError(kind)
    return low_perm


def run_arm_set(dstack, label):
    """For a given stacked subsample, compute the observed (fully-interacted)
    difference under the simplified-partial trick (which equals the byte-
    identical fully-interacted estimator), then run the three permutation
    arms each generating null draws of the SAME estimator."""
    print(f"  arm-set [{label}] N={len(dstack):,} | "
          f"states={dstack['hq_state'].nunique()} | "
          f"months={dstack['ym'].nunique()}", flush=True)
    n = len(dstack)
    partial, sc, mc, lo6, three, y, nS, nM = precompute_base(dstack)
    y_t = partial(y)
    three_t = partial(three)
    low_obs = (dstack['io_grp_bin'].values == 'low').astype(np.int8)
    ym_codes = mc
    print(f"    precompute done. running observed reference ...", flush=True)
    diff_obs, three_high = diff_for_assignment(
        low_obs, partial, three, lo6, y_t, three_t)
    print(f"    observed fully-interacted DIFF (FWL-aux) = "
          f"{diff_obs:+.6f}", flush=True)

    out = {'observed_fully_interacted_diff': diff_obs,
           'observed_three_way_high_IO': three_high,
           'n_obs': int(n),
           'n_state_clusters': int(dstack['hq_state'].nunique()),
           'n_months': int(dstack['ym'].nunique()),
           'n_perm': N_PERM}

    # Build cell maps
    sm_key = sc.astype(np.int64) * (ym_codes.max() + 1) + ym_codes
    cell_rows_sm = defaultdict(list)
    for i in range(n):
        cell_rows_sm[sm_key[i]].append(i)
    cell_rows_sm = {k: np.array(v, dtype=np.int64)
                    for k, v in cell_rows_sm.items()}
    cell_lows_sm = {k: int(low_obs[v].sum())
                    for k, v in cell_rows_sm.items()}
    cell_rows_m = defaultdict(list)
    for i in range(n):
        cell_rows_m[ym_codes[i]].append(i)
    cell_rows_m = {k: np.array(v, dtype=np.int64)
                   for k, v in cell_rows_m.items()}
    cell_lows_m = {k: int(low_obs[v].sum())
                   for k, v in cell_rows_m.items()}

    # Within-permno pool
    permno_arr = dstack['permno'].values
    firm_to_grp = dstack.groupby('permno')['io_grp_bin'].first()
    firm_labels = firm_to_grp.values.copy()
    firm_index = firm_to_grp.index.values
    permno_pos = pd.Series(np.arange(len(firm_index)), index=firm_index)
    row_firm_pos = permno_pos.loc[permno_arr].values.astype(np.int64)
    is_persistent_assign = (
        dstack.groupby('permno')['io_grp_bin'].nunique().eq(1).all())

    arms = []
    arms.append(('within_state_month', 1, cell_rows_sm, cell_lows_sm,
                 None, None, None, None, None))
    arms.append(('within_month_only', 2, None, None,
                 cell_rows_m, cell_lows_m, None, None, None))
    if is_persistent_assign:
        arms.append(('within_permno', 3, None, None, None, None,
                     permno_pos, row_firm_pos, firm_labels))

    diff_arrays = {}
    for kind, seedoff, crs, cls, crm, clm, pp, rfp, fl in arms:
        print(f"  arm [{kind}] running {N_PERM} perms ...", flush=True)
        rng = np.random.RandomState(42 + seedoff)
        diffs = np.empty(N_PERM)
        t0 = time.time()
        for p in range(N_PERM):
            low_perm = gen_permutation(
                sc, ym_codes, low_obs, kind, rng,
                cell_rows_sm=crs, cell_lows_sm=cls,
                cell_rows_m=crm, cell_lows_m=clm,
                permno_pos=pp, row_firm_pos=rfp, firm_labels=fl)
            try:
                d_p, _ = diff_for_assignment(
                    low_perm, partial, three, lo6, y_t, three_t)
                diffs[p] = d_p
            except np.linalg.LinAlgError:
                diffs[p] = np.nan
            if (p + 1) % 100 == 0:
                print(f"    [{kind}] {p+1}/{N_PERM} ({time.time()-t0:.1f}s)",
                      flush=True)
        valid = diffs[~np.isnan(diffs)]
        left_p = float((valid <= diff_obs).sum()) / max(1, len(valid))
        twotail_p = float((np.abs(valid) >= abs(diff_obs)).sum()) / max(
            1, len(valid))
        out[kind] = {
            'left_tail_p': left_p,
            'two_tail_p': twotail_p,
            'null_mean': float(np.mean(valid)),
            'null_std': float(np.std(valid)),
            'null_q025': float(np.percentile(valid, 2.5)),
            'null_q50': float(np.percentile(valid, 50)),
            'null_q975': float(np.percentile(valid, 97.5)),
            'null_min': float(np.min(valid)),
            'null_max': float(np.max(valid)),
            'n_valid': int(len(valid)),
        }
        diff_arrays[kind] = diffs
        print(f"    {kind}: null mean={out[kind]['null_mean']:+.5f} | "
              f"null std={out[kind]['null_std']:.5f} | "
              f"left-tail p={left_p:.4f} | two-tail p={twotail_p:.4f}",
              flush=True)

    return out, diff_arrays


def main():
    print("=== Row 1 permutation/placebo stress-test (v7) ===", flush=True)
    dfm = load_merged_panel()
    print(f"merged panel: {len(dfm):,} firm-months, "
          f"{dfm['permno'].nunique()} permnos", flush=True)

    results = {
        'task': 'Row 1 (Major 1 / freeform / mechanism FIX 3): permutation/'
                'placebo stress-test of within-stock low-IO minus high-IO '
                'three-way difference',
        'panel': 'denominator-corrected Thomson s34 v6 (amendment-deduped + '
                 'sole+shared numerator); estimation sample restricted to '
                 'FOCAL+ret+hq_state non-null',
        'reference_v6_results': {
            'persistent_diff': -0.0211,
            'persistent_wcb_p': 0.144,
            'time_varying_diff': -0.0181,
            'time_varying_wcb_p': 0.264,
        },
        'estimator': 'Fully-interacted stacked design (the v6 headline). '
                     'Per-draw fit via FWL-aux trick that recovers the same '
                     'coefficient as stacked_io_difference at the observed '
                     'assignment, but is fast enough for 1000 draws each '
                     'arm. Reference distribution is the fully-interacted '
                     'difference; mechanism predicts negative.',
    }

    print("\n--- PERSISTENT (between-firm) IO measure ---", flush=True)
    dstack_p = stack_low_high(dfm, 'io_share_persist')
    out_p, da_p = run_arm_set(dstack_p, 'persistent')
    results['persistent'] = out_p
    np.save(os.path.join(EMP, '_v7_perm_null_persist_arm1.npy'),
            da_p.get('within_state_month'))
    np.save(os.path.join(EMP, '_v7_perm_null_persist_arm2.npy'),
            da_p.get('within_month_only'))
    if 'within_permno' in da_p:
        np.save(os.path.join(EMP, '_v7_perm_null_persist_arm3.npy'),
                da_p['within_permno'])

    print("\n--- TIME-VARYING (within-month) IO measure ---", flush=True)
    dstack_tv = stack_low_high(dfm, 'io_share')
    out_tv, da_tv = run_arm_set(dstack_tv, 'time_varying')
    results['time_varying'] = out_tv
    np.save(os.path.join(EMP, '_v7_perm_null_tv_arm1.npy'),
            da_tv.get('within_state_month'))
    np.save(os.path.join(EMP, '_v7_perm_null_tv_arm2.npy'),
            da_tv.get('within_month_only'))

    # Median-split (referee Major 1 (ii)) ------------------
    print("\n--- median-split robustness (two-tercile) ---", flush=True)
    for io_col, mlabel in [('io_share_persist', 'persistent'),
                           ('io_share', 'time_varying')]:
        d = dfm[dfm[io_col].notna()].copy()
        if io_col == 'io_share_persist':
            perm_io = d.groupby('permno')[io_col].mean().dropna()
            med = perm_io.median()
            d['io_grp_bin'] = np.where(d['permno'].map(perm_io) <= med,
                                       'low', 'high')
        else:
            med = d.groupby('ym')[io_col].transform('median')
            d['io_grp_bin'] = np.where(d[io_col] <= med, 'low', 'high')
        sd = stacked_io_difference(d, B=4999, seed=42)
        results[f'median_split_{mlabel}'] = {
            'difference_coef': sd['difference_coef'],
            'se_state_clustered': sd['se_state_clustered'],
            't_state_clustered': sd['t_state_clustered'],
            'wcb_p_value': sd['wcb_p_value'],
            'n_obs': sd['n_obs'],
        }
        print(f"  median-split [{mlabel}]: DIFF="
              f"{sd['difference_coef']:+.6f}, "
              f"state-t={sd['t_state_clustered']:.2f}, "
              f"wcb p={sd['wcb_p_value']:.4f}", flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)
    return results


if __name__ == '__main__':
    main()
