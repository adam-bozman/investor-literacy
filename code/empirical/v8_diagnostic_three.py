# =====================================================================
# v8_diagnostic_three.py
# Three diagnostics on the stable-HQ centerpiece: pre/post-2015 mid-IO split (A),
# wild-cluster bootstrap on the mid-IO coefficient plus Romano-Wolf step-down across
# three headline patterns (B), and continuous-vs-discrete reconciliation via a
# 20-bin IO characterization and a quadratic-IO four-way (C).
#
# Inputs:    _dfm_stable_hq.parquet (stable-HQ cache) and _dfm_v7.parquet (full
#            merged firm-month panel: seed panel + corrected Thomson s34 IO).
# Outputs:   output/stage3a/results_v8_diagnostic.json (no tables written)
# Paper:     Internet Appendix diagnostics — full-panel/PIT mid-IO, 20-bin quadratic,
#            and inference-divergence sections (intermediate diagnostic, no direct
#            main table)
# Run order: see code/00_master.py
# =====================================================================

"""v8 DIAGNOSTIC THREE — Stage 6 r4 deepen directive.

Three time-boxed diagnostic tasks on the cached stable-HQ panel (no new
WRDS pulls). Total time-box: 30 minutes.

Task A — Pre/post-2015 split on the mid-IO concentration (Table 1 headline).
   Run the headline TWFE three-way regression separately within each IO
   tercile (low / mid / high) on pre-2015 (2009-01..2014-12) and post-2015
   (2015-01..2023-12) subsamples of stable-HQ. Persistent and time-varying
   IO. Report gamma + state-cl t + N per cell.

Task B — Inference discipline on the mid-IO three-way:
   (1) Wild-cluster bootstrap p-value on the mid-IO three-way coefficient
       itself (state-clustered, Webb 6-point, B=4999). Persistent and
       time-varying.
   (2) Romano-Wolf step-down family-wise correction across the three
       headline patterns:
         P1: mid-IO three-way coefficient (persistent)
         P2: within-retail wrong-sign literacy gradient on stable-HQ
              (DIFF lowlit - hilit | low-IO, persistent)
         P3: continuous-margin four-way gamma_4 (persistent_level)
       Uses bootstrap t-distributions saved from each WCB. We take the
       standard step-down approach using saved t-boot vectors per
       statistic; joint over-the-three-statistics calibration uses the
       max-t pivot across the three independent state-bootstrap draws.

Task C — Reconcile continuous-margin four-way vs discrete-tercile mid-IO:
   (a) IO-percentile binned characterization: for each of 20 equal-count
       IO bins, compute the conditional mean of `mom * IV * literacy_z *
       ret` (state-FE and month-FE de-meaned). Plot/describe shape.
   (b) Quadratic-IO four-way: augment the continuous four-way with
       `mom * IV * literacy_z * IO^2`. If the quadratic coefficient is
       negative and significant (peak at mid-IO), the inverted-U is real.

Output: output/stage3a/results_v8_diagnostic.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)

from deepen_estimators import (twfe_three_way,
                               wild_cluster_bootstrap_state,
                               stacked_state_group_difference,
                               FOCAL, WEBB)

np.random.seed(42)

DFM_STABLE_CACHE = os.path.join(EMP, "_dfm_stable_hq.parquet")
DFM_V7_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a",
                        "results_v8_diagnostic.json")

B_WCB = 4999
CHUNK = 200

CUT_DATE = pd.Timestamp("2015-01-01")


def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d


# ============================================================
# Task A: pre/post-2015 mid-IO concentration
# ============================================================
def task_A_pre_post_split(dfm):
    print("\n=== TASK A: pre/post-2015 split on mid-IO concentration ===",
          flush=True)
    t0 = time.time()
    d_all = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  stable-HQ estimation sample: {len(d_all):,} firm-months",
          flush=True)

    out = {}
    for measure, io_col, label in [
        ('persistent', 'io_share_persist',
         'persistent (between-firm) IO measure'),
        ('time_varying', 'io_share',
         'genuinely time-varying IO measure'),
    ]:
        print(f"\n--- {measure} ({label}) ---", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        # Build IO terciles on FULL stable-HQ (so tercile definitions are
        # consistent across pre/post subsamples).
        d_io = add_io_terciles(d_io, io_col)

        out[measure] = {"io_measure": label, "splits": {}}
        for period, mask in [
            ("pre_2015", d_io['date'] < CUT_DATE),
            ("post_2015", d_io['date'] >= CUT_DATE),
        ]:
            d_per = d_io[mask].copy()
            print(f"  {period}: {len(d_per):,} firm-months, "
                  f"{d_per['permno'].nunique()} permnos, "
                  f"date {d_per['date'].min().date()}..{d_per['date'].max().date()}",
                  flush=True)
            per_tercile = {}
            for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
                sub = d_per[d_per['io_grp'] == g]
                if len(sub) < 100 or sub['hq_state'].nunique() < 3:
                    per_tercile[g] = {"n_obs": int(len(sub)),
                                      "skip": "insufficient data"}
                    continue
                r = twfe_three_way(sub)
                per_tercile[g] = {k: r[k] for k in
                                  ['coef', 'se', 't', 'se_state', 't_state',
                                   'n_obs', 'se_kind']}
                print(f"    {g}: gamma={r['coef']:+.6f} "
                      f"state_t={r['t_state']:.2f} cgm_t={r['t']:.2f} "
                      f"n={r['n_obs']:,}", flush=True)
            out[measure]["splits"][period] = per_tercile

    out["_meta"] = {"elapsed_s": round(time.time() - t0, 2)}
    print(f"  Task A elapsed: {out['_meta']['elapsed_s']:.1f}s", flush=True)
    return out


# ============================================================
# Task B: WCB on mid-IO coefficient + Romano-Wolf
# ============================================================
def task_B_wcb_mid_and_rw(dfm, dfm_full_v7):
    """(1) WCB on mid-IO coef itself; (2) RW correction across 3 patterns."""
    print("\n=== TASK B: WCB on mid-IO coef + Romano-Wolf correction ===",
          flush=True)
    t0 = time.time()
    d_all = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    out = {"part1_wcb_mid_IO_coef": {},
           "part2_romano_wolf": {},
           "_meta": {}}

    # --- Part 1: WCB on mid-IO coefficient ---
    print("\n  --- Part 1: WCB on mid-IO coefficient itself ---", flush=True)

    # Save t-boot per pattern for RW reuse
    tboot_storage = {}

    for measure, io_col, label in [
        ('persistent', 'io_share_persist',
         'persistent (between-firm) IO measure'),
        ('time_varying', 'io_share',
         'genuinely time-varying IO measure'),
    ]:
        print(f"    {measure}: ...", flush=True)
        d_io = d_all[d_all[io_col].notna()].copy()
        d_io = add_io_terciles(d_io, io_col)
        d_mid = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
        print(f"      mid-IO: n={len(d_mid):,}, "
              f"permnos={d_mid['permno'].nunique()}, "
              f"states={d_mid['hq_state'].nunique()}", flush=True)
        wcb = wild_cluster_bootstrap_state(d_mid, B=B_WCB, seed=42)
        out["part1_wcb_mid_IO_coef"][measure] = {
            "io_measure": label,
            "coef": wcb["coef"],
            "se_state": wcb["se_state"],
            "t_obs": wcb["t_obs"],
            "wcb_p_value": wcb["p_value"],
            "ci_studentized": wcb["ci_studentized"],
            "n_obs": wcb["n_obs"],
            "n_state_clusters": wcb["n_state_clusters"],
            "B": wcb["B"],
        }
        print(f"      mid-IO {measure}: gamma={wcb['coef']:+.6f} "
              f"state-t={wcb['t_obs']:.2f} wcb-p={wcb['p_value']:.4f}",
              flush=True)
        if measure == "persistent":
            # Store the t-boot for RW (persistent is the "headline" in RW)
            tboot_storage["P1_mid_IO_persistent"] = {
                "t_obs": wcb["t_obs"],
                "p_value": wcb["p_value"],
                # We need t_boot vector — re-run with t_boot accessible
                # via raw inline WCB. The function returns quantiles only.
                # Re-fire below with t_boot collection.
            }

    # --- Part 2: Romano-Wolf step-down across 3 patterns ---
    print("\n  --- Part 2: Romano-Wolf step-down across 3 patterns ---",
          flush=True)

    # Pattern 1: mid-IO three-way (persistent) — t_boot vector
    print("    P1: mid-IO three-way (persistent) — collecting t_boot ...",
          flush=True)
    d_io = d_all[d_all['io_share_persist'].notna()].copy()
    d_io = add_io_terciles(d_io, 'io_share_persist')
    d_mid_pers = d_io[d_io['io_grp'] == 'IO2_mid'].copy()
    p1_obs, p1_tboot = _wcb_t_boot(d_mid_pers, 'three_way', None, B=B_WCB,
                                   seed=101)
    print(f"      P1 t_obs={p1_obs:.3f}, |t_obs|={abs(p1_obs):.3f}, "
          f"raw_p={float(np.mean(np.abs(p1_tboot) >= abs(p1_obs))):.4f}",
          flush=True)

    # Pattern 2: within-retail wrong-sign literacy gradient on stable-HQ
    # (DIFF lowlit - hilit | low-IO, persistent)
    print("    P2: within-retail literacy gradient (low-IO, persistent) — "
          "collecting t_boot ...", flush=True)
    d_lowio = d_io[d_io['io_grp'] == 'IO1_low'].copy()
    state_lit = (d_lowio.groupby('hq_state')
                 ['literacy_score_corrected'].mean())
    lit_median = state_lit.median()
    hi_lit_states = set(state_lit[state_lit >= lit_median].index)
    lo_lit_states = set(state_lit[state_lit < lit_median].index)
    d_lowio = d_lowio.copy()
    d_lowio['lit_grp'] = np.where(
        d_lowio['hq_state'].isin(lo_lit_states), 'low_lit', 'high_lit')
    p2_obs, p2_tboot = _stacked_group_t_boot(
        d_lowio, 'lit_grp', 'low_lit', B=B_WCB, seed=102)
    print(f"      P2 t_obs={p2_obs:.3f}, |t_obs|={abs(p2_obs):.3f}, "
          f"raw_p={float(np.mean(np.abs(p2_tboot) >= abs(p2_obs))):.4f}",
          flush=True)

    # Pattern 3: continuous-margin four-way gamma_4 (persistent_level)
    # — uses the FULL v7 panel (because the original four-way result is
    # measured on the full panel, not stable-HQ).
    print("    P3: continuous-margin four-way gamma_4 (persistent_level) — "
          "collecting t_boot ...", flush=True)
    d_v7 = dfm_full_v7.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    d_v7 = d_v7[d_v7['io_share_persist'].notna()].copy()
    p3_obs, p3_tboot = _four_way_t_boot(
        d_v7, 'io_share_persist', B=B_WCB, seed=103)
    print(f"      P3 t_obs={p3_obs:.3f}, |t_obs|={abs(p3_obs):.3f}, "
          f"raw_p={float(np.mean(np.abs(p3_tboot) >= abs(p3_obs))):.4f}",
          flush=True)

    # RW step-down: sort patterns by descending |t_obs|; family-wise p
    # at each step is the empirical probability that
    # max_{j >= step} |t_j_boot| >= |t_(j)_obs|.
    # We aggregate independent t_boot vectors with column-wise max across
    # the remaining set, which is the Romano-Wolf max-t pivot (Romano &
    # Wolf 2005). Since the three bootstraps use different seeds, the
    # bootstrap structures are independent — this is a CONSERVATIVE RW
    # because the empirical correlation across patterns is set to zero;
    # the true RW with joint resampling would be tighter. We note this.
    patterns = [
        {"name": "P1_mid_IO_three_way_persistent", "t_obs": p1_obs,
         "t_boot": p1_tboot, "raw_p":
         float(np.mean(np.abs(p1_tboot) >= abs(p1_obs)))},
        {"name": "P2_within_retail_literacy_DIFF", "t_obs": p2_obs,
         "t_boot": p2_tboot, "raw_p":
         float(np.mean(np.abs(p2_tboot) >= abs(p2_obs)))},
        {"name": "P3_continuous_four_way_persistent", "t_obs": p3_obs,
         "t_boot": p3_tboot, "raw_p":
         float(np.mean(np.abs(p3_tboot) >= abs(p3_obs)))},
    ]
    # Sort descending by |t_obs|
    patterns_sorted = sorted(patterns, key=lambda p: -abs(p["t_obs"]))

    # Build n_boot x K matrix of |t_boot|
    K = len(patterns_sorted)
    B_min = min(len(p["t_boot"]) for p in patterns_sorted)
    T = np.column_stack([np.abs(p["t_boot"][:B_min])
                         for p in patterns_sorted])  # B_min x K
    rw_results = []
    p_prev = 0.0
    for k in range(K):
        # Subset of patterns with rank >= k (i.e., k, k+1, ..., K-1)
        max_t = T[:, k:].max(axis=1)
        p_step = float(np.mean(max_t >= abs(patterns_sorted[k]["t_obs"])))
        # Monotonicity enforcement: RW p must be non-decreasing
        p_adj = max(p_step, p_prev)
        p_prev = p_adj
        rw_results.append({
            "rank": k + 1,
            "pattern": patterns_sorted[k]["name"],
            "t_obs": patterns_sorted[k]["t_obs"],
            "abs_t_obs": abs(patterns_sorted[k]["t_obs"]),
            "raw_p": patterns_sorted[k]["raw_p"],
            "rw_adjusted_p": p_adj,
        })
        print(f"      RW step {k+1}: {patterns_sorted[k]['name']} "
              f"|t|={abs(patterns_sorted[k]['t_obs']):.3f} "
              f"raw_p={patterns_sorted[k]['raw_p']:.4f} -> "
              f"adj_p={p_adj:.4f}", flush=True)

    out["part2_romano_wolf"] = {
        "method": ("Romano-Wolf step-down with independent state-cluster "
                   "wild bootstraps per pattern. Patterns ordered by "
                   "descending |t_obs|. Family-wise p at step k is the "
                   "empirical Prob(max_{j>=k} |t_j_boot| >= |t_(k)_obs|), "
                   "monotonized to be non-decreasing. CONSERVATIVE: "
                   "independent bootstraps ignore positive cross-pattern "
                   "dependence that would tighten the correction."),
        "patterns_ordered_by_abs_t": rw_results,
        "B_per_pattern": B_WCB,
        "B_used_in_RW_max": int(B_min),
        "seed_p1": 101, "seed_p2": 102, "seed_p3": 103,
    }

    out["_meta"] = {"elapsed_s": round(time.time() - t0, 2)}
    print(f"  Task B elapsed: {out['_meta']['elapsed_s']:.1f}s", flush=True)
    return out


# ============================================================
# Task C: Reconcile continuous four-way vs discrete tercile mid-IO
# ============================================================
def task_C_reconcile(dfm_full_v7):
    print("\n=== TASK C: reconcile continuous four-way vs discrete mid-IO "
          "===", flush=True)
    t0 = time.time()
    d_v7 = dfm_full_v7.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()

    out = {"_meta": {}}

    # --- Part (a): IO-percentile binned conditional mean ---
    print("\n  --- Part (a): 20-bin IO conditional mean of "
          "(mom*IV*lit*ret) | FE-residualized ---", flush=True)
    d_pers = d_v7[d_v7['io_share_persist'].notna()].copy()
    n = len(d_pers)
    print(f"    sample: n={n:,}, permnos={d_pers['permno'].nunique()}",
          flush=True)

    # Build the focal four-way product mom*IV*lit and the LHS ret,
    # de-mean both by state+month FE (the simplest residualization).
    # Then within each IO bin, the residualized conditional mean of
    # mom*IV*lit * ret traces out the IO-dependent three-way slope.
    mom = d_pers['mom_12_2'].values.astype(float)
    iv = d_pers['iv'].values.astype(float)
    lit = d_pers['literacy_score_corrected'].values.astype(float)
    threeway_focal = mom * iv * lit
    y = d_pers['ret'].values.astype(float)
    sc = pd.Categorical(d_pers['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d_pers['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1

    # FE residualize both x (the three-way focal product) and y (return)
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    # Also include the 6 lower-order focal terms (the headline 7-term
    # spec) so the residualization isolates the three-way slope.
    lo = d_pers[FOCAL[:6]].values.astype(float)
    Lo = sp.csr_matrix(lo)
    W = sp.hstack([ones, Lo, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)

    def partial(v):
        return v - W @ (WtW_inv @ (W.T @ v))

    threeway_t = partial(threeway_focal)
    y_t = partial(y)

    # 20 bins on persistent IO percentile (firm-level rank, persistent IO is
    # firm-constant within sample)
    perm_io = d_pers.groupby('permno')['io_share_persist'].first()
    perm_pct = perm_io.rank(pct=True, method='average')
    d_pers['io_pct'] = d_pers['permno'].map(perm_pct).astype(float)

    nb = 20
    bin_edges = np.linspace(0, 1, nb + 1)
    bin_assign = np.clip(np.digitize(d_pers['io_pct'].values, bin_edges[1:-1]),
                         0, nb - 1)
    bins = []
    for b in range(nb):
        mask = (bin_assign == b)
        if mask.sum() < 50:
            continue
        # Conditional three-way slope in this IO bin via simple OLS of
        # y_t on threeway_t (already partial-out'd of FE+lower-order).
        xb = threeway_t[mask]
        yb = y_t[mask]
        Sxx = float(xb @ xb)
        slope = float(xb @ yb) / Sxx if Sxx > 0 else np.nan
        io_mean = float(d_pers.loc[mask, 'io_share_persist'].mean())
        bins.append({
            "bin": b + 1,
            "io_pct_lo": float(bin_edges[b]),
            "io_pct_hi": float(bin_edges[b + 1]),
            "io_share_mean": io_mean,
            "conditional_three_way_slope": slope,
            "n_obs": int(mask.sum()),
        })
    out["part_a_io_bin_characterization"] = {
        "method": ("FE+lower-order partial-out then bin-conditional OLS "
                   "slope of y on mom*IV*lit. With persistent (firm-level) "
                   "IO, bins are firm cohorts."),
        "n_bins": nb,
        "bins": bins,
        "shape_summary": _summarize_bin_shape(bins),
    }
    bs = out["part_a_io_bin_characterization"]["shape_summary"]
    print(f"    bin shape: min slope at bin {bs['min_bin']} "
          f"({bs['min_slope']:+.5f}), "
          f"max slope at bin {bs['max_bin']} ({bs['max_slope']:+.5f}), "
          f"inverted-U candidate: {bs['inverted_U_candidate']}",
          flush=True)

    # --- Part (b): Quadratic-IO four-way ---
    print("\n  --- Part (b): Quadratic-IO four-way ---", flush=True)
    quad = _quadratic_four_way(d_v7, 'io_share_persist',
                               label='persistent_level',
                               B=B_WCB, seed=204)
    out["part_b_quadratic_four_way_persistent"] = quad
    print(f"    persistent quadratic: gamma_4 (linear) = "
          f"{quad['gamma_4_linear']:+.6f} (state-t="
          f"{quad['t_state_clustered_linear']:.2f}, "
          f"wcb-p={quad['wcb_p_value_linear']:.4f}); "
          f"gamma_5 (quadratic) = "
          f"{quad['gamma_5_quadratic']:+.6f} (state-t="
          f"{quad['t_state_clustered_quadratic']:.2f}, "
          f"wcb-p={quad['wcb_p_value_quadratic']:.4f})", flush=True)

    quad_tv = _quadratic_four_way(d_v7, 'io_share',
                                  label='time_varying_level',
                                  B=B_WCB, seed=205)
    out["part_b_quadratic_four_way_time_varying"] = quad_tv
    print(f"    time-varying quadratic: gamma_4 (linear) = "
          f"{quad_tv['gamma_4_linear']:+.6f} (state-t="
          f"{quad_tv['t_state_clustered_linear']:.2f}, "
          f"wcb-p={quad_tv['wcb_p_value_linear']:.4f}); "
          f"gamma_5 (quadratic) = "
          f"{quad_tv['gamma_5_quadratic']:+.6f} (state-t="
          f"{quad_tv['t_state_clustered_quadratic']:.2f}, "
          f"wcb-p={quad_tv['wcb_p_value_quadratic']:.4f})", flush=True)

    # Verdict on inverted-U:
    sig_thresh = 0.10
    inverted_U_pers = (quad['gamma_5_quadratic'] < 0 and
                       quad['wcb_p_value_quadratic'] < sig_thresh)
    inverted_U_tv = (quad_tv['gamma_5_quadratic'] < 0 and
                     quad_tv['wcb_p_value_quadratic'] < sig_thresh)
    out["reconciliation_verdict"] = {
        "rule": ("Inverted-U (peak at mid-IO) confirmed if quadratic "
                 "coefficient gamma_5 < 0 with wcb p < 0.10."),
        "persistent_inverted_U_confirmed": inverted_U_pers,
        "time_varying_inverted_U_confirmed": inverted_U_tv,
        "bin_shape_inverted_U_candidate":
            out["part_a_io_bin_characterization"]["shape_summary"][
                "inverted_U_candidate"],
    }
    out["_meta"] = {"elapsed_s": round(time.time() - t0, 2)}
    print(f"  Task C elapsed: {out['_meta']['elapsed_s']:.1f}s", flush=True)
    return out


# ============================================================
# Helpers
# ============================================================
def _summarize_bin_shape(bins):
    if not bins:
        return {"min_bin": None, "max_bin": None,
                "inverted_U_candidate": False}
    slopes = np.array([b["conditional_three_way_slope"] for b in bins])
    n = len(slopes)
    min_idx = int(np.nanargmin(slopes))
    max_idx = int(np.nanargmax(slopes))
    # Inverted-U candidate for the three-way SLOPE: most-negative slope
    # is interior (not at bin 1 or bin n)
    interior = 0 < min_idx < n - 1
    # Compare interior min vs the two ends
    end_avg = 0.5 * (slopes[0] + slopes[-1])
    inverted = interior and slopes[min_idx] < end_avg
    return {
        "min_bin": min_idx + 1,
        "min_slope": float(slopes[min_idx]),
        "max_bin": max_idx + 1,
        "max_slope": float(slopes[max_idx]),
        "first_bin_slope": float(slopes[0]),
        "last_bin_slope": float(slopes[-1]),
        "ends_avg_slope": float(end_avg),
        "inverted_U_candidate": bool(inverted),
        "notes": ("inverted_U_candidate=True if argmin is interior bin "
                  "AND most-negative slope is more negative than the "
                  "average of the two endpoint slopes — consistent with "
                  "a mid-IO trough in the three-way slope."),
    }


def _cluster_mat(codes):
    n = len(codes)
    ug = np.unique(codes)
    idx = np.searchsorted(ug, codes)
    return sp.csr_matrix((np.ones(n), (np.arange(n), idx)),
                         shape=(n, len(ug))), ug


def _wcb_t_boot(d, kind, group_args, B=4999, seed=42):
    """Three-way WCB, returns (t_obs, t_boot vector). Mirrors
    wild_cluster_bootstrap_state but exposes t_boot for RW max-t."""
    rng = np.random.RandomState(seed)
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    x3 = focal[:, 6]
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(focal[:, :6])
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, W6s, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    x3t = x3 - W @ np.linalg.solve(WtW, W.T @ x3)
    yt = y - W @ np.linalg.solve(WtW, W.T @ y)
    Sxx = x3t @ x3t
    coef = float((x3t @ yt) / Sxx)
    Cs, states = _cluster_mat(sc)
    score = x3t * (yt - coef * x3t)
    se_state = float(np.sqrt(((Cs.T @ score) ** 2).sum()) / Sxx)
    t_obs = coef / se_state if se_state > 0 else np.nan
    e_r = yt.copy()
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    t_boot = np.empty(B)
    done = 0
    while done < B:
        b = min(CHUNK, B - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, b))]
        w = w_state[state_idx, :]
        Y = e_r[:, None] * w
        coef_b = (x3t @ Y) / Sxx
        E_b = Y - x3t[:, None] * coef_b[None, :]
        Score_b = x3t[:, None] * E_b
        cl_b = Cs.T @ Score_b
        se_b = np.sqrt((cl_b ** 2).sum(axis=0)) / Sxx
        t_boot[done:done + b] = np.where(se_b > 0, coef_b / se_b, 0.0)
        done += b
    return float(t_obs), t_boot


def _stacked_group_t_boot(d, group_col, low_label, B=4999, seed=42):
    """State-defined group difference test — returns (t_obs, t_boot)."""
    rng = np.random.RandomState(seed)
    n = len(d)
    grp = d[group_col].values
    g_low = (grp == low_label).astype(float)
    focal = d[FOCAL].values.astype(float)
    three = focal[:, 6]
    x_int = g_low * three
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(focal[:, :6])
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, W6s, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)

    def partial(v):
        return v - W @ (WtW_inv @ (W.T @ v))

    X = np.column_stack([three, x_int])
    Xt = np.column_stack([partial(X[:, 0]), partial(X[:, 1])])
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    Cs, states = _cluster_mat(sc)
    S = np.asarray(Cs.T @ (Xt * e[:, None]))
    V_state = XtX_inv @ (S.T @ S) @ XtX_inv
    diff_coef = float(beta[1])
    se_state = float(np.sqrt(V_state[1, 1]))
    t_obs = diff_coef / se_state if se_state > 0 else np.nan

    # restricted bootstrap
    Xt0 = Xt[:, [0]]
    b0 = float((Xt0.T @ yt) / (Xt0.T @ Xt0))
    fit0 = Xt0.flatten() * b0
    e_r = yt - fit0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    a10, a11 = XtX_inv[1, 0], XtX_inv[1, 1]
    Xt0c, Xt1c = Xt[:, 0], Xt[:, 1]
    t_boot = np.empty(B)
    done = 0
    while done < B:
        bb = min(CHUNK, B - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, bb))]
        w = w_state[state_idx, :]
        Y = fit0[:, None] + e_r[:, None] * w
        Beta_b = XtX_inv @ (Xt.T @ Y)
        E_b = Y - Xt @ Beta_b
        S0 = Cs.T @ (Xt0c[:, None] * E_b)
        S1 = Cs.T @ (Xt1c[:, None] * E_b)
        m00 = (S0 ** 2).sum(axis=0)
        m01 = (S0 * S1).sum(axis=0)
        m11 = (S1 ** 2).sum(axis=0)
        V11_b = a10 * a10 * m00 + 2 * a10 * a11 * m01 + a11 * a11 * m11
        se_b = np.sqrt(np.where(V11_b > 0, V11_b, np.nan))
        tb = np.where(se_b > 0, Beta_b[1, :] / se_b, 0.0)
        t_boot[done:done + bb] = np.nan_to_num(tb, nan=0.0)
        done += bb
    return float(t_obs), t_boot


def _four_way_t_boot(d, io_col, B=4999, seed=42):
    """Continuous four-way mom*IV*lit*IO, with 15 lower-order terms +
    state+month FE. Returns (t_obs, t_boot) on gamma_4."""
    rng = np.random.RandomState(seed)
    d = d[d[io_col].notna()].copy()
    n = len(d)
    mom = d['mom_12_2'].values.astype(float)
    iv = d['iv'].values.astype(float)
    lit = d['literacy_score_corrected'].values.astype(float)
    IO = d[io_col].values.astype(float)
    y = d['ret'].values.astype(float)
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
    Cs, states = _cluster_mat(sc)
    score = four_t * (yt - coef * four_t)
    se_state = float(np.sqrt(((Cs.T @ score) ** 2).sum()) / Sxx)
    t_obs = coef / se_state if se_state > 0 else np.nan
    e_r = yt.copy()
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    t_boot = np.empty(B)
    done = 0
    while done < B:
        b = min(CHUNK, B - done)
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
    return float(t_obs), t_boot


def _quadratic_four_way(d, io_col, label, B=4999, seed=42):
    """Quadratic-IO four-way: includes both `mom*IV*lit*IO` and
    `mom*IV*lit*IO^2`. Reports inference on BOTH gamma_4 (linear) and
    gamma_5 (quadratic) using state-clustered SE and restricted WCB
    (separate H0 for each)."""
    rng = np.random.RandomState(seed)
    d = d[d[io_col].notna()].copy()
    n = len(d)
    mom = d['mom_12_2'].values.astype(float)
    iv = d['iv'].values.astype(float)
    lit = d['literacy_score_corrected'].values.astype(float)
    IO = d[io_col].values.astype(float)
    IO2 = IO * IO
    y = d['ret'].values.astype(float)
    # Lower-order: mom, iv, lit, IO, IO2,
    #              mom*iv, mom*lit, mom*IO, mom*IO2,
    #              iv*lit, iv*IO, iv*IO2, lit*IO, lit*IO2,
    #              mom*iv*lit, mom*iv*IO, mom*iv*IO2,
    #              mom*lit*IO, mom*lit*IO2, iv*lit*IO, iv*lit*IO2
    lo = np.column_stack([
        mom, iv, lit, IO, IO2,
        mom * iv, mom * lit, mom * IO, mom * IO2,
        iv * lit, iv * IO, iv * IO2, lit * IO, lit * IO2,
        mom * iv * lit, mom * iv * IO, mom * iv * IO2,
        mom * lit * IO, mom * lit * IO2, iv * lit * IO, iv * lit * IO2,
    ])
    x_lin = mom * iv * lit * IO
    x_quad = mom * iv * lit * IO2
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

    Xt = np.column_stack([partial(x_lin), partial(x_quad)])
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    Cs, states = _cluster_mat(sc)
    S = np.asarray(Cs.T @ (Xt * e[:, None]))  # G x 2
    V_state = XtX_inv @ (S.T @ S) @ XtX_inv
    se_lin = float(np.sqrt(V_state[0, 0]))
    se_quad = float(np.sqrt(V_state[1, 1]))
    coef_lin = float(beta[0])
    coef_quad = float(beta[1])
    t_lin = coef_lin / se_lin if se_lin > 0 else np.nan
    t_quad = coef_quad / se_quad if se_quad > 0 else np.nan

    # WCB on the quadratic coef (H0: gamma_5 = 0): keep linear in null.
    # Restricted residual: y_t - (gamma_4 * x_lin_t)|_{restricted-fit}.
    Xt0 = Xt[:, [0]]
    b0 = float((Xt0.T @ yt) / (Xt0.T @ Xt0))
    fit0 = Xt0.flatten() * b0
    e_r = yt - fit0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    a10, a11 = XtX_inv[1, 0], XtX_inv[1, 1]
    Xt0c, Xt1c = Xt[:, 0], Xt[:, 1]
    t_boot_quad = np.empty(B)
    done = 0
    while done < B:
        bb = min(CHUNK, B - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, bb))]
        w = w_state[state_idx, :]
        Y = fit0[:, None] + e_r[:, None] * w
        Beta_b = XtX_inv @ (Xt.T @ Y)
        E_b = Y - Xt @ Beta_b
        S0 = Cs.T @ (Xt0c[:, None] * E_b)
        S1 = Cs.T @ (Xt1c[:, None] * E_b)
        m00 = (S0 ** 2).sum(axis=0)
        m01 = (S0 * S1).sum(axis=0)
        m11 = (S1 ** 2).sum(axis=0)
        V11_b = a10 * a10 * m00 + 2 * a10 * a11 * m01 + a11 * a11 * m11
        se_b = np.sqrt(np.where(V11_b > 0, V11_b, np.nan))
        tb = np.where(se_b > 0, Beta_b[1, :] / se_b, 0.0)
        t_boot_quad[done:done + bb] = np.nan_to_num(tb, nan=0.0)
        done += bb
    p_wcb_quad = float(np.mean(np.abs(t_boot_quad) >= abs(t_quad)))

    # Also WCB on the linear coef (H0: gamma_4 = 0): keep quadratic in null.
    rng2 = np.random.RandomState(seed + 1)
    Xt0b = Xt[:, [1]]
    b0b = float((Xt0b.T @ yt) / (Xt0b.T @ Xt0b))
    fit0b = Xt0b.flatten() * b0b
    e_rb = yt - fit0b
    a00, a01 = XtX_inv[0, 0], XtX_inv[0, 1]
    t_boot_lin = np.empty(B)
    done = 0
    while done < B:
        bb = min(CHUNK, B - done)
        w_state = WEBB[rng2.randint(0, 6, size=(G, bb))]
        w = w_state[state_idx, :]
        Y = fit0b[:, None] + e_rb[:, None] * w
        Beta_b = XtX_inv @ (Xt.T @ Y)
        E_b = Y - Xt @ Beta_b
        S0 = Cs.T @ (Xt0c[:, None] * E_b)
        S1 = Cs.T @ (Xt1c[:, None] * E_b)
        m00 = (S0 ** 2).sum(axis=0)
        m01 = (S0 * S1).sum(axis=0)
        m11 = (S1 ** 2).sum(axis=0)
        V00_b = a00 * a00 * m00 + 2 * a00 * a01 * m01 + a01 * a01 * m11
        se_b = np.sqrt(np.where(V00_b > 0, V00_b, np.nan))
        tb = np.where(se_b > 0, Beta_b[0, :] / se_b, 0.0)
        t_boot_lin[done:done + bb] = np.nan_to_num(tb, nan=0.0)
        done += bb
    p_wcb_lin = float(np.mean(np.abs(t_boot_lin) >= abs(t_lin)))

    # Implied turning point (if gamma_5 < 0, peak at -gamma_4/(2*gamma_5))
    if coef_quad != 0:
        io_star = -coef_lin / (2 * coef_quad)
    else:
        io_star = None
    return {
        "io_label": label,
        "n_obs": int(n),
        "n_state_clusters": int(G),
        "gamma_4_linear": coef_lin,
        "se_state_clustered_linear": se_lin,
        "t_state_clustered_linear": float(t_lin),
        "wcb_p_value_linear": p_wcb_lin,
        "gamma_5_quadratic": coef_quad,
        "se_state_clustered_quadratic": se_quad,
        "t_state_clustered_quadratic": float(t_quad),
        "wcb_p_value_quadratic": p_wcb_quad,
        "implied_io_turning_point": (float(io_star)
                                     if io_star is not None else None),
        "B": B,
        "spec": ("r ~ 21 lower-order terms (incl IO, IO^2 and all "
                 "lower-order interactions) + state FE + month FE + "
                 "mom*IV*lit*IO + mom*IV*lit*IO^2"),
    }


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=== v8 DIAGNOSTIC THREE (Stage 6 r4 deepen directive) ===",
          flush=True)
    print(f"loading stable-HQ cache: {DFM_STABLE_CACHE}", flush=True)
    dfm = pd.read_parquet(DFM_STABLE_CACHE)
    print(f"  stable-HQ: {len(dfm):,} firm-months, "
          f"{dfm['permno'].nunique()} permnos", flush=True)
    # ensure date column is datetime
    dfm['date'] = pd.to_datetime(dfm['date'])

    print(f"loading full v7 panel: {DFM_V7_CACHE}", flush=True)
    dfm_v7 = pd.read_parquet(DFM_V7_CACHE)
    dfm_v7['date'] = pd.to_datetime(dfm_v7['date'])
    print(f"  full v7: {len(dfm_v7):,} firm-months, "
          f"{dfm_v7['permno'].nunique()} permnos", flush=True)

    results = {
        "task": ("v8 diagnostic three — Stage 6 r4 deepen directive on the "
                 "stable-HQ centerpiece. Tasks A (pre/post-2015 split on "
                 "mid-IO), B (WCB on mid-IO coef + Romano-Wolf across 3 "
                 "patterns), C (reconcile continuous four-way vs discrete "
                 "mid-IO via 20-bin characterization + quadratic-IO "
                 "four-way)."),
        "samples": {
            "stable_hq_firm_months": int(len(dfm)),
            "stable_hq_firms": int(dfm['permno'].nunique()),
            "full_v7_firm_months": int(len(dfm_v7)),
            "full_v7_firms": int(dfm_v7['permno'].nunique()),
        },
        "config": {
            "B_wcb": B_WCB,
            "cut_date": str(CUT_DATE.date()),
            "seed": 42,
        },
    }

    results["task_A_pre_post_2015"] = task_A_pre_post_split(dfm)
    results["task_B_wcb_and_rw"] = task_B_wcb_mid_and_rw(dfm, dfm_v7)
    results["task_C_reconciliation"] = task_C_reconcile(dfm_v7)

    results["_total_elapsed_s"] = round(time.time() - t_start, 2)
    print(f"\n=== TOTAL ELAPSED: {results['_total_elapsed_s']:.1f}s ===",
          flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
