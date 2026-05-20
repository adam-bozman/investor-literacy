# =====================================================================
# v10_parametric_grid.py
# Parametric functional-form grid (cubic-IO, log-IO, kink spline, 20-bin conditional means) testing robustness of the non-monotone quadratic-IO headline shape.
#
# Inputs:    standardized firm-month panel (stable-HQ)
# Outputs:   output/stage3a/results_v10_parametric_grid.json
# Paper:     T7 tab:form_grid (functional-form/quadratic grid; earlier than v11_chow_quadratic.py)
# Run order: see code/00_master.py
# =====================================================================

"""v10 TASK 2 [FIX] — Parametric functional-form grid for the quadratic-IO
headline.

Comment 1 (v6 structured referee): 5%-significance of the headline (gamma_5
WCB p = 0.017) load-bears on the quadratic functional form of IO. Test
whether the non-monotone shape survives in alternative parametric
specifications:

  (a) Cubic-IO four-way: mom*IV*lit*(IO + IO^2 + IO^3). Report gamma on
      each polynomial term. Does cubic term add significant content? Does
      gamma_5 on IO^2 remain significant?
  (b) Log-IO transformation: mom*IV*lit*log(IO+eps). Does the literacy
      slope still concentrate at intermediate values of log(IO)?
  (c) Kink-IO spline: mom*IV*lit*(IO + max(IO - 0.3, 0)). Test kink coef.
  (d) 20-bin IO conditional means: extract the conditional mean of
      (lit*IV*mom*ret) within each IO bin on stable-HQ to visualize the
      U-shape robustness.

Output: output/stage3a/results_v10_parametric_grid.json
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

from v9_helpers import load_stable_hq, save_json, OUT
from deepen_estimators import FOCAL, WEBB

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v10_parametric_grid.json")
B_WCB = 4999
CHUNK = 200


def _cluster_mat(codes):
    n = len(codes)
    ug = np.unique(codes)
    idx = np.searchsorted(ug, codes)
    C = sp.csr_matrix((np.ones(n), (np.arange(n), idx)),
                       shape=(n, len(ug)))
    return C, ug


def fit_with_focal_terms(d, io_basis_names, io_basis_values, label,
                         B=4999, seed=42):
    """Fit a four-way regression with arbitrary IO basis.

    Lower-order controls: mom, iv, lit, IO_basis components, and all
    pairwise products with mom, iv, lit, and the three-way mom*iv*lit
    (which is the panel's mom_x_iv_x_literacy_corr — already computed).

    Focal regressors: mom*iv*lit*IO_basis_k for each k in io_basis_names.

    Returns dict with coef and t/p for each focal regressor.
    """
    rng = np.random.RandomState(seed)
    n = len(d)
    mom = d['mom_12_2'].values.astype(float)
    iv = d['iv'].values.astype(float)
    lit = d['literacy_score_corrected'].values.astype(float)
    momiv = d['mom_x_iv'].values.astype(float)
    momlit = d['mom_x_literacy_corr'].values.astype(float)
    ivlit = d['iv_x_literacy_corr'].values.astype(float)
    momivlit = d['mom_x_iv_x_literacy_corr'].values.astype(float)
    y = d['ret'].values.astype(float)
    K = len(io_basis_names)
    IO_basis = np.column_stack(io_basis_values)  # n x K

    # Lower-order construction (mirroring v8 _quadratic_four_way):
    # all main effects, all pairwise interactions, all three-ways
    # involving IO basis terms.
    # Main effects: mom, iv, lit, IO_b1...IO_bK
    # Pairwise with mom: mom*iv, mom*lit, mom*IO_bk
    # Pairwise with iv: iv*lit, iv*IO_bk
    # Pairwise with lit: lit*IO_bk
    # Three-ways NOT including focal: mom*iv*lit, mom*iv*IO_bk,
    #                                  mom*lit*IO_bk, iv*lit*IO_bk
    lo_blocks = [mom, iv, lit]
    for k in range(K):
        lo_blocks.append(IO_basis[:, k])
    lo_blocks.append(momiv)
    lo_blocks.append(momlit)
    for k in range(K):
        lo_blocks.append(mom * IO_basis[:, k])
    lo_blocks.append(ivlit)
    for k in range(K):
        lo_blocks.append(iv * IO_basis[:, k])
    for k in range(K):
        lo_blocks.append(lit * IO_basis[:, k])
    lo_blocks.append(momivlit)
    for k in range(K):
        lo_blocks.append(momiv * IO_basis[:, k])
    for k in range(K):
        lo_blocks.append(momlit * IO_basis[:, k])
    for k in range(K):
        lo_blocks.append(ivlit * IO_basis[:, k])
    lo = np.column_stack(lo_blocks)

    # Focal: K columns of mom*iv*lit*IO_bk
    focal = np.column_stack([momivlit * IO_basis[:, k] for k in range(K)])

    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = int(sc.max()) + 1, int(mc.max()) + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Lo = sp.csr_matrix(lo)
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)),
                       shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)),
                       shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, Lo, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)

    def partial(v):
        if v.ndim == 1:
            return v - W @ (WtW_inv @ (W.T @ v))
        # for matrix v
        cols = []
        for j in range(v.shape[1]):
            cj = v[:, j] - W @ (WtW_inv @ (W.T @ v[:, j]))
            cols.append(cj)
        return np.column_stack(cols)

    Xt = partial(focal)
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    Cs, states = _cluster_mat(sc)
    S = np.asarray(Cs.T @ (Xt * e[:, None]))  # G x K
    V_state = XtX_inv @ (S.T @ S) @ XtX_inv
    se_focal = np.sqrt(np.maximum(np.diag(V_state), 0))
    t_focal = np.where(se_focal > 0, beta / se_focal, np.nan)
    G = len(states)

    # WCB on EACH focal coef separately (each H0: gamma_k = 0). We use
    # restricted residuals leaving other focal coefs FREE in the
    # restricted fit, then run wild bootstrap.
    # For each focal k, restricted fit: leaving all other (K-1) focal
    # regressors as full controls.
    wcb_ps = {}
    for k in range(K):
        # Build a design that keeps all controls + all OTHER focal
        # regressors, but excludes focal_k.
        other_focal_idx = [j for j in range(K) if j != k]
        if other_focal_idx:
            other_focal = focal[:, other_focal_idx]
            W_k_full = sp.hstack(
                [W, sp.csr_matrix(other_focal)]).tocsc()
            WtW_k = (W_k_full.T @ W_k_full).toarray()
            WtW_k_inv = np.linalg.inv(WtW_k)

            def partial_k(v, W_=W_k_full, Wi=WtW_k_inv):
                if v.ndim == 1:
                    return v - W_ @ (Wi @ (W_.T @ v))
                cols = []
                for j in range(v.shape[1]):
                    cj = v[:, j] - W_ @ (Wi @ (W_.T @ v[:, j]))
                    cols.append(cj)
                return np.column_stack(cols)
            Xt_k = partial_k(focal[:, [k]])
            yt_k = partial_k(y)
        else:
            Xt_k = Xt[:, [k]]
            yt_k = yt
        Sxx_k = float(Xt_k.T @ Xt_k)
        coef_k = float((Xt_k.T @ yt_k)[0]) / Sxx_k
        e_k = yt_k - Xt_k.flatten() * coef_k
        # state-cluster SE for the focal
        score_k = Xt_k.flatten() * e_k
        # SE under k-restricted model
        cl_score = Cs.T @ score_k.reshape(-1, 1)
        se_k = float(np.sqrt((cl_score ** 2).sum()) / Sxx_k)
        t_k = coef_k / se_k if se_k > 0 else np.nan
        # Restricted residual under H0: gamma_k = 0
        e_r = yt_k  # since with no contribution from focal_k, the
                    # restricted residual is just yt_k
        # Bootstrap
        state_idx = np.searchsorted(states, sc)
        rng_k = np.random.RandomState(seed + k)
        t_boot = np.empty(B)
        done = 0
        while done < B:
            bb = min(CHUNK, B - done)
            w_state = WEBB[rng_k.randint(0, 6, size=(G, bb))]
            w = w_state[state_idx, :]
            Y = e_r[:, None] * w
            coef_b = (Xt_k.flatten() @ Y) / Sxx_k
            E_b = Y - Xt_k.flatten()[:, None] * coef_b[None, :]
            Score_b = Xt_k.flatten()[:, None] * E_b
            cl_b = Cs.T @ Score_b
            se_b = np.sqrt((cl_b ** 2).sum(axis=0)) / Sxx_k
            tb = np.where(se_b > 0, coef_b / se_b, 0.0)
            t_boot[done:done + bb] = np.nan_to_num(tb, nan=0.0)
            done += bb
        p_k = float(np.mean(np.abs(t_boot) >= abs(t_k)))
        wcb_ps[io_basis_names[k]] = {
            "coef_focal_alone": coef_k,
            "se_focal_alone": se_k,
            "t_focal_alone": float(t_k),
            "wcb_p": p_k,
        }

    out = {
        "label": label,
        "n_obs": int(n),
        "n_state_clusters": int(G),
        "K_focal": int(K),
        "io_basis_names": list(io_basis_names),
        "focal_coefs": {nm: float(beta[i])
                         for i, nm in enumerate(io_basis_names)},
        "focal_se_state": {nm: float(se_focal[i])
                            for i, nm in enumerate(io_basis_names)},
        "focal_t_state": {nm: float(t_focal[i])
                           for i, nm in enumerate(io_basis_names)},
        "focal_wcb": wcb_ps,
    }
    return out


def task_d_bin_means(d, io_col, n_bins=20):
    """Conditional mean of (lit * iv * mom * ret) within each IO bin,
    state-FE and month-FE de-meaned."""
    d = d.dropna(subset=FOCAL + [io_col, 'ret', 'hq_state']).copy()
    n = len(d)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    # composite focal three-way times return
    composite_signed_ret = d['mom_x_iv_x_literacy_corr'].values * d['ret'].values
    # de-mean by state + month FE
    nS, nM = int(sc.max()) + 1, int(mc.max()) + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)),
                       shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)),
                       shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)
    y = composite_signed_ret
    yt = y - W @ (WtW_inv @ (W.T @ y))
    d['composite_ret_demeaned'] = yt

    # Bin by IO
    d['io_bin'] = pd.qcut(d[io_col], n_bins,
                          labels=False, duplicates='drop')
    bin_means = d.groupby('io_bin').agg(
        n_obs=('ret', 'size'),
        mean_io=(io_col, 'mean'),
        mean_composite_ret_demeaned=('composite_ret_demeaned', 'mean'),
        sd_composite_ret_demeaned=('composite_ret_demeaned', 'std'),
    ).reset_index()
    bin_means['se_mean'] = (bin_means['sd_composite_ret_demeaned']
                             / np.sqrt(bin_means['n_obs']))
    return bin_means.to_dict('records')


def main():
    t0 = time.time()
    print("=== v10 TASK 2: parametric IO grid on stable-HQ ===",
          flush=True)
    d = load_stable_hq()
    d = d.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    print(f"  stable-HQ panel: {len(d):,} firm-months")

    results = {
        "task": "v10 TASK 2 — Parametric IO grid on stable-HQ",
        "comment_referenced": "v6 structured Comment 1",
        "sample_pre_io_filter_n": int(len(d)),
    }

    # Persistent IO is the headline
    io_col = 'io_share_persist'
    d_io = d[d[io_col].notna()].copy()
    IO = d_io[io_col].values.astype(float)
    print(f"  n with persistent IO: {len(d_io):,}")
    print(f"  IO range: ({IO.min():.4f}, {IO.max():.4f}) "
          f"mean={IO.mean():.4f} median={np.median(IO):.4f}")
    EPS = 1e-3  # floor for log

    # =============================================
    # Spec A: Cubic-IO four-way
    # =============================================
    print("\n=== Spec A: Cubic-IO four-way ===", flush=True)
    spec_a = fit_with_focal_terms(
        d_io,
        io_basis_names=['IO', 'IO_sq', 'IO_cu'],
        io_basis_values=[IO, IO ** 2, IO ** 3],
        label="cubic_IO_four_way",
        B=B_WCB, seed=301,
    )
    for nm, c in spec_a['focal_coefs'].items():
        wp = spec_a['focal_wcb'][nm]['wcb_p']
        ts = spec_a['focal_t_state'][nm]
        print(f"  {nm}: coef={c:+.6f}, state-t={ts:.3f}, wcb-p={wp:.4f}")
    results['spec_A_cubic'] = spec_a

    # =============================================
    # Spec B: log-IO four-way
    # =============================================
    print("\n=== Spec B: log-IO four-way ===", flush=True)
    log_IO = np.log(np.maximum(IO, EPS))
    spec_b = fit_with_focal_terms(
        d_io,
        io_basis_names=['log_IO'],
        io_basis_values=[log_IO],
        label="log_IO_four_way",
        B=B_WCB, seed=302,
    )
    for nm, c in spec_b['focal_coefs'].items():
        wp = spec_b['focal_wcb'][nm]['wcb_p']
        ts = spec_b['focal_t_state'][nm]
        print(f"  {nm}: coef={c:+.6f}, state-t={ts:.3f}, wcb-p={wp:.4f}")
    results['spec_B_log_IO'] = spec_b

    # =============================================
    # Spec C: kink-IO four-way (kink at IO=0.3)
    # =============================================
    print("\n=== Spec C: Kink-IO four-way (kink at 0.3) ===", flush=True)
    KINK = 0.3
    IO_kink = np.maximum(IO - KINK, 0)
    spec_c = fit_with_focal_terms(
        d_io,
        io_basis_names=['IO', 'IO_kink_at_0p3'],
        io_basis_values=[IO, IO_kink],
        label="kink_IO_at_0p3_four_way",
        B=B_WCB, seed=303,
    )
    for nm, c in spec_c['focal_coefs'].items():
        wp = spec_c['focal_wcb'][nm]['wcb_p']
        ts = spec_c['focal_t_state'][nm]
        print(f"  {nm}: coef={c:+.6f}, state-t={ts:.3f}, wcb-p={wp:.4f}")
    results['spec_C_kink_IO'] = spec_c

    # =============================================
    # Spec D: 20-bin IO conditional means
    # =============================================
    print("\n=== Spec D: 20-bin IO conditional means ===", flush=True)
    bin_means_persist = task_d_bin_means(d_io, io_col, n_bins=20)
    print("  bin  n_obs  mean_io  composite_ret_demeaned  se")
    for b in bin_means_persist:
        print(f"  {b['io_bin']:3.0f}  {b['n_obs']:5d}  "
              f"{b['mean_io']:.4f}  "
              f"{b['mean_composite_ret_demeaned']:+.5f}  "
              f"{b['se_mean']:.5f}")
    results['spec_D_20bin_persistent'] = bin_means_persist

    # =============================================
    # Summary verdict
    # =============================================
    # The headline 5% clearance is on the quadratic gamma_5 = 0.0151
    # with wcb-p = 0.017. We assess whether the cubic IO^2 coef remains
    # significant and whether the kink and log specs are consistent.
    cubic_io_sq_p = spec_a['focal_wcb']['IO_sq']['wcb_p']
    cubic_io_sq_coef = spec_a['focal_coefs']['IO_sq']
    cubic_io_cu_p = spec_a['focal_wcb']['IO_cu']['wcb_p']
    cubic_io_cu_coef = spec_a['focal_coefs']['IO_cu']
    log_IO_p = spec_b['focal_wcb']['log_IO']['wcb_p']
    log_IO_coef = spec_b['focal_coefs']['log_IO']
    kink_p = spec_c['focal_wcb']['IO_kink_at_0p3']['wcb_p']
    kink_coef = spec_c['focal_coefs']['IO_kink_at_0p3']
    linear_p = spec_c['focal_wcb']['IO']['wcb_p']
    linear_coef = spec_c['focal_coefs']['IO']

    summary = {
        "cubic_IO_sq_coef": cubic_io_sq_coef,
        "cubic_IO_sq_wcb_p": cubic_io_sq_p,
        "cubic_IO_cu_coef": cubic_io_cu_coef,
        "cubic_IO_cu_wcb_p": cubic_io_cu_p,
        "log_IO_coef": log_IO_coef,
        "log_IO_wcb_p": log_IO_p,
        "kink_IO_at_0p3_coef": kink_coef,
        "kink_IO_at_0p3_wcb_p": kink_p,
        "kink_IO_linear_term_coef": linear_coef,
        "kink_IO_linear_term_wcb_p": linear_p,
    }

    if (cubic_io_sq_p < 0.05 and cubic_io_sq_coef > 0
            and abs(kink_p) < 0.10 and kink_coef > 0):
        verdict = "ROBUST_TO_FUNCTIONAL_FORM"
    elif cubic_io_sq_p < 0.10 and cubic_io_sq_coef > 0:
        verdict = "WEAKLY_ROBUST_AT_10PCT"
    else:
        verdict = "FORM_FRAGILE"
    summary['verdict'] = verdict
    results['summary'] = summary
    results['meta'] = {"elapsed_s": round(time.time() - t0, 2),
                       "seed": 42, "B_wcb": B_WCB}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===")
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===")


if __name__ == '__main__':
    main()
