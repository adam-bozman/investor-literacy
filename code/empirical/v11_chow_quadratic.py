# =====================================================================
# v11_chow_quadratic.py
# Formal Chow test (Wald, RSS-F, and wild-cluster bootstrap) for pre/post-2015 equality of the quadratic-IO four-way coefficient gamma_5 on stable-HQ.
#
# Inputs:    standardized firm-month panel (stable-HQ)
# Outputs:   output/stage3a/results_v11_chow.json
# Paper:     T7 tab:form_grid (latest Chow/quadratic functional-form result)
# Run order: see code/00_master.py
# =====================================================================

"""v11 — Chow test for pre/post-2015 equality of gamma_5 (quadratic-IO term)
in the four-way specification on stable-HQ.

Stage 6 r7 [FIX-EMPIRIC] (Comment 6). The v9 pre/post-2015 inference fit
the quadratic-IO four-way SEPARATELY on each subsample and reported
gamma_5^pre = 0.0152 and gamma_5^post = 0.0153 (state-cluster t = 1.49
and 1.76 respectively), with no joint test of equality. This script
runs the FORMAL CHOW TEST: pool the two subsamples into a single
regression with a `post` indicator interacted on the focal regressors
(and all lower-order terms / state FE / month FE), then test the single
restriction gamma_5^pre = gamma_5^post = 0 by:

  (a) joint cluster-robust covariance (state-CR1) of (gamma_5^pre,
      gamma_5^post) and the standard Wald test on the linear restriction
      gamma_5^pre - gamma_5^post = 0; F = t^2.
  (b) RSS-based F test: F = (RSS_R - RSS_U) / q / (RSS_U / (N - k)),
      q = 1 (one restriction), with both RSS_R and RSS_U computed from
      the same stacked design (controls fully interacted with `post`).
  (c) Wild-cluster bootstrap (Webb, state-level, B = 999) p-value on the
      same restriction (gamma_5^pre - gamma_5^post = 0), using restricted
      residuals from the constrained model.

The full design is:
  ret = baseline lower-order terms (21 quadratic-IO four-way terms)
      + (post indicator interacted with each baseline term)
      + state FE + post-interacted state FE
      + month FE
      + gamma_4^pre * (mom*iv*lit*IO)_pre + gamma_4^post * (mom*iv*lit*IO)_post
      + gamma_5^pre * (mom*iv*lit*IO^2)_pre + gamma_5^post * (mom*iv*lit*IO^2)_post

This is equivalent to two stacked separate regressions (each pre/post
side has its own slopes for everything except month FE — month FE are
NOT post-interacted because each calendar month falls entirely on one
side of the 2015 cut), so the Chow F on gamma_5 isolates the structural
break in the quadratic-IO four-way coefficient.

Identifiability note: a separate `post` column is OMITTED from the
design because it is perfectly collinear with the sum of post-side
month dummies (every post observation is in some post month, so sum of
post-month-FE = post). State FE are post-interacted to allow state-
specific intercept shifts across the break.

Output:
  code/empirical/v11_chow_quadratic.py  (this file)
  output/stage3a/results_v11_chow.json
  output/stage3a/_v11_chow.log
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
OUT_JSON = os.path.join(OUT, "results_v11_chow.json")
B_WCB = 999
CUT_DATE = pd.Timestamp("2015-01-01")
CHUNK = 100


def _cluster_mat(codes):
    n = len(codes)
    ug = np.unique(codes)
    idx = np.searchsorted(ug, codes)
    C = sp.csr_matrix((np.ones(n), (np.arange(n), idx)),
                      shape=(n, len(ug)))
    return C, ug


def build_design(d, io_col):
    """Build the Chow design.

    Returns:
      W_full       sparse n x p_full design (intercept + 21 lower-order
                    + (post x each of 21 lower-order)
                    + state FE + post x state FE + month FE)
      x_lin_pre    n-vector (1-post) * (mom*iv*lit*IO)
      x_lin_post   n-vector  post   * (mom*iv*lit*IO)
      x_quad_pre   n-vector (1-post) * (mom*iv*lit*IO^2)
      x_quad_post  n-vector  post   * (mom*iv*lit*IO^2)
      sc           state codes
      mc           month codes
    """
    n = len(d)
    mom = d['mom_12_2'].values.astype(float)
    iv = d['iv'].values.astype(float)
    lit = d['literacy_score_corrected'].values.astype(float)
    IO = d[io_col].values.astype(float)
    IO2 = IO * IO
    post = (d['date'] >= CUT_DATE).values.astype(float)
    pre = 1.0 - post

    # 21 lower-order terms (exact mirror of _quadratic_four_way controls)
    lo = np.column_stack([
        mom, iv, lit, IO, IO2,
        mom * iv, mom * lit, mom * IO, mom * IO2,
        iv * lit, iv * IO, iv * IO2, lit * IO, lit * IO2,
        mom * iv * lit, mom * iv * IO, mom * iv * IO2,
        mom * lit * IO, mom * lit * IO2,
        iv * lit * IO, iv * lit * IO2,
    ])  # n x 21
    # Each lower-order term interacted with post (so each side has its
    # own slope on each control). Together with `lo`, this allows the
    # entire 21-term control set to differ pre vs post — exactly
    # equivalent to two stacked separate regressions for the controls.
    lo_post = lo * post[:, None]

    # Focal regressors: gamma_4 and gamma_5 each split into pre / post
    x_lin = mom * iv * lit * IO
    x_quad = mom * iv * lit * IO2
    x_lin_pre = x_lin * pre
    x_lin_post = x_lin * post
    x_quad_pre = x_quad * pre
    x_quad_post = x_quad * post

    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = int(sc.max()) + 1, int(mc.max()) + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    Lo = sp.csr_matrix(lo)
    LoP = sp.csr_matrix(lo_post)
    # State FE: drop one column to avoid collinearity with intercept
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)),
                       shape=(n, nS))[:, 1:]
    # post x state FE: drop one column to avoid collinearity with the
    # sum of post-side month FEs (which already span the post shift for
    # one reference state)
    Sd_p = sp.csr_matrix((post, (rows, sc)),
                         shape=(n, nS))[:, 1:]
    # Month FE: drop one
    Md = sp.csr_matrix((np.ones(n), (rows, mc)),
                       shape=(n, nM))[:, 1:]
    # NB: month FE are NOT post-interacted because each calendar month
    # falls entirely on one side of CUT_DATE = 2015-01-01.
    # NB: a separate `post` column is OMITTED from the design — it is
    # perfectly collinear with the sum of post-side month FE.

    W_full = sp.hstack([ones, Lo, LoP, Sd, Sd_p, Md]).tocsc()
    return (W_full, x_lin_pre, x_lin_post, x_quad_pre, x_quad_post,
            sc, mc)


def fit_unrestricted(d, io_col):
    """Fit the unrestricted Chow model where gamma_4 and gamma_5 each
    have separate pre/post slopes. Returns dict with the 4 focal coefs,
    their state-CR1 covariance, RSS, and pieces needed for the WCB
    bootstrap.
    """
    n = len(d)
    (W, xlp, xlpo, xqp, xqpo, sc, mc) = build_design(d, io_col)
    focal = np.column_stack([xlp, xlpo, xqp, xqpo])  # 4 columns
    WtW = (W.T @ W).toarray()
    try:
        WtW_inv = np.linalg.inv(WtW)
        condition = float(np.linalg.cond(WtW))
        used_pinv = False
    except np.linalg.LinAlgError:
        WtW_inv = np.linalg.pinv(WtW)
        condition = float('inf')
        used_pinv = True

    def partial(v):
        if v.ndim == 1:
            return v - W @ (WtW_inv @ (W.T @ v))
        cols = []
        for j in range(v.shape[1]):
            cj = v[:, j] - W @ (WtW_inv @ (W.T @ v[:, j]))
            cols.append(cj)
        return np.column_stack(cols)

    y = d['ret'].values.astype(float)
    Xt = partial(focal)
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    rss_u = float(e @ e)

    # state-CR1 covariance for the 4 focal coefs
    Cs, states = _cluster_mat(sc)
    S = np.asarray(Cs.T @ (Xt * e[:, None]))  # G x 4
    V_focal = XtX_inv @ (S.T @ S) @ XtX_inv

    return {
        'W': W, 'Xt': Xt, 'yt': yt, 'beta': beta, 'e': e,
        'XtX_inv': XtX_inv, 'V_focal': V_focal, 'sc': sc,
        'states': states, 'Cs': Cs, 'rss_u': rss_u, 'n': int(n),
        'k_full': W.shape[1] + 4,
        'condition': condition, 'used_pinv': used_pinv,
    }


def fit_restricted_gamma5(d, io_col):
    """Fit the model under H0: gamma_5^pre = gamma_5^post.

    Implementation: replace (xqp, xqpo) by their SUM (= x_quad), with
    one common gamma_5 coefficient. Keep gamma_4^pre and gamma_4^post
    distinct (the test isolates the gamma_5 restriction).

    Returns rss_r and the restricted gamma_5 estimate.
    """
    (W, xlp, xlpo, xqp, xqpo, sc, mc) = build_design(d, io_col)
    x_quad_combined = xqp + xqpo
    focal_r = np.column_stack([xlp, xlpo, x_quad_combined])  # 3 cols
    WtW = (W.T @ W).toarray()
    try:
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        WtW_inv = np.linalg.pinv(WtW)

    def partial(v):
        if v.ndim == 1:
            return v - W @ (WtW_inv @ (W.T @ v))
        cols = []
        for j in range(v.shape[1]):
            cj = v[:, j] - W @ (WtW_inv @ (W.T @ v[:, j]))
            cols.append(cj)
        return np.column_stack(cols)

    y = d['ret'].values.astype(float)
    Xt = partial(focal_r)
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    rss_r = float(e @ e)
    gamma5_common = float(beta[2])
    return {'rss_r': rss_r, 'gamma5_common': gamma5_common,
            'beta_r': beta, 'e_r': e, 'Xt_r': Xt, 'yt_r': yt}


def wcb_chow_gamma5(d, io_col, full, restricted, B=999, seed=42):
    """Wild-cluster bootstrap of the Chow restriction
    gamma_5^pre = gamma_5^post at the state level.

    Approach: under H0, use the restricted residuals e_r from the model
    that imposes gamma_5^pre = gamma_5^post. Resample e_r by multiplying
    by Webb 6-point weights drawn at the STATE level (one weight per
    state cluster), yielding bootstrap y* = y_hat_H0 + e_r * w(s). For
    each bootstrap sample we refit the UNRESTRICTED model and compute:
        t_b = (gamma_5^pre_b - gamma_5^post_b) / se(diff)_b
    where se(diff) is the state-CR1 SE of the contrast on the
    bootstrap sample. The p-value is the empirical share of |t_b| >=
    |t_obs|.
    """
    rng = np.random.RandomState(seed)
    n = len(d)
    Xt_u = full['Xt']
    XtX_inv_u = full['XtX_inv']
    Cs = full['Cs']
    states = full['states']
    sc = full['sc']
    G = len(states)
    state_idx = np.searchsorted(states, sc)

    L = np.array([0.0, 0.0, 1.0, -1.0])  # gamma_5^pre - gamma_5^post

    beta_u = full['beta']
    V_focal = full['V_focal']
    contrast_obs = float(L @ beta_u)
    se_contrast = float(np.sqrt(L @ V_focal @ L))
    t_obs = contrast_obs / se_contrast if se_contrast > 0 else np.nan

    # Restricted residual in the partial-residualized space matches the
    # unrestricted partial space because both share the same W
    # projection. Fitted y under H0:
    e_r = restricted['e_r']
    yt_u = full['yt']
    yt_H0_hat = yt_u - e_r

    t_boot = np.empty(B)
    contrast_boot = np.empty(B)
    XtT = Xt_u.T
    Xt_arr = Xt_u
    done = 0
    while done < B:
        bb = min(CHUNK, B - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, bb))]   # G x bb
        w = w_state[state_idx, :]                          # n x bb
        Y_b = yt_H0_hat[:, None] + e_r[:, None] * w        # n x bb
        Beta_b = XtX_inv_u @ (XtT @ Y_b)                   # 4 x bb
        E_b = Y_b - Xt_arr @ Beta_b                        # n x bb
        contrast_b = L @ Beta_b                            # bb
        se_b = np.empty(bb)
        for jj in range(bb):
            scores_j = Xt_arr * E_b[:, jj:jj + 1]
            S_j = np.asarray(Cs.T @ scores_j)
            V_j = XtX_inv_u @ (S_j.T @ S_j) @ XtX_inv_u
            v_c = float(L @ V_j @ L)
            se_b[jj] = np.sqrt(max(v_c, 0))
        t_b = np.where(se_b > 0, contrast_b / se_b, 0.0)
        t_boot[done:done + bb] = np.nan_to_num(t_b, nan=0.0)
        contrast_boot[done:done + bb] = contrast_b
        done += bb

    p_wcb = float(np.mean(np.abs(t_boot) >= abs(t_obs)))
    q = np.percentile(t_boot, [2.5, 97.5])
    ci_stud = [contrast_obs - q[1] * se_contrast,
               contrast_obs - q[0] * se_contrast]
    return {
        'contrast_obs': contrast_obs,
        'se_contrast_state_CR1': se_contrast,
        't_obs': float(t_obs),
        'wcb_p_value': p_wcb,
        'wcb_ci_studentized': [float(ci_stud[0]), float(ci_stud[1])],
        'wcb_t_boot_q': [float(q[0]), float(np.median(t_boot)),
                         float(q[1])],
        'B': B, 'G': int(G),
    }


def main():
    t0 = time.time()
    print("=== v11 Chow test for gamma_5 pre/post-2015 equality ===",
          flush=True)
    print(f"  CUT_DATE = {CUT_DATE.date()}, B_WCB = {B_WCB}",
          flush=True)

    d = load_stable_hq()
    print(f"  stable-HQ panel: {len(d):,} firm-months", flush=True)

    io_col = 'io_share_persist'
    cols_needed = FOCAL + ['ret', 'hq_state', 'date', 'ym', io_col,
                            'permno']
    d = d.dropna(subset=cols_needed).copy()
    d = d[d[io_col].notna()].copy()
    n_total = len(d)
    n_pre = int((d['date'] < CUT_DATE).sum())
    n_post = int((d['date'] >= CUT_DATE).sum())
    print(f"  estimation sample: n={n_total:,}, "
          f"pre={n_pre:,}, post={n_post:,}", flush=True)
    n_states = int(d['hq_state'].nunique())
    print(f"  states: {n_states}", flush=True)

    results = {
        'task': ("v11 Chow test for gamma_5 pre/post-2015 equality in "
                 "the quadratic-IO four-way on stable-HQ persistent IO"),
        'comment_referenced': "Stage 6 r7 Comment 6 [FIX-EMPIRIC]",
        'cut_date': str(CUT_DATE.date()),
        'io_measure': 'io_share_persist (persistent IO)',
        'sample': {
            'n_total': n_total, 'n_pre': n_pre, 'n_post': n_post,
            'n_states': n_states, 'n_firms': int(d['permno'].nunique()),
        },
        'B_wcb': B_WCB,
        'design': ("21 lower-order quadratic-IO controls (mirror of "
                   "_quadratic_four_way) + (post x each lower-order) "
                   "+ state FE + post x state FE + month FE + 4 focal "
                   "regressors (gamma_4^pre, gamma_4^post, gamma_5^pre, "
                   "gamma_5^post). A standalone post column is omitted "
                   "(collinear with the sum of post-month dummies). "
                   "Restriction tested: gamma_5^pre = gamma_5^post (one "
                   "restriction)."),
    }

    # =============================================
    # Step 1: Fit the unrestricted Chow model
    # =============================================
    print("\n--- Step 1: Unrestricted Chow model ---", flush=True)
    tu = time.time()
    full = fit_unrestricted(d, io_col)
    print(f"  unrestricted fit: {time.time() - tu:.1f}s",
          flush=True)
    print(f"  design columns: {full['W'].shape[1]:,} + 4 focal "
          f"= {full['k_full']:,}", flush=True)
    print(f"  RSS_U = {full['rss_u']:.6f}", flush=True)
    print(f"  condition number: {full['condition']:.3e}, "
          f"used_pinv = {full['used_pinv']}", flush=True)
    beta_u = full['beta']
    V_focal = full['V_focal']
    se_u = np.sqrt(np.diag(V_focal))
    t_u = beta_u / np.where(se_u > 0, se_u, np.nan)
    coef_names = ['gamma4_pre', 'gamma4_post',
                  'gamma5_pre', 'gamma5_post']
    print("\n  Unrestricted focal estimates (state-CR1):")
    print(f"    {'coef':<14} {'estimate':>12} {'se':>10} {'t':>8}")
    for i, nm in enumerate(coef_names):
        print(f"    {nm:<14} {beta_u[i]:>+12.6f} "
              f"{se_u[i]:>10.6f} {t_u[i]:>8.3f}", flush=True)

    unr_table = {nm: {'coef': float(beta_u[i]),
                       'se_state_CR1': float(se_u[i]),
                       't_state_CR1': float(t_u[i])}
                  for i, nm in enumerate(coef_names)}
    results['unrestricted_focal'] = unr_table

    # =============================================
    # Step 2: Fit the restricted model (gamma_5 common)
    # =============================================
    print("\n--- Step 2: Restricted model (gamma_5^pre = gamma_5^post) ---",
          flush=True)
    tr = time.time()
    restr = fit_restricted_gamma5(d, io_col)
    print(f"  restricted fit: {time.time() - tr:.1f}s",
          flush=True)
    print(f"  RSS_R = {restr['rss_r']:.6f}", flush=True)
    print(f"  gamma_5 (common, restricted) = "
          f"{restr['gamma5_common']:+.6f}", flush=True)

    # =============================================
    # Step 3: RSS-based F test
    # =============================================
    from scipy import stats
    rss_u = full['rss_u']
    rss_r = restr['rss_r']
    q_restr = 1
    k_full = full['k_full']
    n_obs = full['n']
    df1 = q_restr
    df2 = n_obs - k_full
    F_rss = ((rss_r - rss_u) / q_restr) / (rss_u / df2)
    p_F_rss = 1.0 - stats.f.cdf(F_rss, df1, df2)
    p_chi2_rss = 1.0 - stats.chi2.cdf(F_rss * q_restr, q_restr)
    print(f"\n  RSS-based F test:")
    print(f"    RSS_R   = {rss_r:.6f}")
    print(f"    RSS_U   = {rss_u:.6f}")
    print(f"    df1 = {df1}, df2 = {df2:,}")
    print(f"    F = {F_rss:.4f}, p (asymp F) = {p_F_rss:.4f}, "
          f"p (asymp chi2) = {p_chi2_rss:.4f}", flush=True)
    results['F_test_RSS_based'] = {
        'rss_R': rss_r, 'rss_U': rss_u, 'df1': df1, 'df2': df2,
        'F_statistic': float(F_rss),
        'p_value_asymptotic_F': float(p_F_rss),
        'p_value_asymptotic_chi2': float(p_chi2_rss),
        'note': ("Standard OLS Chow F under homoskedasticity. Reported "
                 "alongside cluster-robust Wald for context; cluster-"
                 "robust is the primary inference."),
    }

    # =============================================
    # Step 4: Cluster-robust Wald (Chow) test on the contrast
    # =============================================
    print("\n--- Step 3: Cluster-robust Wald (Chow) ---", flush=True)
    L = np.array([0.0, 0.0, 1.0, -1.0])
    contrast = float(L @ beta_u)
    V_contrast = float(L @ V_focal @ L)
    se_contrast = float(np.sqrt(V_contrast))
    t_contrast = contrast / se_contrast if se_contrast > 0 else np.nan
    W_chow = t_contrast ** 2
    p_chow_asymp = 1.0 - stats.chi2.cdf(W_chow, 1)
    p_t_asymp = 2.0 * (1.0 - stats.norm.cdf(abs(t_contrast)))
    print(f"  contrast: gamma_5^pre - gamma_5^post = "
          f"{contrast:+.6f}")
    print(f"  state-CR1 SE: {se_contrast:.6f}")
    print(f"  t-stat (state-CR1): {t_contrast:.4f}")
    print(f"  Wald chi^2(1) = {W_chow:.4f}, p = {p_chow_asymp:.4f}",
          flush=True)
    results['cluster_robust_Wald_Chow'] = {
        'contrast_estimate': contrast,
        'se_state_CR1': se_contrast,
        't_state_CR1': float(t_contrast),
        'wald_chi2_1df': float(W_chow),
        'p_value_asymptotic_wald_chi2': float(p_chow_asymp),
        'p_value_asymptotic_t': float(p_t_asymp),
        'restriction': 'gamma_5^pre - gamma_5^post = 0',
    }

    # =============================================
    # Step 5: Wild-cluster bootstrap on the same contrast
    # =============================================
    print("\n--- Step 4: WCB on the Chow contrast (B = "
          f"{B_WCB}) ---", flush=True)
    tw = time.time()
    wcb = wcb_chow_gamma5(d, io_col, full, restr, B=B_WCB, seed=42)
    print(f"  WCB elapsed: {time.time() - tw:.1f}s")
    print(f"  WCB p-value: {wcb['wcb_p_value']:.4f}")
    print(f"  WCB t-distribution q[2.5, 50, 97.5]: "
          f"{wcb['wcb_t_boot_q']}", flush=True)
    results['WCB_Chow'] = wcb

    # =============================================
    # Step 6: Verdict
    # =============================================
    # Primary p-value: WCB (cluster-robust, small-cluster-valid).
    p_primary = wcb['wcb_p_value']
    if p_primary > 0.10:
        verdict = "FAILS_TO_REJECT"
        interpretation = (
            "gamma_5^pre = gamma_5^post is supported by the formal "
            "Chow test (WCB-p = {:.4f} > 0.10). The structural-stability "
            "claim for the quadratic-IO four-way coefficient across the "
            "2015 break is formally backed; the paper elevates the "
            "structural-stability subsection in §Mechanism per "
            "Comment 9.".format(p_primary))
    elif p_primary >= 0.10 and p_primary < 0.20:
        verdict = "MARGINAL"
        interpretation = (
            "Marginal evidence (0.10 <= WCB-p = {:.4f} < 0.20). The "
            "paper should hedge: report the test honestly, neither "
            "elevate to a structural anchor nor demote to a "
            "footnote.".format(p_primary))
    else:  # p_primary < 0.10
        verdict = "REJECTS_EQUALITY"
        interpretation = (
            "WCB-p = {:.4f} < 0.10 rejects gamma_5^pre = gamma_5^post. "
            "The stability claim must be demoted per the structured "
            "Major 6 path with explicit hedging.".format(p_primary))

    results['verdict'] = verdict
    results['interpretation'] = interpretation
    results['summary'] = {
        'gamma5_pre_unrestricted': float(beta_u[2]),
        'gamma5_post_unrestricted': float(beta_u[3]),
        'gamma5_pre_se_state_CR1': float(se_u[2]),
        'gamma5_post_se_state_CR1': float(se_u[3]),
        'gamma5_pre_t_state_CR1': float(t_u[2]),
        'gamma5_post_t_state_CR1': float(t_u[3]),
        'gamma5_common_restricted': float(restr['gamma5_common']),
        'gamma4_pre_unrestricted': float(beta_u[0]),
        'gamma4_post_unrestricted': float(beta_u[1]),
        'gamma4_pre_se_state_CR1': float(se_u[0]),
        'gamma4_post_se_state_CR1': float(se_u[1]),
        'contrast_estimate': contrast,
        'contrast_se_state_CR1': se_contrast,
        'contrast_t_state_CR1': float(t_contrast),
        'F_statistic_RSS': float(F_rss),
        'p_F_RSS_asymp_F': float(p_F_rss),
        'wald_chi2_state_CR1': float(W_chow),
        'p_wald_state_CR1_asymp_chi2': float(p_chow_asymp),
        'p_wcb_state_CR1': float(wcb['wcb_p_value']),
        'B_wcb': B_WCB,
        'verdict': verdict,
    }
    results['meta'] = {
        'elapsed_s': round(time.time() - t0, 2),
        'seed': 42,
        'B_wcb': B_WCB,
        'cut_date': str(CUT_DATE.date()),
    }

    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {verdict} ===")
    print(f"=== Interpretation: {interpretation}")
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===")


if __name__ == '__main__':
    main()
