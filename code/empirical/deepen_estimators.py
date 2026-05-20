# =====================================================================
# deepen_estimators.py
# Shared TWFE state+month three-way panel estimator with two-way CGM, state-clustered CR1, and restricted Webb wild-cluster-bootstrap inference, plus the stacked IO difference-in-slopes design.
#
# Inputs:    imported as a module (no direct file reads)
# Outputs:   printed diagnostics only
# Paper:     Shared library module — estimators imported by 31 scripts (no direct table)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive r1 — shared estimators (items b, c, d).

TWFE state+month panel estimator for the three-way coefficient, with:
  - two-way state x month CGM clustering (the seed headline SE),
  - one-way state-clustered CR1 SE (always PSD),
  - restricted (impose-the-null) wild-cluster bootstrap at the STATE level,
    Webb 6-point weights — the credible inference procedure.

STACKED specification for the IO difference test (item b). Per the deepen
directive: a single regression with the IO group fully interacted with the
three-way. The `low_IO x three_way` coefficient IS the (low - high) difference in
the three-way slope, and its SE is state-clustered / two-way CGM / wild-cluster
bootstrapped on the STACKED sample so the shared state/month structure is
respected (round-0 differenced two disjoint-subsample SEs assuming independence).

The IO group is interacted with ALL focal terms (the 6 lower-order focal regressors
AND the three-way), so the three-way difference is not contaminated by the
restriction that lower-order slopes be common across groups. State and month FE
are common (identification stays within-state-month); this is the standard
fully-interacted difference-in-slopes design. With common FE the `low x three_way`
coefficient equals the difference of the two per-group three-way slopes that one
would get from separate regressions sharing the same FE grid.

Performance: the wild-cluster bootstrap is chunked over B. Cluster sums use a
single sparse matmul (C.T @ scores).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)

FOCAL = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
         'mom_x_literacy_corr', 'iv_x_literacy_corr',
         'mom_x_iv_x_literacy_corr']

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                 np.sqrt(0.5), 1.0, np.sqrt(1.5)])

CHUNK = 400  # bootstrap reps per chunk; n x CHUNK arrays stay < ~1 GB for n~300k


def _cluster_matrix(codes):
    """n x G sparse 0/1 cluster-indicator matrix from integer cluster codes."""
    n = len(codes)
    ug = np.unique(codes)
    idx = np.searchsorted(ug, codes)
    C = sp.csr_matrix((np.ones(n), (np.arange(n), idx)), shape=(n, len(ug)))
    return C, ug


def _design(d, extra=None):
    """intercept + 6 lower-order focal + state FE + month FE (+ optional extra
    dense columns). Returns (W csc, state_codes, month_codes, nS, nM)."""
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(focal[:, :6])
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    blocks = [ones, W6s, Sd, Md]
    if extra is not None and extra.shape[1] > 0:
        blocks.append(sp.csr_matrix(extra))
    return sp.hstack(blocks).tocsc(), sc, mc, nS, nM


def twfe_three_way(d):
    """TWFE state+month FE; three-way coefficient. Two-way CGM SE with one-way
    state-clustered CR1 fallback. Returns dict."""
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    x3 = focal[:, 6]
    y = d['ret'].values.astype(float)
    W, sc, mc, nS, nM = _design(d)
    WtW = (W.T @ W).toarray()
    x3t = x3 - W @ np.linalg.solve(WtW, W.T @ x3)
    yt = y - W @ np.linalg.solve(WtW, W.T @ y)
    Sxx = x3t @ x3t
    coef = float((x3t @ yt) / Sxx)
    e = yt - coef * x3t
    score = x3t * e
    Cs, _ = _cluster_matrix(sc)
    Cm, _ = _cluster_matrix(mc)
    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    Ci, _ = _cluster_matrix(inter)
    ss_s = float(((Cs.T @ score) ** 2).sum())
    ss_m = float(((Cm.T @ score) ** 2).sum())
    ss_i = float(((Ci.T @ score) ** 2).sum())
    meat_cgm = ss_s + ss_m - ss_i
    se_state = float(np.sqrt(ss_s) / Sxx)
    if meat_cgm > 0:
        se = float(np.sqrt(meat_cgm) / Sxx)
        se_kind = "cgm_two_way"
    else:
        se = se_state
        se_kind = "state_clustered_CR1_fallback"
    return {"coef": coef, "se": se, "t": coef / se if se > 0 else np.nan,
            "se_state": se_state,
            "t_state": coef / se_state if se_state > 0 else np.nan,
            "n_obs": int(n), "se_kind": se_kind, "var": se ** 2}


def wild_cluster_bootstrap_state(d, B=4999, seed=42):
    """Restricted (H0: beta3=0) wild-cluster bootstrap at the STATE level for the
    three-way coefficient. Webb 6-point weights, chunked over B."""
    rng = np.random.RandomState(seed)
    n = len(d)
    focal = d[FOCAL].values.astype(float)
    x3 = focal[:, 6]
    y = d['ret'].values.astype(float)
    W, sc, mc, nS, nM = _design(d)
    WtW = (W.T @ W).toarray()
    x3t = x3 - W @ np.linalg.solve(WtW, W.T @ x3)
    yt = y - W @ np.linalg.solve(WtW, W.T @ y)
    Sxx = x3t @ x3t
    coef = float((x3t @ yt) / Sxx)
    e = yt - coef * x3t
    Cs, states = _cluster_matrix(sc)
    score = x3t * e
    se_state = float(np.sqrt(((Cs.T @ score) ** 2).sum()) / Sxx)
    t_obs = coef / se_state if se_state > 0 else np.nan
    e_r = yt.copy()           # restricted residuals under H0: beta3 = 0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    t_boot = np.empty(B)
    coef_boot = np.empty(B)
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
        coef_boot[done:done + b] = coef_b
        done += b
    p_val = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))
    q = np.percentile(t_boot, [2.5, 97.5])
    ci_stud = [coef - q[1] * se_state, coef - q[0] * se_state]
    ci_pct = [float(np.percentile(coef_boot + coef, 2.5)),
              float(np.percentile(coef_boot + coef, 97.5))]
    return {"coef": coef, "se_state": se_state, "t_obs": float(t_obs),
            "p_value": p_val,
            "ci_studentized": [float(ci_stud[0]), float(ci_stud[1])],
            "ci_percentile": ci_pct,
            "t_boot_q": [float(q[0]), float(np.median(t_boot)), float(q[1])],
            "B": B, "n_obs": int(n), "n_state_clusters": int(G)}


def stacked_io_difference(d_stack, B=4999, seed=42):
    """STACKED fully-interacted IO-difference specification (deepen directive
    item b).

    d_stack = union of two disjoint subsamples with column 'io_grp_bin' in
    {'low','high'}. The design:
        controls W = intercept + 6 lower-order focal + low_IO main effect
                     + (low_IO x each of the 6 lower-order focal)
                     + state FE + month FE
        focal regressors = [three_way, (low_IO x three_way)]
    The coefficient on (low_IO x three_way) IS the (low - high) difference in the
    three-way slope, with the lower-order focal slopes free to differ by group
    and common within-state-month identification. Inference on that coefficient:
    state-clustered CR1, two-way state x month CGM, and a restricted
    wild-cluster bootstrap at the state level (H0: difference = 0), all on the
    STACKED sample. Bootstrap chunked over B.
    """
    rng = np.random.RandomState(seed)
    d = d_stack
    n = len(d)
    low_ind = (d['io_grp_bin'].values == 'low').astype(float)
    focal = d[FOCAL].values.astype(float)
    lo6 = focal[:, :6]                       # 6 lower-order focal
    three = focal[:, 6]
    x_int = low_ind * three                  # the difference term
    y = d['ret'].values.astype(float)
    sc = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    mc = pd.Categorical(d['ym']).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(lo6)
    lowm = sp.csr_matrix(low_ind.reshape(-1, 1))
    W6s_int = sp.csr_matrix(lo6 * low_ind[:, None])     # low_IO x 6 lower-order
    Sd = sp.csr_matrix((np.ones(n), (rows, sc)), shape=(n, nS))[:, 1:]
    Md = sp.csr_matrix((np.ones(n), (rows, mc)), shape=(n, nM))[:, 1:]
    W = sp.hstack([ones, W6s, lowm, W6s_int, Sd, Md]).tocsc()
    WtW = (W.T @ W).toarray()
    WtW_inv = np.linalg.inv(WtW)

    def partial(v):
        return v - W @ (WtW_inv @ (W.T @ v))

    X = np.column_stack([three, x_int])
    Xt = np.column_stack([partial(X[:, 0]), partial(X[:, 1])])
    yt = partial(y)
    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xt.T @ yt)          # [three_way_high, difference]
    e = yt - Xt @ beta
    scores = Xt * e[:, None]              # n x 2

    Cs, states = _cluster_matrix(sc)
    Cm, _ = _cluster_matrix(mc)
    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    Ci, _ = _cluster_matrix(inter)

    def meat(C):
        S = np.asarray(C.T @ scores)      # G x 2
        return S.T @ S

    meat_state = meat(Cs)
    meat_cgm = meat(Cs) + meat(Cm) - meat(Ci)
    V_state = XtX_inv @ meat_state @ XtX_inv
    V_cgm = XtX_inv @ meat_cgm @ XtX_inv
    diff_coef = float(beta[1])
    se_state = float(np.sqrt(V_state[1, 1]))
    cgm_psd = bool(V_cgm[1, 1] > 0)
    se_cgm = float(np.sqrt(V_cgm[1, 1])) if cgm_psd else np.nan
    t_obs = diff_coef / se_state if se_state > 0 else np.nan

    # restricted wild-cluster bootstrap on the interaction (H0: difference = 0)
    Xt0 = Xt[:, [0]]
    b0 = float((Xt0.T @ yt) / (Xt0.T @ Xt0))
    fit0 = Xt0.flatten() * b0
    e_r = yt - fit0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    a10, a11 = XtX_inv[1, 0], XtX_inv[1, 1]
    Xt0c, Xt1c = Xt[:, 0], Xt[:, 1]
    diff_t_boot = np.empty(B)
    done = 0
    while done < B:
        bb = min(CHUNK, B - done)
        w_state = WEBB[rng.randint(0, 6, size=(G, bb))]      # G x bb
        w = w_state[state_idx, :]                             # n x bb
        Y = fit0[:, None] + e_r[:, None] * w                  # n x bb
        Beta_b = XtX_inv @ (Xt.T @ Y)                         # 2 x bb
        E_b = Y - Xt @ Beta_b                                 # n x bb
        S0 = Cs.T @ (Xt0c[:, None] * E_b)                     # G x bb
        S1 = Cs.T @ (Xt1c[:, None] * E_b)                     # G x bb
        m00 = (S0 ** 2).sum(axis=0)
        m01 = (S0 * S1).sum(axis=0)
        m11 = (S1 ** 2).sum(axis=0)
        V11_b = a10 * a10 * m00 + 2 * a10 * a11 * m01 + a11 * a11 * m11
        se_b = np.sqrt(np.where(V11_b > 0, V11_b, np.nan))
        tb = np.where(se_b > 0, Beta_b[1, :] / se_b, 0.0)
        diff_t_boot[done:done + bb] = np.nan_to_num(tb, nan=0.0)
        done += bb
    p_wcb = float(np.mean(np.abs(diff_t_boot) >= np.abs(t_obs)))
    q = np.percentile(diff_t_boot, [2.5, 97.5])
    ci_stud = [diff_coef - q[1] * se_state, diff_coef - q[0] * se_state]

    return {
        "three_way_high_IO": float(beta[0]),
        "difference_coef": diff_coef,
        "se_state_clustered": se_state,
        "t_state_clustered": float(t_obs),
        "se_cgm_two_way": se_cgm,
        "t_cgm_two_way": float(diff_coef / se_cgm) if cgm_psd else None,
        "cgm_psd": cgm_psd,
        "wcb_p_value": p_wcb,
        "wcb_ci_studentized": [float(ci_stud[0]), float(ci_stud[1])],
        "wcb_t_boot_q": [float(q[0]), float(q[1])],
        "n_obs": int(n), "n_state_clusters": int(G), "B": B,
        "spec": "fully-interacted: low_IO x {6 lower-order focal, three_way}, "
                "common state+month FE",
    }


def stacked_state_group_difference(d, group_col, low_label, B=4999, seed=42):
    """STACKED difference test where the GROUP is a function of STATE (e.g.
    low-literacy vs high-literacy states). Used by item (d).

    Because the group is state-defined, a group main effect and group x
    lower-order interactions would be COLLINEAR with the state fixed effects
    (and the stacked_io_difference design would be near-singular). The correct
    spec for a state-defined group is:
        ret ~ 6 lower-order focal + state FE + month FE
              + three_way + (group_low x three_way)
    The group main effect and group x lower-order terms are intentionally
    omitted — they are absorbed by the state FE (group) and are not separately
    identified relative to state-specific lower-order slopes; here we keep the
    lower-order focal slopes common and let only the three-way differ by group,
    which is the difference-in-three-way-slopes the mechanism test needs.
    Inference: state-clustered CR1, two-way CGM, restricted wild-cluster
    bootstrap at the state level (H0: group difference = 0).

    d[group_col] takes two values; low_label is the value whose three-way is
    contrasted against the other (difference = low_label group minus other).
    """
    rng = np.random.RandomState(seed)
    n = len(d)
    glow = (d[group_col].values == low_label).astype(float)
    focal = d[FOCAL].values.astype(float)
    lo6 = focal[:, :6]
    three = focal[:, 6]
    x_int = glow * three
    y = d["ret"].values.astype(float)
    sc = pd.Categorical(d["hq_state"]).codes.astype(np.int64)
    mc = pd.Categorical(d["ym"]).codes.astype(np.int64)
    nS, nM = sc.max() + 1, mc.max() + 1
    rows = np.arange(n)
    ones = sp.csr_matrix(np.ones((n, 1)))
    W6s = sp.csr_matrix(lo6)
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
    scores = Xt * e[:, None]
    Cs, states = _cluster_matrix(sc)
    Cm, _ = _cluster_matrix(mc)
    inter = sc.astype(np.int64) * nM + mc.astype(np.int64)
    Ci, _ = _cluster_matrix(inter)

    def meat(C):
        S = np.asarray(C.T @ scores)
        return S.T @ S

    V_state = XtX_inv @ meat(Cs) @ XtX_inv
    meat_cgm = meat(Cs) + meat(Cm) - meat(Ci)
    V_cgm = XtX_inv @ meat_cgm @ XtX_inv
    diff_coef = float(beta[1])
    se_state = float(np.sqrt(V_state[1, 1]))
    cgm_psd = bool(V_cgm[1, 1] > 0)
    se_cgm = float(np.sqrt(V_cgm[1, 1])) if cgm_psd else np.nan
    t_obs = diff_coef / se_state if se_state > 0 else np.nan

    Xt0 = Xt[:, [0]]
    b0 = float((Xt0.T @ yt) / (Xt0.T @ Xt0))
    fit0 = Xt0.flatten() * b0
    e_r = yt - fit0
    G = len(states)
    state_idx = np.searchsorted(states, sc)
    a10, a11 = XtX_inv[1, 0], XtX_inv[1, 1]
    Xt0c, Xt1c = Xt[:, 0], Xt[:, 1]
    diff_t_boot = np.empty(B)
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
        diff_t_boot[done:done + bb] = np.nan_to_num(tb, nan=0.0)
        done += bb
    p_wcb = float(np.mean(np.abs(diff_t_boot) >= np.abs(t_obs)))
    q = np.percentile(diff_t_boot, [2.5, 97.5])
    ci_stud = [diff_coef - q[1] * se_state, diff_coef - q[0] * se_state]
    return {
        "three_way_reference_group": float(beta[0]),
        "difference_coef": diff_coef,
        "se_state_clustered": se_state,
        "t_state_clustered": float(t_obs),
        "se_cgm_two_way": se_cgm,
        "t_cgm_two_way": float(diff_coef / se_cgm) if cgm_psd else None,
        "cgm_psd": cgm_psd,
        "wcb_p_value": p_wcb,
        "wcb_ci_studentized": [float(ci_stud[0]), float(ci_stud[1])],
        "wcb_t_boot_q": [float(q[0]), float(q[1])],
        "n_obs": int(n), "n_state_clusters": int(G), "B": B,
        "spec": "state-defined group: three_way + group_low x three_way, common "
                "6 lower-order focal + state FE + month FE (group main effect and "
                "group x lower-order omitted — collinear with state FE)",
    }
