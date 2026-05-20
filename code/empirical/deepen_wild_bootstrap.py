# =====================================================================
# deepen_wild_bootstrap.py
# State-level restricted (impose-the-null) wild-cluster bootstrap with Webb weights (B=9999) on the TWFE-CGM three-way coefficient, plus an unrestricted WCB studentized CI.
#
# Inputs:    standardized firm-month panel, output/seed/corrected_results.json
# Outputs:   output/stage3a/deepen_wild_bootstrap.{json,md}, tables/fig_wild_bootstrap_dist.{pdf,png}, tables/tab_wild_bootstrap.tex
# Paper:     tab_wild_bootstrap.tex + fig_wild_bootstrap_dist (IA full inference battery)
# Run order: see code/00_master.py
# =====================================================================

"""Deepen directive Item 1 — State-level wild-cluster bootstrap on the TWFE-CGM
three-way coefficient.

The current paper foregrounds the TWFE-CGM two-way-clustered t = -5.32 and treats
the wild-cluster bootstrap (p = 0.301) as a sensitivity note. The prior IA was
ambiguous about whether the bootstrap was applied to the FM or the TWFE estimate.
RESOLVED HERE: the corrected_results.json wild_cluster_bootstrap entry (p = 0.301)
sits under the FM imp3 block with t_hat = -0.85 = the Fama-MacBeth NW(12) t-stat,
so the prior p = 0.301 was computed on the FM estimate, NOT the TWFE estimate.

This script runs the restricted (impose-the-null) wild-cluster bootstrap on the
TWFE state+month panel estimator, clustering at the STATE level (51 clusters),
with Webb 6-point weights and B = 9999. It also runs an unrestricted WCB to build
the studentized bootstrap CI.

Procedure (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017; MacKinnon 2023):
  1. Estimate the FULL TWFE model -> gamma_hat. Studentize against the *state-
     clustered* CR1 SE (the WCB clusters at the state level).
  2. Restricted model = full model minus the three-way column -> e_tilde, yhat_r.
  3. For b = 1..B: draw Webb 6-point weight w_g per state cluster g;
     y*_ig = yhat_r,i + w_{g(i)} * e_tilde_i ; re-estimate FULL model;
     t*_b = gamma*_b / SE_state-clustered(gamma*_b).
  4. Restricted-WCB p-value = share of |t*_b| >= |t_hat| (symmetric two-tailed).
  5. Studentized 95% CI from the unrestricted WCB t-distribution.

Efficiency: Frisch-Waugh-Lovell. With W = all regressors except the three-way,
residualize x3 and y against W ONCE. Then gamma = (x3t'yt)/(x3t'x3t), and because
the restricted fit lies in span(W), y*_tilde = M_W(w*e_tilde). Every per-cluster
quantity needed for gamma* and its state-clustered SE reduces to G-dimensional
algebra with three precomputed objects (p_g, Sxx_g, Q), so a rep is O(G^2), not
O(n*k). Validated against the direct full-design estimator (exact match).

Output: output/stage3a/deepen_wild_bootstrap.{json,md}
      + output/stage3a/tables/fig_wild_bootstrap_dist.{pdf,png}
      + output/stage3a/tables/tab_wild_bootstrap.tex
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical"
PANEL = os.path.join(ROOT, "output", "seed", "data", "processed", "panel_corrected_standardized.parquet")
OUT_JSON = os.path.join(ROOT, "output", "stage3a", "deepen_wild_bootstrap.json")
OUT_MD = os.path.join(ROOT, "output", "stage3a", "deepen_wild_bootstrap.md")
FIG = os.path.join(ROOT, "output", "stage3a", "tables", "fig_wild_bootstrap_dist.pdf")
TABLE = os.path.join(ROOT, "output", "stage3a", "tables", "tab_wild_bootstrap.tex")

FOCAL = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
         'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']
B = 9999
WEBB6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                  np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def two_way_cgm_se(Xfull, y, beta, g_state, g_month, XX_inv, idx3):
    """CGM two-way state x month clustered SE for coefficient idx3 (the seed
    headline SE). Computed once on the observed data only."""
    e = y - Xfull @ beta
    k = Xfull.shape[1]

    def meat(g):
        m = np.zeros((k, k))
        for gid in np.unique(g):
            ix = np.where(g == gid)[0]
            s = Xfull[ix].T @ e[ix]
            m += np.outer(s, s)
        return m

    M_state = meat(g_state)
    M_month = meat(g_month)
    n_months = int(g_month.max()) + 1
    inter = g_state.astype(np.int64) * n_months + g_month.astype(np.int64)
    M_inter = meat(inter)
    V = XX_inv @ (M_state + M_month - M_inter) @ XX_inv
    return float(np.sqrt(max(V[idx3, idx3], 0.0)))


def main():
    print("=== loading panel ===")
    df = pd.read_parquet(PANEL)
    needed = FOCAL + ['ret', 'hq_state', 'date']
    d = df.dropna(subset=needed).copy()
    d['ym'] = d['date'].dt.to_period('M').astype(str)
    d = d.sort_values('hq_state').reset_index(drop=True)  # contiguous state blocks
    n = len(d)
    print(f"n_obs = {n:,}, n_states = {d['hq_state'].nunique()}, "
          f"n_months = {d['ym'].nunique()}")

    focal = d[FOCAL].values.astype(float)
    x3 = focal[:, 6].copy()                       # the three-way
    sd = pd.get_dummies(d['hq_state'].astype(str), prefix='S',
                        drop_first=True, dtype=float)
    mdum = pd.get_dummies(d['ym'], prefix='M', drop_first=True, dtype=float)
    # W = everything except the three-way; Xfull = W plus the three-way at idx 7
    W = np.hstack([np.ones((n, 1)), focal[:, :6], sd.values, mdum.values])
    Xfull = np.hstack([np.ones((n, 1)), focal, sd.values, mdum.values])
    y = d['ret'].values.astype(float)
    kW = W.shape[1]
    k = Xfull.shape[1]
    idx3_full = 7  # 1 intercept + 6 lower-order focal, then the three-way

    g_state = pd.Categorical(d['hq_state']).codes.astype(np.int64)
    g_month = pd.Categorical(d['ym']).codes.astype(np.int64)
    G = len(np.unique(g_state))
    starts = np.searchsorted(g_state, np.arange(G))
    ends = np.append(starts[1:], n)
    csum = lambda prod: np.add.reduceat(prod, starts)  # contiguous-block sums

    # --- FWL residualization (once) ---
    WtW_inv = np.linalg.pinv(W.T @ W)
    resid = lambda v: v - W @ (WtW_inv @ (W.T @ v))
    x3t = resid(x3)
    yt = resid(y)
    Sxx = x3t @ x3t
    gamma_hat = float((x3t @ yt) / Sxx)
    e_full = yt - gamma_hat * x3t       # full-model residual
    e_tilde = yt                        # restricted residual = M_W y = yt

    # CR1 small-sample factor for one-way state clustering
    c_cr1 = (G / (G - 1.0)) * ((n - 1.0) / (n - kW - 1.0))
    se_state = float(np.sqrt(c_cr1 * (csum(x3t * e_full) ** 2).sum()) / Sxx)
    t_state = gamma_hat / se_state

    # the seed headline two-way CGM SE (computed once, direct)
    XX_inv_full = np.linalg.pinv(Xfull.T @ Xfull)
    beta_full = XX_inv_full @ (Xfull.T @ y)
    se_cgm = two_way_cgm_se(Xfull, y, beta_full, g_state, g_month,
                            XX_inv_full, idx3_full)
    t_cgm = gamma_hat / se_cgm

    print(f"\nTWFE three-way gamma_hat = {gamma_hat:.6f}")
    print(f"  two-way state x month CGM SE = {se_cgm:.6f}, t = {t_cgm:.4f}  "
          f"(seed headline)")
    print(f"  one-way state-clustered (CR1) SE = {se_state:.6f}, t = {t_state:.4f}"
          f"  (WCB studentizes against this)")

    # --- precompute the G-dimensional bootstrap objects ---
    p_g = csum(x3t * e_tilde)        # x3t_g' e_tilde_g
    Sxx_g = csum(x3t * x3t)          # x3t_g' x3t_g
    C = np.zeros((kW, G))
    D = np.zeros((G, kW))
    for g in range(G):
        lo, hi = starts[g], ends[g]
        C[:, g] = W[lo:hi].T @ e_tilde[lo:hi]
        D[g, :] = x3t[lo:hi] @ W[lo:hi]
    Q = D @ (WtW_inv @ C)            # G x G : x3t_g' proj_g = Q[g,:] @ w_g

    def wcb(impose_null, n_reps):
        """Restricted (impose_null=True) or unrestricted WCB. Returns t* and g*."""
        if impose_null:
            base_resid = e_tilde
            center = 0.0
            base_p_g = p_g
        else:
            base_resid = e_full
            center = gamma_hat
            # for the unrestricted bootstrap, y* = yhat_full + w*e_full;
            # y*_tilde = M_W(yhat_full + w*e_full). M_W yhat_full = gamma_hat*x3t
            # (the part of the full fit orthogonal to W). x3t'(M_W yhat_full) =
            # gamma_hat*Sxx. So x3t'y*_tilde = gamma_hat*Sxx + sum_g w_g*p_g^full.
            base_p_g = csum(x3t * e_full)
        tstars = np.empty(n_reps)
        gstars = np.empty(n_reps)
        # Cg and Dg for the chosen base_resid (restricted: e_tilde already done)
        if impose_null:
            Qb = Q
            pg_b = p_g
            const_num = 0.0
        else:
            Cb = np.zeros((kW, G))
            Db_dot = np.zeros((G, kW))
            for g in range(G):
                lo, hi = starts[g], ends[g]
                Cb[:, g] = W[lo:hi].T @ e_full[lo:hi]
                Db_dot[g, :] = x3t[lo:hi] @ W[lo:hi]
            Qb = Db_dot @ (WtW_inv @ Cb)
            pg_b = base_p_g
            const_num = gamma_hat * Sxx
        for b in range(n_reps):
            w_g = WEBB6[np.random.randint(0, 6, size=G)]
            gstar = (const_num + pg_b @ w_g) / Sxx
            # per-cluster x3t_g' estar_g, estar = ystar_t - gstar*x3t
            # ystar_t = M_W(w*base_resid)  [+ gamma_hat*x3t in unrestricted case,
            #   but that part contributes gamma_hat*Sxx_g per cluster]
            if impose_null:
                sc = w_g * pg_b - Qb @ w_g - gstar * Sxx_g
            else:
                sc = (gamma_hat * Sxx_g + w_g * pg_b - Qb @ w_g
                      - gstar * Sxx_g)
            se_s = np.sqrt(c_cr1 * (sc @ sc)) / Sxx
            tstars[b] = (gstar - center) / se_s if se_s > 0 else 0.0
            gstars[b] = gstar
        return tstars, gstars

    print(f"\n=== restricted WCB (B={B}, Webb-6, state clusters) — p-value ===")
    t_restr, g_restr = wcb(True, B)
    p_wcb = float(np.mean(np.abs(t_restr) >= np.abs(t_state)))
    p_wcb_vs_cgm = float(np.mean(np.abs(t_restr) >= np.abs(t_cgm)))

    print(f"=== unrestricted WCB (B={B}) — studentized CI ===")
    t_unr, g_unr = wcb(False, B)
    q_lo, q_hi = np.percentile(t_unr, [2.5, 97.5])
    ci_lo = gamma_hat - q_hi * se_state
    ci_hi = gamma_hat - q_lo * se_state
    g_ci_lo, g_ci_hi = np.percentile(g_unr, [2.5, 97.5])

    survives_05 = p_wcb < 0.05
    survives_10 = p_wcb < 0.10

    print(f"\n{'='*62}")
    print(f"RESTRICTED WCB p (vs state-clustered t={t_state:.3f}): {p_wcb:.4f}")
    print(f"RESTRICTED WCB p (vs CGM t={t_cgm:.3f}):           {p_wcb_vs_cgm:.4f}")
    print(f"studentized 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"percentile 95% CI:  [{g_ci_lo:.6f}, {g_ci_hi:.6f}]")
    print(f"survives 5%: {survives_05} | survives 10%: {survives_10}")
    print(f"{'='*62}")

    # --- figure ---
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(t_restr, bins=80, color="#888888", alpha=0.7, density=True,
            label=f"restricted WCB null dist. (B={B})")
    ax.axvline(t_state, color="crimson", lw=2,
               label=f"observed TWFE $t$ (state-clustered) = {t_state:.2f}")
    ax.axvline(-t_state, color="crimson", lw=2, ls="--")
    ax.axvline(t_cgm, color="navy", lw=1.5, ls=":",
               label=f"observed TWFE $t$ (CGM two-way) = {t_cgm:.2f}")
    ax.set_xlabel("bootstrap $t$-statistic on the three-way coefficient")
    ax.set_ylabel("density")
    ax.set_title("State-level wild-cluster bootstrap (Webb 6-point) — TWFE three-way")
    ax.legend(fontsize=8, loc="upper left")
    txt = (f"restricted WCB $p$ = {p_wcb:.3f}\n"
           f"95% studentized CI = [{ci_lo:.4f}, {ci_hi:.4f}]\n"
           f"{G} state clusters, {n:,} firm-months")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top",
            ha="right", bbox=dict(boxstyle="round", fc="white", ec="grey",
                                  alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG)
    fig.savefig(FIG.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    res = {
        "design": {
            "n_obs": int(n), "n_params_full": int(k),
            "n_state_clusters": int(G), "n_months": int(d['ym'].nunique()),
            "estimator": "TWFE state+month FE; coef on mom_x_iv_x_literacy_corr",
        },
        "point_estimate": {
            "gamma_hat": gamma_hat,
            "se_cgm_two_way": se_cgm, "t_cgm_two_way": float(t_cgm),
            "se_state_clustered_CR1": se_state,
            "t_state_clustered_CR1": float(t_state),
        },
        "wild_cluster_bootstrap": {
            "B": B, "weights": "Webb 6-point",
            "cluster_level": f"state ({G} clusters)", "restricted": True,
            "p_value_vs_state_clustered_t": p_wcb,
            "p_value_vs_cgm_t": p_wcb_vs_cgm,
            "studentized_ci_95": [ci_lo, ci_hi],
            "percentile_ci_95_on_coef": [g_ci_lo, g_ci_hi],
            "bootstrap_t_quantiles": {
                "p1": float(np.percentile(t_restr, 1)),
                "p2.5": float(np.percentile(t_restr, 2.5)),
                "p5": float(np.percentile(t_restr, 5)),
                "p50": float(np.percentile(t_restr, 50)),
                "p95": float(np.percentile(t_restr, 95)),
                "p97.5": float(np.percentile(t_restr, 97.5)),
                "p99": float(np.percentile(t_restr, 99)),
            },
            "bootstrap_t_min": float(t_restr.min()),
            "bootstrap_t_max": float(t_restr.max()),
            "frac_t_below_neg2": float(np.mean(t_restr < -2.0)),
            "survives_5pct": bool(survives_05),
            "survives_10pct": bool(survives_10),
        },
        "ambiguity_resolution": (
            "The prior IA wild_cluster_bootstrap entry (p = 0.301) sits under the "
            "FM imp3 block in corrected_results.json with t_hat = -0.85, the "
            "Fama-MacBeth NW(12) t-stat. The prior p = 0.301 was therefore "
            "computed on the FM estimate, NOT the TWFE estimate. This script runs "
            "the restricted wild-cluster bootstrap explicitly on the TWFE "
            "state+month panel estimator (gamma_hat = -0.0117), studentized "
            "against the one-way state-clustered CR1 SE."
        ),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)

    with open(TABLE, 'w', encoding='utf-8') as f:
        f.write("\\begin{tabular}{lc}\n\\hline\\hline\n")
        f.write(f"TWFE three-way coefficient $\\hat\\gamma$ & ${gamma_hat:.4f}$ \\\\\n")
        f.write(f"Two-way (state$\\times$month) CGM SE & ${se_cgm:.4f}$ \\\\\n")
        f.write(f"\\quad CGM $t$-statistic & ${t_cgm:.2f}$ \\\\\n")
        f.write(f"One-way state-clustered (CR1) SE & ${se_state:.4f}$ \\\\\n")
        f.write(f"\\quad state-clustered $t$-statistic & ${t_state:.2f}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"Wild-cluster bootstrap $p$-value & ${p_wcb:.3f}$ \\\\\n")
        f.write("\\quad (restricted, Webb 6-point, $B=9999$, state clusters) & \\\\\n")
        f.write(f"95\\% studentized bootstrap CI & $[{ci_lo:.4f},\\ {ci_hi:.4f}]$ \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Deepen Item 1 — State-level Wild-Cluster Bootstrap on the "
                "TWFE Three-Way Coefficient\n\n")
        f.write("## Ambiguity resolved\n\n" + res["ambiguity_resolution"] + "\n\n")
        f.write("## Point estimate\n\n")
        f.write(f"- TWFE three-way $\\hat\\gamma$ = `{gamma_hat:.6f}`\n")
        f.write(f"- Two-way state$\\times$month CGM SE = `{se_cgm:.6f}`, "
                f"$t$ = `{t_cgm:.4f}` (the seed headline)\n")
        f.write(f"- One-way state-clustered (CR1) SE = `{se_state:.6f}`, "
                f"$t$ = `{t_state:.4f}` (the WCB studentizes against this)\n\n")
        f.write(f"## Wild-cluster bootstrap (restricted, Webb 6-point, "
                f"B = {B}, state-level clustering)\n\n")
        f.write(f"- **Bootstrap p-value (vs state-clustered $t$ = {t_state:.2f}): "
                f"`{p_wcb:.4f}`**\n")
        f.write(f"- Bootstrap p-value (vs CGM $t$ = {t_cgm:.2f}): "
                f"`{p_wcb_vs_cgm:.4f}`\n")
        f.write(f"- 95% studentized bootstrap CI: `[{ci_lo:.6f}, {ci_hi:.6f}]`\n")
        f.write(f"- 95% percentile CI on the bootstrap coefficient: "
                f"`[{g_ci_lo:.6f}, {g_ci_hi:.6f}]`\n")
        f.write(f"- Bootstrap $t$ distribution: min `{t_restr.min():.2f}`, "
                f"p2.5 `{np.percentile(t_restr,2.5):.2f}`, "
                f"p50 `{np.percentile(t_restr,50):.2f}`, "
                f"p97.5 `{np.percentile(t_restr,97.5):.2f}`, "
                f"max `{t_restr.max():.2f}`\n")
        f.write(f"- Fraction of bootstrap $t$ below $-2$: "
                f"`{np.mean(t_restr < -2.0):.4f}`\n\n")
        f.write("## Verdict\n\n")
        if survives_05:
            f.write(f"**The headline SURVIVES the credible estimator at 5%.** "
                    f"The restricted state-level wild-cluster bootstrap p-value "
                    f"is {p_wcb:.3f} < 0.05.\n")
        elif survives_10:
            f.write(f"**The headline is MARGINAL under the credible estimator.** "
                    f"The restricted state-level wild-cluster bootstrap p-value "
                    f"is {p_wcb:.3f}: significant at 10% but not at 5%. The "
                    f"TWFE-CGM t = -5.32 substantially over-states the evidence; "
                    f"the credible state-level inference procedure delivers only "
                    f"marginal significance.\n")
        else:
            f.write(f"**The headline does NOT survive the credible estimator.** "
                    f"The restricted state-level wild-cluster bootstrap p-value "
                    f"is {p_wcb:.3f} >= 0.10. The TWFE-CGM t = -5.32 is an "
                    f"over-rejection artifact of two-way clustering with few "
                    f"state clusters; the credible inference procedure cannot "
                    f"reject the null. Per the referees' own 'what would be "
                    f"publishable' sections, the honest paper is the "
                    f"disciplined-null frame.\n")
    print(f"json -> {OUT_JSON}\nmd   -> {OUT_MD}\ntex  -> {TABLE}\nfig  -> {FIG}")
    return res


if __name__ == '__main__':
    main()
