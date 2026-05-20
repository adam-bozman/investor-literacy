# =====================================================================
# bayes_shrinkage_literacy.py
# Empirical-Bayes (James-Stein) shrinkage of noisy state-wave NFCS literacy toward the wave grand mean, then re-runs the headline TWFE on the shrunk moderator.
#
# Inputs:    5 NFCS waves (raw CSVs), standardized firm-month panel
# Outputs:   output/stage3a/bayes_shrinkage_literacy.{json,md}
# Paper:     SUPERSEDED by v10_eiv_classical.py (empirical-Bayes shrinkage line replaced by classical EIV)
# Run order: see code/00_master.py
# =====================================================================

"""REVISE pass — Bayesian-shrinkage literacy robustness.

Per output/stage4/triage_v1.md item (4) and the structured scorer's substantive
feedback item (4): state-level NFCS literacy estimates are noisy. NFCS Big Five
is administered to ~25,000 households nationally per wave; within-state
subsamples range from ~250 to ~3,000 respondents. The self-attack computed a
signal-to-noise ratio of roughly 1.5x in the moderator. Classical
errors-in-variables theory says the OLS coefficient on a noisily-measured
regressor is attenuated toward zero; the honest robustness is to shrink each
state's literacy estimate toward the wave grand mean using its own sampling
variance (empirical-Bayes / James-Stein), then re-run the headline.

Procedure:
  1. Read the 5 NFCS waves; recode the Big Five with the codebook-correct codes
     (M6=1, M7=3, M8=2, M9=2, M10=1 — same as 00_build_corrected_literacy.py).
  2. For each state-wave: weighted share with >= 3/5 correct (p_hat); Kish
     effective sample size n_eff = (sum w)^2 / sum(w^2); sampling variance
     V_s = p_hat(1-p_hat)/n_eff.
  3. Empirical-Bayes shrinkage within each wave:
       V_total = cross-state variance of p_hat
       V_sampling_bar = mean of V_s
       V_true = max(V_total - V_sampling_bar, eps)
       B_s = V_s / (V_s + V_true)   # shrinkage weight toward grand mean
       p_shrunk_s = B_s * grand_mean + (1 - B_s) * p_hat_s
  4. Map raw and shrunk literacy onto the firm-month panel by (state, wave).
     Wave assignment: 2009->2009-2011, 2012->2012-2014, 2015->2015-2017,
     2018->2018-2020, 2021->2021-2023.
  5. Verify the raw mapping reproduces the panel's z-scored literacy (within-
     month rank correlation ~1.0).
  6. Z-score shrunk literacy within month; rebuild mom*iv*lit_shrunk_z; re-run
     the headline TWFE state+month two-way state x month CGM.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(r"C:/Users/adam.bozman/OneDrive - Washington State University (email.wsu.edu)/Research/investor-attention-empirical")
NFCS_DIR = ROOT / "output" / "seed" / "data" / "raw" / "nfcs"
PANEL = ROOT / "output" / "seed" / "data" / "processed" / "panel_corrected_standardized.parquet"
OUT_JSON = ROOT / "output" / "stage3a" / "bayes_shrinkage_literacy.json"
OUT_MD = ROOT / "output" / "stage3a" / "bayes_shrinkage_literacy.md"

CORRECT_CODES = {"m6": 1, "m7": 3, "m8": 2, "m9": 2, "m10": 1}
STATEQ_MAP = {
    1: "AL", 2: "AK", 3: "AZ", 4: "AR", 5: "CA", 6: "CO", 7: "CT", 8: "DE",
    9: "DC", 10: "FL", 11: "GA", 12: "HI", 13: "ID", 14: "IL", 15: "IN",
    16: "IA", 17: "KS", 18: "KY", 19: "LA", 20: "ME", 21: "MD", 22: "MA",
    23: "MI", 24: "MN", 25: "MS", 26: "MO", 27: "MT", 28: "NE", 29: "NV",
    30: "NH", 31: "NJ", 32: "NM", 33: "NY", 34: "NC", 35: "ND", 36: "OH",
    37: "OK", 38: "OR", 39: "PA", 40: "RI", 41: "SC", 42: "SD", 43: "TN",
    44: "TX", 45: "UT", 46: "VT", 47: "VA", 48: "WA", 49: "WV", 50: "WI",
    51: "WY",
}
WAVES = {2009: "nfcs_2009.csv", 2012: "nfcs_2012.csv", 2015: "nfcs_2015.csv",
         2018: "nfcs_2018.csv", 2021: "nfcs_2021.csv"}


def two_way_clustered_se(y, X, cluster_state, cluster_month):
    n, k = X.shape
    XX_inv = np.linalg.pinv(X.T @ X)
    beta = XX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        m = np.zeros((k, k))
        for gid in np.unique(g):
            idx = np.where(g == gid)[0]
            Xge = X[idx].T @ e[idx]
            m += np.outer(Xge, Xge)
        return m

    M_state = meat(cluster_state)
    M_month = meat(cluster_month)
    n_months = int(cluster_month.max()) + 1
    inter = cluster_state.astype(np.int64) * n_months + cluster_month.astype(np.int64)
    M_inter = meat(inter)
    V = XX_inv @ (M_state + M_month - M_inter) @ XX_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0))


def build_wave_raw(csv_path, year, weight_col="wgt_n2"):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.lower()
    for q, code in CORRECT_CODES.items():
        df[f"{q}_c"] = (pd.to_numeric(df[q], errors="coerce") == code).astype(float)
    df["n_correct"] = df[[f"{q}_c" for q in CORRECT_CODES]].sum(axis=1)
    if weight_col in df.columns:
        df["w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    else:
        df["w"] = 1.0
    rows = []
    for stq, grp in df.groupby("stateq"):
        try:
            ab = STATEQ_MAP.get(int(stq))
        except (TypeError, ValueError):
            continue
        if ab is None or grp["w"].sum() == 0:
            continue
        w = grp["w"].values
        correct = (grp["n_correct"] >= 3).astype(float).values
        w_sum = w.sum()
        p_hat = (w * correct).sum() / w_sum
        # Kish effective sample size
        n_eff = (w_sum ** 2) / (w ** 2).sum()
        # sampling variance of the weighted proportion
        v_s = p_hat * (1 - p_hat) / n_eff
        rows.append({"state": f"US-{ab}", "year": year, "p_hat": p_hat,
                     "n_resp": len(grp), "n_eff": n_eff, "v_s": v_s})
    return pd.DataFrame(rows)


def empirical_bayes_shrink(wave_df):
    """Within-wave empirical-Bayes shrinkage of state literacy estimates."""
    p = wave_df["p_hat"].values
    v_s = wave_df["v_s"].values
    grand_mean = p.mean()
    v_total = p.var(ddof=1)
    v_sampling_bar = v_s.mean()
    v_true = max(v_total - v_sampling_bar, 1e-8)
    B = v_s / (v_s + v_true)               # shrinkage weight toward grand mean
    p_shrunk = B * grand_mean + (1 - B) * p
    out = wave_df.copy()
    out["grand_mean"] = grand_mean
    out["v_total"] = v_total
    out["v_true"] = v_true
    out["shrink_weight"] = B
    out["p_shrunk"] = p_shrunk
    return out


def wave_for_year(y):
    if y <= 2011:
        return 2009
    if y <= 2014:
        return 2012
    if y <= 2017:
        return 2015
    if y <= 2020:
        return 2018
    return 2021


def main():
    print("=== building raw + shrunk state-wave literacy ===")
    parts = []
    for yr, fname in WAVES.items():
        w = build_wave_raw(NFCS_DIR / fname, yr)
        w = empirical_bayes_shrink(w)
        print(f"  wave {yr}: {len(w)} states, grand_mean={w['grand_mean'].iloc[0]:.4f}, "
              f"v_true={w['v_true'].iloc[0]:.6f}, mean shrink weight={w['shrink_weight'].mean():.3f}, "
              f"median n_eff={w['n_eff'].median():.0f}")
        parts.append(w)
    lit = pd.concat(parts, ignore_index=True)

    # corr between raw and shrunk
    overall_corr = lit["p_hat"].corr(lit["p_shrunk"])
    print(f"overall raw-vs-shrunk literacy correlation: {overall_corr:.4f}")
    print(f"mean shrink weight across all state-waves: {lit['shrink_weight'].mean():.3f}")

    print("\n=== loading firm-month panel ===")
    df = pd.read_parquet(PANEL)
    df["ym_str"] = df["date"].dt.to_period("M").astype(str)
    df["wave_year"] = df["year"].apply(wave_for_year)

    # map raw + shrunk literacy by (state, wave)
    lit_map = lit.set_index(["state", "year"])[["p_hat", "p_shrunk"]]
    df = df.merge(
        lit_map.rename(columns={"p_hat": "lit_raw", "p_shrunk": "lit_shrunk"}),
        left_on=["hq_state", "wave_year"], right_index=True, how="left")
    print(f"panel after literacy merge: {df.shape}; "
          f"lit_raw missing: {df['lit_raw'].isna().sum()}, "
          f"lit_shrunk missing: {df['lit_shrunk'].isna().sum()}")

    needed = ['ret', 'mom_12_2', 'iv', 'literacy_score_corrected',
              'mom_x_iv', 'mom_x_literacy_corr', 'iv_x_literacy_corr',
              'mom_x_iv_x_literacy_corr', 'hq_state', 'lit_raw', 'lit_shrunk']
    d = df.dropna(subset=needed).copy()
    print(f"estimation sample: {d.shape}")

    # VERIFY: raw literacy mapping should reproduce the panel's z-scored literacy
    # (within-month rank correlation should be ~1.0)
    d["lit_raw_z"] = d.groupby("ym_str")["lit_raw"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    rank_corr = spearmanr(d["lit_raw_z"], d["literacy_score_corrected"]).statistic
    print(f"VERIFICATION: within-month rank corr (reconstructed raw lit_z vs "
          f"panel literacy_score_corrected) = {rank_corr:.4f}")
    if rank_corr < 0.95:
        print("  WARNING: reconstruction does not match panel literacy well. "
              "Wave-assignment or coding may differ from the seed builder.")

    # z-score shrunk literacy within month; rebuild three-way
    d["lit_shrunk_z"] = d.groupby("ym_str")["lit_shrunk"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    # rebuild interaction terms with shrunk literacy, z-scored within month
    d["mom_x_lit_shrunk"] = d["mom_12_2"] * d["lit_shrunk_z"]
    d["iv_x_lit_shrunk"] = d["iv"] * d["lit_shrunk_z"]
    d["mom_x_iv_x_lit_shrunk"] = d["mom_x_iv"] * d["lit_shrunk_z"]
    for c in ["mom_x_lit_shrunk", "iv_x_lit_shrunk", "mom_x_iv_x_lit_shrunk", "lit_shrunk_z"]:
        d[c] = d.groupby("ym_str")[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    states = pd.get_dummies(d["hq_state"].astype(str), prefix="S", drop_first=True, dtype=float)
    months = pd.get_dummies(d["ym_str"], prefix="M", drop_first=True, dtype=float)
    state_codes = pd.Categorical(d["hq_state"]).codes
    month_codes = pd.Categorical(d["ym_str"]).codes
    y = d["ret"].values.astype(float)

    def run(focal_cols, label):
        X = np.hstack([np.ones((len(d), 1)), d[focal_cols].values.astype(float),
                       states.values, months.values])
        beta, se = two_way_clustered_se(y, X, state_codes, month_codes)
        idx = 1 + focal_cols.index([c for c in focal_cols if "x_iv_x" in c][0])
        coef, se_ = float(beta[idx]), float(se[idx])
        print(f"  [{label}] coef={coef:.6f}, se={se_:.6f}, t={coef/se_:.4f}")
        return {"coef": coef, "se": se_, "t": coef / se_, "n_obs": int(len(d)), "label": label}

    print("\n=== running specifications ===")
    base_cols = ['mom_12_2', 'iv', 'literacy_score_corrected', 'mom_x_iv',
                 'mom_x_literacy_corr', 'iv_x_literacy_corr', 'mom_x_iv_x_literacy_corr']
    shrunk_cols = ['mom_12_2', 'iv', 'lit_shrunk_z', 'mom_x_iv',
                   'mom_x_lit_shrunk', 'iv_x_lit_shrunk', 'mom_x_iv_x_lit_shrunk']
    spec_base = run(base_cols, "baseline (raw NFCS literacy)")
    spec_shrunk = run(shrunk_cols, "shrunk (empirical-Bayes literacy)")

    delta = spec_shrunk["coef"] - spec_base["coef"]
    res = {
        "n_state_waves": int(len(lit)),
        "mean_shrink_weight": float(lit["shrink_weight"].mean()),
        "raw_vs_shrunk_corr": float(overall_corr),
        "median_n_eff": float(lit["n_eff"].median()),
        "min_n_eff": float(lit["n_eff"].min()),
        "verification_rank_corr": float(rank_corr),
        "spec_baseline": spec_base,
        "spec_shrunk": spec_shrunk,
        "delta_coef": float(delta),
        "delta_in_baseline_se": float(delta / spec_base["se"]),
        "sign_preserved": bool(np.sign(spec_shrunk["coef"]) == np.sign(spec_base["coef"])),
        "still_significant": bool(abs(spec_shrunk["t"]) >= 1.96),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Bayesian-Shrinkage Literacy Robustness\n\n")
        f.write("Per `output/stage4/triage_v1.md` item (4) and the structured "
                "scorer's substantive feedback. State-level NFCS literacy "
                "estimates are noisy (median Kish effective sample size "
                f"{res['median_n_eff']:.0f}, minimum {res['min_n_eff']:.0f}). "
                "Classical errors-in-variables theory implies the OLS "
                "coefficient on a noisily-measured regressor is attenuated "
                "toward zero. This robustness shrinks each state's literacy "
                "estimate toward the wave grand mean using its own sampling "
                "variance (empirical-Bayes / James-Stein), then re-runs the "
                "headline.\n\n")
        f.write("## Shrinkage diagnostics\n\n")
        f.write(f"- State-waves: {res['n_state_waves']}\n")
        f.write(f"- Mean shrinkage weight toward grand mean: "
                f"{res['mean_shrink_weight']:.3f} "
                f"(0 = no shrinkage, 1 = full collapse to grand mean)\n")
        f.write(f"- Raw-vs-shrunk literacy correlation: {res['raw_vs_shrunk_corr']:.4f}\n")
        f.write(f"- Median Kish effective sample size: {res['median_n_eff']:.0f}\n")
        f.write(f"- Verification: within-month rank corr of reconstructed raw "
                f"literacy_z vs. the panel's `literacy_score_corrected` = "
                f"{res['verification_rank_corr']:.4f} "
                f"({'reconstruction matches the seed panel' if res['verification_rank_corr'] > 0.95 else 'WARNING: reconstruction diverges from seed panel'})\n\n")
        f.write("## Headline coefficient: raw vs. shrunk literacy\n\n")
        f.write("| Specification | three-way coef | SE | t-stat | n_obs |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Baseline (raw NFCS literacy) | {spec_base['coef']:.6f} | "
                f"{spec_base['se']:.6f} | {spec_base['t']:.4f} | {spec_base['n_obs']:,} |\n")
        f.write(f"| **Empirical-Bayes shrunk literacy** | **{spec_shrunk['coef']:.6f}** | "
                f"**{spec_shrunk['se']:.6f}** | **{spec_shrunk['t']:.4f}** | "
                f"**{spec_shrunk['n_obs']:,}** |\n\n")
        f.write(f"**Change:** {delta:+.6f} ({res['delta_in_baseline_se']:+.3f} baseline SEs). "
                f"Sign {'preserved' if res['sign_preserved'] else 'FLIPPED'}; "
                f"{'still significant at 5%' if res['still_significant'] else 'NO LONGER significant at 5%'}.\n\n")
        if res['sign_preserved'] and res['still_significant']:
            if abs(spec_shrunk['coef']) >= abs(spec_base['coef']):
                f.write("**Verdict: ROBUST (and consistent with EIV attenuation).** "
                        "The shrunk-literacy coefficient is at least as large in "
                        "magnitude as the raw-literacy coefficient and remains "
                        "significant. This is the direction classical "
                        "errors-in-variables theory predicts: correcting "
                        "measurement error in the moderator un-attenuates the "
                        "coefficient. The headline is not a measurement-error "
                        "artifact.\n")
            else:
                f.write("**Verdict: ROBUST.** The shrunk-literacy coefficient "
                        "preserves sign and significance. It is somewhat smaller "
                        "in magnitude than the raw-literacy coefficient — the "
                        "shrinkage pulls noisy small-state estimates toward the "
                        "grand mean, mechanically compressing the moderator's "
                        "cross-sectional spread — but the regularity survives "
                        "the measurement-error correction.\n")
        else:
            f.write("**Verdict: NOT ROBUST to measurement-error correction.** "
                    "The headline coefficient does not survive empirical-Bayes "
                    "shrinkage of the noisy state-level literacy estimates. The "
                    "paper must disclose that the regularity is sensitive to "
                    "NFCS state-level measurement error and downgrade the "
                    "headline accordingly.\n")
    print(f"\ndelta = {delta:+.6f} ({res['delta_in_baseline_se']:+.3f} baseline SEs)")
    print(f"sign preserved: {res['sign_preserved']}, still significant: {res['still_significant']}")
    return res


if __name__ == "__main__":
    main()
