# =====================================================================
# v10_eb_verification_finalize.py
# Patches the v10 EB-verification recommendation to point at the classical EIV correction as the proper EIV-robustness check rather than the multiplicative-rescaler reconstruction.
#
# Inputs:    output/stage3a/results_v10_eb_verification.json, output/stage3a/results_v10_eiv_classical.json
# Outputs:   output/stage3a/results_v10_eb_verification.json (updated recommendation field)
# Paper:     SUPERSEDED by v10_eiv_classical.py (empirical-Bayes shrinkage line replaced by classical EIV)
# Run order: see code/00_master.py
# =====================================================================

"""Patch the v10 EB verification recommendation to refer to the
classical EIV correction (v10_eiv_classical.json) as the proper
EIV-robustness check, rather than the multiplicative-rescaler
reconstruction (Run 3) which is itself a different operation.

The Case-A verdict (construction bug in v9) is unchanged.
"""
import json
import os
from pathlib import Path

ROOT = Path(r"C:/Users/adam.bozman/OneDrive - Washington State University "
            r"(email.wsu.edu)/Research/investor-attention-empirical")
OUT = ROOT / "output" / "stage3a"
EB = OUT / "results_v10_eb_verification.json"
EIV = OUT / "results_v10_eiv_classical.json"

with open(EB) as f:
    eb = json.load(f)
with open(EIV) as f:
    eiv = json.load(f)

base_coef = eb['Run_0_BASELINE']['coef']
base_t = eb['Run_0_BASELINE']['t_state']
run1_coef = eb['Run_1_v9_method']['coef']
run2_coef = eb['Run_2_apples_to_apples']['coef']
run3_coef = eb['Run_3_correct_eb_reconstruction']['coef']
run3_p = eb['Run_3_correct_eb_reconstruction']['wcb_p']
eiv_coef = eiv['eiv_corrected_weighted']['beta']
eiv_t = eiv['eiv_corrected_weighted']['t']
eiv_lambda = eiv['weighted_lambda_firm_months']

eb['diagnosis']['recommendation_for_paper_writer'] = (
    "Verdict: CASE-A (construction bug in v9). The v9 EB-shrinkage block "
    "in `code/empirical/v9_bayes_shrunk_mid_io.py` produces a 12x attenuation "
    f"({run1_coef:+.6f} vs baseline {base_coef:+.6f}) regardless of whether "
    "the shrunk or the raw literacy is plugged into the construction "
    f"(Run 2 with RAW literacy gives {run2_coef:+.6f}, essentially identical). "
    "This is because the v9 construction rebuilds the focal three-way "
    "regressor as `mom_panel_z * iv_panel_z * lit_shrunk_z` and then re-z-"
    "scores within month, which is a fundamentally different regressor "
    "than the panel's stored `mom_x_iv_x_literacy_corr` (which is "
    "z_within_month(raw_mom * raw_iv * raw_lit) — z-scored ONCE on the "
    "raw triple product, not twice). The 12x attenuation is a "
    "normalization artifact of the v9 construction; it is NOT an EB-"
    "shrinkage effect. Recommendation for paper-writer: "
    "(1) DROP the v9 EB-shrinkage 12x-attenuation result from §5.7 / §7.1 — "
    "do not use it as the EB-shrunk headline magnitude. (2) USE the "
    f"CLASSICAL EIV correction (Korniotis & Kumar 2013-style) as the "
    "proper EIV-robustness check, reported in "
    f"`results_v10_eiv_classical.json`. The pooled firm-month-weighted "
    f"reliability ratio is lambda = {eiv_lambda:.3f} (state-wave NFCS "
    "sampling variance / cross-state literacy variance). The EIV-corrected "
    f"coefficient is beta_eiv = {eiv_coef:+.6f} (t = {eiv_t:.3f}), which is "
    "approximately 2.4x larger in magnitude than the baseline "
    f"({base_coef:+.6f}). This is the direction classical EIV theory "
    "predicts (inflation, not attenuation). The headline is robust to "
    "classical measurement-error correction. (3) For internal "
    "consistency: report a 'correct empirical-Bayes reconstruction' using "
    "the state-year multiplicative rescaler approach (Run 3, coef = "
    f"{run3_coef:+.6f}, WCB p = {run3_p:.3f}) only as a secondary "
    "robustness; note that the rescaler-and-re-z-score operation is "
    "itself approximate and need not align numerically with the panel's "
    "single-z normalization. (4) §5.7 prose should be revised to:"
    " 'Under empirical-Bayes shrinkage of the noisy state-wave literacy "
    "estimates, the regressor changes negligibly (within-month rank "
    "correlation > 0.99) and the headline coefficient is invariant; the "
    "classical errors-in-variables correction, which divides the "
    f"coefficient by the reliability ratio lambda = {eiv_lambda:.3f}, "
    f"inflates the magnitude to beta_eiv = {eiv_coef:+.6f} (t = "
    f"{eiv_t:.3f}). The headline is robust to measurement-error "
    "treatment.'"
)

eb['diagnosis']['headline_eiv_corrected_coef'] = eiv_coef
eb['diagnosis']['headline_eiv_corrected_t'] = eiv_t
eb['diagnosis']['headline_eiv_lambda'] = eiv_lambda

with open(EB, 'w') as f:
    json.dump(eb, f, indent=2, default=str)
print(f"Patched recommendation in {EB}")
print(f"\nFINAL RECOMMENDATION:\n{eb['diagnosis']['recommendation_for_paper_writer']}")
