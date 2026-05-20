#!/usr/bin/env python3
# =====================================================================
# 00_master.py
# Master run script / run-order manifest for the replication package of
#   "The Interaction Zone: State Financial Literacy and the Cross-Section
#    of the Momentum Anomaly."
#
# This file documents the order in which the analysis scripts in
# code/empirical/ should be run and which manuscript/appendix exhibit each
# one produces. It can be run directly to execute the pipeline end-to-end
# (see RUN below), but note the manual-intervention and data-license caveats
# flagged in each stage. By default it runs in DRY-RUN mode and only prints
# the plan.
#
# -------------------------------------------------------------------------
# IMPORTANT CAVEATS (read before running):
#
#   1. DATA LICENSE. Steps 1-2 pull from WRDS (CRSP, Compustat, Thomson
#      Reuters s34) and SEC EDGAR. CRSP/Compustat/Thomson are subscription
#      data and are NOT redistributed with this repo (see data/raw/README.md).
#      You must have WRDS credentials and the persistent WRDS server running
#      (see code/utils/wrds_server.py / wrds_client.py) for these steps.
#      NFCS waves are public (FINRA Foundation) but also not shipped.
#
#   2. HARD-CODED PATHS. The analysis scripts were written against the
#      original project tree and hard-code ROOT to a path ending in
#      ".../Research/investor-attention-empirical" (NOT this -submission
#      repo). Before running, either (a) edit ROOT at the top of each script
#      to point at this repo, or (b) run them from a checkout that matches
#      the original layout. Outputs land under output/stage3a/ and
#      output/stage3a/tables/ in whichever tree ROOT names.
#
#   3. SHARED LIBRARY MODULES. deepen_estimators.py (imported by ~31 scripts)
#      and v9_helpers.py (imported by the v9_* scripts) are libraries, not
#      steps -- never run them directly; they must be importable on sys.path
#      (run scripts from inside code/empirical/, or add it to PYTHONPATH).
#
#   4. VERSION LADDERS. Several scripts are superseded earlier iterations
#      kept for provenance (marked SUPERSEDED below). The LIVE pipeline is
#      the unmarked steps. Superseded scripts need not be run to reproduce
#      the paper.
#
#   5. LONG WRDS PULLS. The Thomson s34 download (60 quarters) and the EDGAR
#      10-K-header pull are long-running and were executed in stages with
#      reconnects in the original run. Expect manual babysitting.
# =====================================================================

import subprocess
import sys
from pathlib import Path

EMP = Path(__file__).resolve().parent / "empirical"

# Set to True to actually execute (after fixing ROOT paths + WRDS). Default
# prints the plan only.
RUN = False

# (script, one-line purpose, paper exhibit). None script == narrative marker.
PIPELINE = [
    ("== STAGE 1: DATA COLLECTION (WRDS + EDGAR; license-restricted, manual) ==", None, None),
    ("deepen_thomson_s34_download_v6.py", "Download Thomson Reuters s34 13F holdings, 2009Q1-2023Q4 (LIVE IO source)", "feeds T2/T3/T4 + IA Thomson s34 construction"),
    ("deepen_hq_edgar_pull.py",           "Pull SEC EDGAR 10-K SGML-header HQ state history (point-in-time HQ)", "IA PIT-HQ / stable-HQ construction"),

    ("== STAGE 2: DATA CLEANING / PANEL BUILD ==", None, None),
    ("deepen_thomson_io_panel_v6.py", "Build the institutional-ownership panel (amendment-dedup, sole+shared) from s34 (LIVE builder)", "feeds T2/T3/T4 + IA s34 construction"),
    ("deepen_hq_relocation_droptest.py", "Construct HQ-relocation flags (hdr_changed / sec_disagree) -> stable-HQ subsample", "IA stable-HQ construction; tab_hq_relocation_droptest_v6.tex"),
    ("deepen_13f_split_v6.py",        "Merge IO panel into firm-month panel; build IO terciles + within-retail splits (LIVE)", "T2/T3/T4 + IA within-retail IV-decile"),

    ("== STAGE 3: HEADLINE RESULTS ==", None, None),
    ("v7_stable_hq_centerpiece.py", "Mid-IO three-way + within-retail gradient on the stable-HQ subsample (HEADLINE)", "T2 tab:headline, T5 tab:iv_within_mid_io + IA stable-HQ"),
    ("v9_iv_decile_within_mid_io.py", "Within-mid-IO IV-decile decomposition (persistent, stable-HQ, size-controlled)", "T5 tab:iv_within_mid_io + IA within-retail IV-decile"),
    ("v9_quadratic_io_stable_hq.py", "Quadratic-IO continuous-margin (interior IO* minimum) on stable-HQ", "T7 tab:form_grid + IA 20-bin quadratic"),
    ("v7_four_way_continuous.py",    "Four-way momentum x IV x literacy x IO continuous specification", "T7 tab:form_grid + IA 20-bin quadratic"),

    ("== STAGE 4: SUMMARY STATS & DESCRIPTIVES ==", None, None),
    ("v9_placeholders_resolve.py",  "Resolve descriptive/auxiliary numbers (HHI, college, s34 coverage, relocator chars)", "T1 tab:descriptives + IA HHI / alt-literacy / s34 / relocator"),

    ("== STAGE 5: MAIN-TABLE ROBUSTNESS ==", None, None),
    ("v7_pit_hq_centerpiece.py",    "Point-in-time HQ literacy reassignment; nested baseline + mid-IO collapse", "T6 tab:nested + IA PIT diagnostics / PIT mid-IO"),
    ("v7_pit_hq_literacy.py",       "PIT aggregate-headline that motivates the PIT centerpiece (origin of correction)", "IA PIT diagnostics"),
    ("deepen_college_horserace.py", "Literacy vs state college-attainment horse-race within mid-IO", "tab_college_horserace.tex (results/robustness)"),
    ("subsample_split.py",          "Pre/post-2015 subsample split (Fama-MacBeth Newey-West)", "T8 tab:prepost_grad + IA 5-window grid"),
    ("v9_pre_post_2015_inference.py", "Sub-period cross-state literacy gradient inference, size-controlled mid-IO", "T8 tab:prepost_grad + IA 5-window grid"),

    ("== STAGE 6: EXTENDED ROBUSTNESS (largely Internet Appendix) ==", None, None),
    ("deepen_wild_bootstrap.py",       "State-level wild-cluster bootstrap (Webb 6-pt, null imposed)", "tab_wild_bootstrap.tex + fig_wild_bootstrap_dist (IA inference battery)"),
    ("deepen_inference_divergence.py", "Leverage / jackknife-by-month inference-divergence diagnosis", "tab_leverage_analysis.tex (IA inference-divergence)"),
    ("v7_precision_fm.py",             "Precision Fama-MacBeth estimator", "IA full inference battery"),
    ("v7_permutation_stress_test.py",  "Permutation / randomization-inference stress test", "IA full inference battery"),
    ("v10_eiv_classical.py",           "Classical errors-in-variables correction, per-NFCS-wave reliability", "IA classical EIV (replaced empirical-Bayes)"),
    ("v10_iv_decile_tv.py",            "Time-varying IO within-retail IV-decile decomposition", "IA within-retail IV-decile (time-varying)"),
    ("v11_chow_quadratic.py",          "Chow / functional-form quadratic grid (latest)", "T7 tab:form_grid"),
    ("v10_parametric_grid.py",         "Parametric functional-form grid (earlier than v11)", "T7 tab:form_grid"),
    ("kk_orthogonality.py",            "Korniotis-Kumar business-cycle orthogonality", "IA Korniotis-Kumar block"),
    ("decile_trend_and_kk_multiproxy.py", "KK multi-proxy (unemployment + coincident index) + decile trend", "IA Korniotis-Kumar + within-retail IV-decile"),
    ("kk_bayes_subsample.py",          "KK block on subsamples", "IA Korniotis-Kumar + 5-window grid"),
    ("ff12_and_portfolio.py",          "Fama-French 12 industry / portfolio robustness", "IA industry/portfolio robustness"),
    ("v9_relocator_decomp.py",         "Relocator-vs-stable decomposition + refined-relocator", "IA relocator-vs-stable + refined-relocator"),
    ("v7_refined_reloc_and_iv_decomposition.py", "Refined-relocator subsample + within-retail IV decomposition", "IA refined-relocator + within-retail IV-decile"),
    ("v7_within_retail_sub_decile.py", "Within-retail sub-decile detail", "IA within-retail detail"),
    ("v9_variance_channel_diagnostic.py", "Variance-channel mechanism diagnostic", "IA mechanism / variance channel"),
    ("v9_vintage2009_terciles.py",     "2009-vintage fixed-tercile robustness", "IA robustness"),

    ("== STAGE 7: SECONDARY DiD DESIGN (Internet Appendix) ==", None, None),
    ("v9_did_point_estimate.py",   "Staggered-adoption DiD point estimate on the four-way outcome", "T9 tab:fs_iv + IA DiD stack"),
    ("v9_state_pension_control.py", "State-pension control robustness for the DiD (WRDS pull; caches parquet)", "T9 tab:fs_iv support + IA DiD"),
    ("v7_goodman_bacon.py",        "Goodman-Bacon decomposition of the DiD", "IA DiD stack"),
    ("v7_twfe_eiv_lag36.py",       "TWFE + lag-36 EIV-instrumented DiD diagnostics", "IA classical-EIV / DiD diagnostics"),

    ("== STAGE 8: FIGURES ==", None, None),
    ("# Figures (figure2_migration.png, figure3_time.png, punchline_size_controlled.png,", None, None),
    ("#  punchline_state_map.png, state_literacy_tilegram.png) were produced by the", None, None),
    ("#  state-view / plotting scripts in the original project's output/state_views build.", None, None),
    ("#  The PNG/PDF outputs ship in output/figures/. Regenerate from the panel once ROOT", None, None),
    ("#  is repointed; the plotting code lives with the state_views build in the source tree.", None, None),

    ("== SUPERSEDED (kept for provenance; NOT needed to reproduce) ==", None, None),
    ("# EDGAR-13F proxy line (replaced by Thomson s34): deepen_13f_download.py,", None, None),
    ("#   deepen_13f_download_v4.py, deepen_io_panel.py, deepen_13f_split.py,", None, None),
    ("#   deepen_13f_split_v4.py, deepen_13f_split_v5.py, deepen_thomson_io_panel.py,", None, None),
    ("#   deepen_thomson_s34_download.py  (-> IA EDGAR-proxy cross-check only)", None, None),
    ("# Empirical-Bayes shrinkage line (replaced by classical EIV, v10_eiv_classical.py):", None, None),
    ("#   bayes_shrinkage_literacy.py, v9_bayes_shrunk_mid_io.py, v10_eb_verification.py,", None, None),
    ("#   v10_eb_verification_finalize.py", None, None),
    ("# Misc intermediate diagnostics: v8_diagnostic_three.py, reproduction_check.py,", None, None),
    ("#   v9_seven_pct_worked.py, v9_placeholders_resolve.py (partial)", None, None),
]


def main():
    print("=" * 70)
    print("REPLICATION RUN PLAN — The Interaction Zone")
    print("RUN =", RUN, "(set RUN=True in this file to execute; read caveats first)")
    print("=" * 70)
    for name, purpose, paper in PIPELINE:
        if purpose is None:
            print(name)
            continue
        print(f"\n  {name}")
        print(f"      purpose : {purpose}")
        print(f"      paper   : {paper}")
        if RUN:
            script = EMP / name
            if not script.exists():
                print(f"      SKIP    : {script} not found")
                continue
            print(f"      RUNNING : {script}")
            subprocess.run([sys.executable, str(script)], cwd=str(EMP), check=True)


if __name__ == "__main__":
    main()
