# =====================================================================
# v9_relocator_decomp.py
# Decomposes the full-panel-vs-stable-HQ mid/low ratio gap via by-tercile relocator counts and a relocator-only headline three-way (selection diagnostic).
#
# Inputs:    _dfm_v7.parquet, _dfm_stable_hq.parquet, _hq_edgar_state_v6.parquet (via v9_helpers); deepen_estimators.twfe_three_way
# Outputs:   output/stage3a/results_v9_relocator_decomposition.json (printed diagnostics + JSON)
# Paper:     IA relocator-vs-stable + refined-relocator (droptest) sections
# Run order: see code/00_master.py
# =====================================================================

"""v9 Test 1+2 — Relocator decomposition + relocator-only headline.

Triage [FIX] Item 1 (Critical). The structured referee's Comment 1 showed
mid/low ratio is only 1.09 on full panel under static literacy (IA Table 4)
vs 2.84 on stable-HQ (paper's Table 2). Tests:

  (1) By-tercile firm-month counts on (a) relocators (1,366 firms) and
      (b) stable-HQ (6,081 firms): are relocator firm-months over-
      represented in IO1 of the FULL panel? Report counts and shares.

  (2) Relocator-only headline: three-way TWFE on the 1,366-firm relocator
      subsample alone (persistent + time-varying literacy, state + month
      clustered). Report gamma_low, gamma_mid, gamma_high and mid/low
      ratio. If relocators contribute coefficients near zero in IO1, the
      stable-HQ "1.09 -> 2.84" jump is sample-selection driven.

Output: output/stage3a/results_v9_relocator_decomposition.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from v9_helpers import (load_full_panel, load_stable_hq, build_relocator_set,
                        add_io_terciles, save_json, OUT)
from deepen_estimators import twfe_three_way, FOCAL

np.random.seed(42)
OUT_JSON = os.path.join(OUT, "results_v9_relocator_decomposition.json")


def by_tercile_counts(d, label, io_col):
    """Compute by-tercile firm-month counts and shares."""
    d_io = d[d[io_col].notna()].copy()
    d_io = add_io_terciles(d_io, io_col)
    counts = d_io.groupby('io_grp').size().to_dict()
    perm_counts = d_io.groupby('io_grp')['permno'].nunique().to_dict()
    total = sum(counts.values())
    out = {
        "label": label,
        "io_col": io_col,
        "firm_month_counts": {k: int(counts.get(k, 0))
                              for k in ['IO1_low', 'IO2_mid', 'IO3_high']},
        "firm_month_shares": {k: float(counts.get(k, 0) / total)
                              for k in ['IO1_low', 'IO2_mid', 'IO3_high']},
        "firm_counts": {k: int(perm_counts.get(k, 0))
                        for k in ['IO1_low', 'IO2_mid', 'IO3_high']},
        "total_firm_months": int(total),
    }
    return out


def relocator_share_by_tercile(d_full, reloc_set, io_col):
    """For each IO tercile of the full panel, what fraction of firm-months
    are relocator firm-months?"""
    d_io = d_full[d_full[io_col].notna()].copy()
    d_io = add_io_terciles(d_io, io_col)
    d_io['is_reloc'] = d_io['permno'].isin(reloc_set).astype(int)
    out = {}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = d_io[d_io['io_grp'] == g]
        n_total = len(sub)
        n_reloc = int(sub['is_reloc'].sum())
        out[g] = {
            "n_firm_months_total": int(n_total),
            "n_firm_months_relocator": n_reloc,
            "relocator_share_of_firm_months": float(n_reloc / max(n_total, 1)),
            "n_firms_total": int(sub['permno'].nunique()),
            "n_firms_relocator": int(sub[sub['is_reloc'] == 1]['permno'].nunique()),
        }
    return out


def headline_by_tercile(d, label, io_col):
    """Headline three-way per tercile."""
    d_io = d[d[io_col].notna()].copy()
    d_all = d_io.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    d_all = add_io_terciles(d_all, io_col)
    out = {"label": label, "io_col": io_col,
           "estimation_sample_n": int(len(d_all)), "tercile_results": {}}
    for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
        sub = d_all[d_all['io_grp'] == g]
        if len(sub) < 100 or sub['hq_state'].nunique() < 3:
            out['tercile_results'][g] = {
                "skip": "insufficient data",
                "n_obs": int(len(sub)),
                "n_states": int(sub['hq_state'].nunique())}
            continue
        r = twfe_three_way(sub)
        out['tercile_results'][g] = {
            k: r[k] for k in ['coef', 'se', 't', 'se_state', 't_state',
                              'n_obs', 'se_kind']}
        print(f"    {label} {g}: gamma={r['coef']:+.6f} "
              f"state_t={r['t_state']:.2f} n={r['n_obs']:,}", flush=True)
    return out


def main():
    t0 = time.time()
    print("=== v9 Test 1+2: Relocator decomposition + relocator-only headline "
          "===", flush=True)

    print("\n--- Loading panels ---", flush=True)
    d_full = load_full_panel()
    d_stable = load_stable_hq()
    print(f"  full panel: {len(d_full):,} firm-months, "
          f"{d_full['permno'].nunique()} permnos", flush=True)
    print(f"  stable-HQ: {len(d_stable):,} firm-months, "
          f"{d_stable['permno'].nunique()} permnos", flush=True)

    print("\n--- Building relocator set (1,366-firm v6 def) ---", flush=True)
    reloc_set = build_relocator_set()
    n_reloc = len(reloc_set)
    print(f"  relocator firms: {n_reloc}", flush=True)
    d_reloc = d_full[d_full['permno'].isin(reloc_set)].copy()
    print(f"  relocator firm-months: {len(d_reloc):,}, "
          f"{d_reloc['permno'].nunique()} permnos", flush=True)

    results = {
        "test": "v9 Test 1+2: relocator decomposition + relocator-only headline",
        "triage_fix": "Item 1 (Critical)",
        "samples": {
            "full_panel_firm_months": int(len(d_full)),
            "full_panel_firms": int(d_full['permno'].nunique()),
            "stable_hq_firm_months": int(len(d_stable)),
            "stable_hq_firms": int(d_stable['permno'].nunique()),
            "relocator_firm_months": int(len(d_reloc)),
            "relocator_firms": int(d_reloc['permno'].nunique()),
            "n_relocator_set": n_reloc,
        },
    }

    # ============================================================
    # Part 1: By-tercile firm-month counts (Comment 1 reconciliation)
    # ============================================================
    print("\n=== Part 1: By-tercile firm-month counts (persistent IO) ===",
          flush=True)
    results['part1_counts'] = {}
    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n  --- {label} ---", flush=True)
        full_cnt = by_tercile_counts(d_full, "full_panel", io_col)
        stable_cnt = by_tercile_counts(d_stable, "stable_hq", io_col)
        reloc_cnt = by_tercile_counts(d_reloc, "relocator_only", io_col)
        reloc_share_full = relocator_share_by_tercile(d_full, reloc_set,
                                                     io_col)
        print(f"    full panel firm-month shares: "
              f"{full_cnt['firm_month_shares']}", flush=True)
        print(f"    stable-HQ firm-month shares: "
              f"{stable_cnt['firm_month_shares']}", flush=True)
        print(f"    relocator-only firm-month shares: "
              f"{reloc_cnt['firm_month_shares']}", flush=True)
        print(f"    relocator share of each tercile (full panel):",
              flush=True)
        for g in ['IO1_low', 'IO2_mid', 'IO3_high']:
            print(f"      {g}: "
                  f"{reloc_share_full[g]['relocator_share_of_firm_months']:.3f}"
                  f" ({reloc_share_full[g]['n_firm_months_relocator']:,}/"
                  f"{reloc_share_full[g]['n_firm_months_total']:,})",
                  flush=True)
        results['part1_counts'][label] = {
            "full_panel": full_cnt,
            "stable_hq": stable_cnt,
            "relocator_only": reloc_cnt,
            "relocator_share_within_full_panel_terciles": reloc_share_full,
        }

    # ============================================================
    # Part 2: Relocator-only headline regression (does the headline survive?)
    # ============================================================
    print("\n=== Part 2: Headline three-way on relocators-only ===",
          flush=True)
    results['part2_headlines'] = {}
    for io_col, label in [('io_share_persist', 'persistent_IO'),
                          ('io_share', 'time_varying_IO')]:
        print(f"\n  --- {label} ---", flush=True)
        # Stable-HQ headline (for reference)
        stable_hl = headline_by_tercile(d_stable, "stable_hq", io_col)
        # Full panel
        full_hl = headline_by_tercile(d_full, "full_panel", io_col)
        # Relocator only
        reloc_hl = headline_by_tercile(d_reloc, "relocator_only", io_col)

        # Compute mid/low ratios where defined
        def ratio(d):
            try:
                if ('IO2_mid' in d['tercile_results']
                        and 'IO1_low' in d['tercile_results']
                        and 'skip' not in d['tercile_results']['IO2_mid']
                        and 'skip' not in d['tercile_results']['IO1_low']):
                    mid = d['tercile_results']['IO2_mid']['coef']
                    low = d['tercile_results']['IO1_low']['coef']
                    return float(mid / low) if low != 0 else None
            except KeyError:
                return None
            return None

        stable_hl['mid_low_ratio'] = ratio(stable_hl)
        full_hl['mid_low_ratio'] = ratio(full_hl)
        reloc_hl['mid_low_ratio'] = ratio(reloc_hl)

        print(f"    stable-HQ mid/low ratio: {stable_hl['mid_low_ratio']}",
              flush=True)
        print(f"    full-panel mid/low ratio: {full_hl['mid_low_ratio']}",
              flush=True)
        print(f"    relocator-only mid/low ratio: "
              f"{reloc_hl['mid_low_ratio']}", flush=True)

        results['part2_headlines'][label] = {
            "stable_hq": stable_hl,
            "full_panel": full_hl,
            "relocator_only": reloc_hl,
        }

    # ============================================================
    # Verdict
    # ============================================================
    # Mechanism: if relocator-only coefficient is near zero in IO1, that
    # confirms the stable-HQ 2.84 mid/low ratio is partly selection-driven
    # (relocators in IO1 were *attenuating* the full-panel mid/low ratio).
    persistent = results['part2_headlines']['persistent_IO']
    relo_low_pers = (persistent['relocator_only']['tercile_results']
                     .get('IO1_low', {}).get('coef'))
    stable_low_pers = (persistent['stable_hq']['tercile_results']
                       .get('IO1_low', {}).get('coef'))
    full_low_pers = (persistent['full_panel']['tercile_results']
                     .get('IO1_low', {}).get('coef'))

    verdict_notes = []
    verdict_notes.append(
        f"Stable-HQ IO1 persistent gamma = {stable_low_pers}; "
        f"full-panel IO1 persistent gamma = {full_low_pers}; "
        f"relocator-only IO1 persistent gamma = {relo_low_pers}.")
    if (relo_low_pers is not None and stable_low_pers is not None
            and full_low_pers is not None):
        # Is the full-panel coefficient closer to the relocator-only than the
        # stable-HQ? Then stable-HQ Z is selection-driven attenuation.
        d_stable_to_full = abs(full_low_pers - stable_low_pers)
        d_reloc_to_full = abs(full_low_pers - relo_low_pers)
        # Or: does relocator-only IO1 attenuate (sign change)?
        attenuation_reloc = abs(relo_low_pers) < 0.5 * abs(stable_low_pers)
        verdict_notes.append(
            f"|relocator_only_IO1 / stable_HQ_IO1| = "
            f"{abs(relo_low_pers / stable_low_pers):.3f} "
            f"(attenuation_indicator = {attenuation_reloc}).")

    results['verdict'] = (
        "DESCRIPTIVE" if (relo_low_pers is None) else
        ("CONFIRMS_SELECTION_DRIVEN" if attenuation_reloc else "WEAKENS"))
    results['verdict_notes'] = verdict_notes

    results['meta'] = {"elapsed_s": round(time.time() - t0, 2),
                       "seed": 42}
    save_json(results, OUT_JSON)
    print(f"\n=== Verdict: {results['verdict']} ===", flush=True)
    print(f"=== Elapsed: {results['meta']['elapsed_s']:.1f}s ===",
          flush=True)


if __name__ == '__main__':
    main()
