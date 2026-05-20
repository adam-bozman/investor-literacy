# =====================================================================
# v7_within_retail_sub_decile.py
# Partitions the low-IO (retail-dominated) tercile into literacy deciles, IV
# deciles, and a 3x3 literacy-by-IV joint grid, estimating the three-way
# coefficient in each cell.
#
# Inputs:    _dfm_v7.parquet (cached merged firm-month panel: seed panel + corrected
#            Thomson s34 IO).
# Outputs:   output/stage3a/results_v7_sub_decile.json (no tables written)
# Paper:     Internet Appendix within-retail detail section
# Run order: see code/00_master.py
# =====================================================================

"""v7 — Row 2 (referee Major 2; triager row 2): within-retail sub-decile
partition of the low-IO tercile.

The structured referee asks: within the low-IO tercile (the "retail-dominated"
slice), partition firms into sub-deciles on literacy_z AND on iv. Do the wrong-
signed within-retail literacy gradient (the +0.0022, p=0.071 in v6) hold
uniformly, or concentrate in particular literacy bands and / or particular IV
bands? The discriminating reading: if it concentrates in *low* literacy + LOW
IV bands, the residual ownership / local-bias-intensity reading is favored
over the participation-margin / arbitrage-capacity reading.

Three analyses:
  (A) Sub-decile partition by LITERACY_Z within low-IO. Compute the three-way
      coefficient in each of 10 deciles. The mechanism predicts a monotone
      pattern (more negative three-way at higher literacy_z within low-IO).
      The v6 between-bin result is +0.0022 (low - high literacy STATES).
  (B) Sub-decile partition by IV within low-IO. Compute the three-way
      coefficient in each of 10 IV deciles. The v6 result is that decile 9
      only is family-wise significant (+0.0087 continuous-rank trend, deciles
      4-6 at +0.049).
  (C) Joint partition: 3 literacy bins x 3 IV bins within low-IO, three-way
      in each cell. The mechanism predicts the LOWEST three-way (most
      negative) in the high-literacy x mid-IV cell.

Output: results_v7_sub_decile.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = (r"C:/Users/adam.bozman/OneDrive - Washington State University "
        r"(email.wsu.edu)/Research/investor-attention-empirical")
EMP = os.path.join(ROOT, "code", "empirical")
sys.path.insert(0, EMP)
from deepen_estimators import twfe_three_way, FOCAL

np.random.seed(42)
DFM_CACHE = os.path.join(EMP, "_dfm_v7.parquet")
OUT_JSON = os.path.join(ROOT, "output/stage3a/results_v7_sub_decile.json")


def load_panel():
    return pd.read_parquet(DFM_CACHE)


def add_io_terciles(d, io_col):
    perm_io = d.groupby('permno')[io_col].mean().dropna()
    terc = pd.qcut(perm_io, 3, labels=['IO1_low', 'IO2_mid', 'IO3_high'])
    d = d.copy()
    d['io_grp'] = d['permno'].map(terc).astype('object')
    return d


def main():
    print("=== Row 2 within-retail sub-decile partition (v7) ===",
          flush=True)
    dfm = load_panel()
    dfm = dfm.dropna(subset=FOCAL + ['ret', 'hq_state']).copy()
    dfm = add_io_terciles(dfm, 'io_share_persist')
    d_lowio = dfm[dfm['io_grp'] == 'IO1_low'].copy()
    print(f"low-IO (high retail) firm-months: {len(d_lowio):,}, "
          f"permnos: {d_lowio['permno'].nunique()}", flush=True)

    results = {
        'task': 'Row 2 (referee Major 2; triager row 2): within-retail '
                'sub-decile partition tests. The low-IO tercile is split '
                'into sub-deciles on literacy and IV; the three-way is '
                'estimated in each cell. Discriminates between residual '
                'ownership heterogeneity (within-retail effect concentrated '
                'in extremes) and a uniformly-wrong-signed sophistication '
                'gradient.',
        'low_io_n_firm_months': int(len(d_lowio)),
        'low_io_n_permnos': int(d_lowio['permno'].nunique()),
    }

    # ----- (A) LITERACY DECILES within low-IO -----
    print("\n--- (A) literacy deciles within low-IO ---", flush=True)
    # rank firms by their persistent literacy (firm-mean literacy_z, since
    # literacy_z is state-level it's effectively a state-firm match)
    firm_lit = d_lowio.groupby('permno')['literacy_score_corrected'].mean()
    firm_lit_dec = pd.qcut(firm_lit, 10,
                           labels=[f'D{i}' for i in range(1, 11)],
                           duplicates='drop')
    d_lowio['lit_dec'] = d_lowio['permno'].map(firm_lit_dec).astype('object')
    lit_dec_results = []
    for dec in sorted(d_lowio['lit_dec'].dropna().unique()):
        sub = d_lowio[d_lowio['lit_dec'] == dec]
        if len(sub) < 1000 or sub['hq_state'].nunique() < 5:
            continue
        r = twfe_three_way(sub)
        mean_lit = float(sub['literacy_score_corrected'].mean())
        lit_dec_results.append({
            'decile': dec,
            'mean_literacy_z': mean_lit,
            'gamma': r['coef'],
            't_cgm_two_way': r['t'],
            'se_kind': r['se_kind'],
            't_state_clustered': r['t_state'],
            'n_obs': int(r['n_obs']),
            'n_states': int(sub['hq_state'].nunique()),
            'n_permnos': int(sub['permno'].nunique()),
        })
        print(f"  lit dec {dec} (mean lit_z={mean_lit:+.2f}): "
              f"gamma={r['coef']:+.6f}, state-t={r['t_state']:.2f}, "
              f"N={r['n_obs']:,}", flush=True)
    results['literacy_deciles'] = lit_dec_results

    # ----- (B) IV DECILES within low-IO -----
    print("\n--- (B) IV deciles within low-IO ---", flush=True)
    firm_iv = d_lowio.groupby('permno')['iv'].mean()
    firm_iv_dec = pd.qcut(firm_iv, 10, labels=[f'D{i}' for i in range(1, 11)],
                          duplicates='drop')
    d_lowio['iv_dec'] = d_lowio['permno'].map(firm_iv_dec).astype('object')
    iv_dec_results = []
    for dec in sorted(d_lowio['iv_dec'].dropna().unique()):
        sub = d_lowio[d_lowio['iv_dec'] == dec]
        if len(sub) < 1000 or sub['hq_state'].nunique() < 5:
            continue
        r = twfe_three_way(sub)
        mean_iv = float(sub['iv'].mean())
        iv_dec_results.append({
            'decile': dec,
            'mean_iv_z': mean_iv,
            'gamma': r['coef'],
            't_cgm_two_way': r['t'],
            't_state_clustered': r['t_state'],
            'se_kind': r['se_kind'],
            'n_obs': int(r['n_obs']),
            'n_states': int(sub['hq_state'].nunique()),
            'n_permnos': int(sub['permno'].nunique()),
        })
        print(f"  iv dec {dec} (mean iv_z={mean_iv:+.2f}): "
              f"gamma={r['coef']:+.6f}, state-t={r['t_state']:.2f}, "
              f"N={r['n_obs']:,}", flush=True)
    results['iv_deciles'] = iv_dec_results

    # ----- (C) 3x3 JOINT PARTITION (literacy x IV) -----
    print("\n--- (C) 3x3 literacy x IV joint partition within low-IO ---",
          flush=True)
    firm_lit_terc = pd.qcut(firm_lit, 3, labels=['LIT1', 'LIT2', 'LIT3'],
                            duplicates='drop')
    firm_iv_terc = pd.qcut(firm_iv, 3, labels=['IV1', 'IV2', 'IV3'],
                           duplicates='drop')
    d_lowio['lit_terc'] = d_lowio['permno'].map(firm_lit_terc).astype('object')
    d_lowio['iv_terc'] = d_lowio['permno'].map(firm_iv_terc).astype('object')
    joint = []
    for ltc in ['LIT1', 'LIT2', 'LIT3']:
        for itc in ['IV1', 'IV2', 'IV3']:
            sub = d_lowio[(d_lowio['lit_terc'] == ltc)
                          & (d_lowio['iv_terc'] == itc)]
            if len(sub) < 1000 or sub['hq_state'].nunique() < 5:
                continue
            r = twfe_three_way(sub)
            joint.append({
                'lit_terc': ltc, 'iv_terc': itc,
                'gamma': r['coef'],
                't_cgm_two_way': r['t'],
                't_state_clustered': r['t_state'],
                'n_obs': int(r['n_obs']),
                'n_states': int(sub['hq_state'].nunique()),
                'n_permnos': int(sub['permno'].nunique()),
            })
            print(f"  {ltc} x {itc}: gamma={r['coef']:+.6f}, "
                  f"state-t={r['t_state']:.2f}, N={r['n_obs']:,}",
                  flush=True)
    results['lit_x_iv_joint'] = joint

    # ----- summary -----
    if lit_dec_results:
        gammas_lit = [r['gamma'] for r in lit_dec_results]
        results['summary'] = {
            'literacy_decile_gamma_range': (min(gammas_lit), max(gammas_lit)),
            'literacy_decile_n_negative': sum(1 for g in gammas_lit if g < 0),
            'literacy_decile_n_positive': sum(1 for g in gammas_lit if g >= 0),
            'iv_decile_gamma_range': (
                min(r['gamma'] for r in iv_dec_results),
                max(r['gamma'] for r in iv_dec_results)),
            'iv_decile_n_negative': sum(
                1 for r in iv_dec_results if r['gamma'] < 0),
            'iv_decile_n_positive': sum(
                1 for r in iv_dec_results if r['gamma'] >= 0),
        }
    print(f"\nsummary: {results.get('summary')}", flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== wrote {OUT_JSON} ===", flush=True)


if __name__ == '__main__':
    main()
