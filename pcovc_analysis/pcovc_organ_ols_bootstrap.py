"""
Adds bootstrap CIs to pcovc_mutation_organ_correlation_expanded.py's
multivariate OLS coefficients (mutation feature -> per-organ selection),
the same rigor already applied to the pxz_ mutation-to-cluster weights in
pcovc_mutation_features_expanded.py. Reports, per organ, which of the 13
expanded properties have a coefficient whose 95% bootstrap CI excludes 0
(stable) versus not (indistinguishable from noise at n=2,007).

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input
from pcovc_mutation_organ_correlation_expanded import (
    select_low_redundancy_properties, N_CLUSTERS, ORIGINAL_4)
from pcovc_mutation_features import vr_region, AA_ORDER

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
RNG = 0
N_BOOT = 1000


def main():
    X_tissue = build_mouse_pca_input()
    aa_index = pd.read_csv('../data/meta/aa_index_0_center_norm_df.csv', index_col=0)
    aa_index.index = AA_ORDER
    reps, _ = select_low_redundancy_properties(aa_index, n_clusters=N_CLUSTERS)
    net_charge = aa_index['Positive_charge'] - aa_index['Negative_charge']
    props = aa_index[reps].copy()
    props['net_charge'] = net_charge

    lookup = pd.read_csv('../data/meta/AAV2scan_chip_lookup_table.txt')
    wt_by_pos = lookup[lookup['is_wt_aa'] == 1].set_index('abs_pos')['aa'].to_dict()

    rows = []
    for abs_pos, aa, wt_bc, lib_type in X_tissue.index:
        wt_aa = wt_by_pos.get(abs_pos)
        row = {'vr_region': vr_region(abs_pos), 'lib_type': lib_type}
        if aa in props.index and wt_aa in props.index:
            for col in props.columns:
                row[f'{col}_delta'] = props.loc[aa, col] - props.loc[wt_aa, col]
        else:
            for col in props.columns:
                row[f'{col}_delta'] = np.nan
        rows.append(row)
    feat = pd.DataFrame(rows)
    delta_cols = [f'{c}_delta' for c in props.columns]
    feat[delta_cols] = feat[delta_cols].fillna(0.0)
    feat = pd.get_dummies(feat, columns=['vr_region', 'lib_type'], drop_first=False)
    feat.index = X_tissue.index

    dummy_cols = [c for c in feat.columns if c not in delta_cols]
    ref_dummy = [c for c in dummy_cols if c.endswith('_none') or c.endswith('_sub')]
    model_dummy_cols = [c for c in dummy_cols if c not in ref_dummy]
    all_cols = delta_cols + model_dummy_cols
    feature_names = all_cols

    Xall = feat[all_cols].copy()
    Xall[delta_cols] = StandardScaler().fit_transform(Xall[delta_cols])
    Xall = Xall.astype(float).to_numpy()
    y_all = X_tissue[TISSUES].to_numpy()
    n = Xall.shape[0]

    # --- point estimate ---
    point = {}
    for i, tissue in enumerate(TISSUES):
        point[tissue] = LinearRegression().fit(Xall, y_all[:, i]).coef_

    # --- bootstrap ---
    rng = np.random.default_rng(RNG)
    boot = {t: [] for t in TISSUES}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb, yb = Xall[idx], y_all[idx]
        for i, tissue in enumerate(TISSUES):
            boot[tissue].append(LinearRegression().fit(Xb, yb[:, i]).coef_)

    summary_rows = []
    for tissue in TISSUES:
        arr = np.array(boot[tissue])
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        stable = (lo > 0) | (hi < 0)
        for feat_name, m, l, h, st, pt in zip(feature_names, mean, lo, hi, stable, point[tissue]):
            summary_rows.append(dict(organ=tissue, feature=feat_name, point_estimate=pt,
                                      boot_mean=m, boot_lo=l, boot_hi=h, stable=st))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, 'expanded_organ_ols_bootstrap.csv'), index=False)

    print("Stable (bootstrap CI excludes 0) numeric-property coefficients per organ, ranked by |mean|:")
    for tissue in TISSUES:
        sub = summary[(summary.organ == tissue) & (summary.stable) & (summary.feature.isin(delta_cols))].copy()
        sub = sub.reindex(sub.boot_mean.abs().sort_values(ascending=False).index)
        print(f"\n  {tissue} ({len(sub)} stable numeric properties):")
        print(sub[['feature', 'boot_mean', 'boot_lo', 'boot_hi']].head(6).to_string(index=False))

    n_numeric_tests = len(TISSUES) * len(delta_cols)
    n_stable_numeric = summary[summary.feature.isin(delta_cols)].stable.sum()
    print(f"\n{n_stable_numeric}/{n_numeric_tests} organ x numeric-property coefficients are bootstrap-stable")

    # --- figure ---
    def _short(name, max_len=24):
        name = name.replace('_delta', '').replace('_', ' ')
        return name if len(name) <= max_len else name[:max_len - 1] + '…'

    fig, axes = plt.subplots(nrows=len(TISSUES), figsize=[8, 2.6 * len(TISSUES)], constrained_layout=True)
    for ax, tissue in zip(axes, TISSUES):
        sub = summary[(summary.organ == tissue) & (summary.feature.isin(delta_cols))].copy()
        sub = sub.reindex(sub.boot_mean.abs().sort_values(ascending=False).index).head(6)
        sub = sub.iloc[::-1]
        labels = [_short(f) for f in sub.feature]
        colors = ['#B0B0B0' if not st else ('#4C72B0' if m < 0 else '#DD8452')
                  for m, st in zip(sub.boot_mean, sub.stable)]
        xerr = np.array([sub.boot_mean - sub.boot_lo, sub.boot_hi - sub.boot_mean])
        ax.barh(labels, sub.boot_mean, xerr=xerr, color=colors, capsize=2, ecolor='black', error_kw={'linewidth': 0.8})
        ax.axvline(0, color='black', lw=.6)
        ax.set_title(f'{tissue} (gray = bootstrap CI includes 0)', fontsize=10)
        ax.tick_params(axis='y', labelsize=8.5)
    axes[-1].set_xlabel('standardized OLS coefficient (bootstrap mean +/- 95% CI)', fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, 'expanded_organ_ols_bootstrap.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT_DIR}/expanded_organ_ols_bootstrap.png")


if __name__ == '__main__':
    main()
