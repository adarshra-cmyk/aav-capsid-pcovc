"""
Which mutation features correlate with selection in which organ, directly?

pcovc_mutation_features.py goes mutation-features -> 3 tropism clusters,
which routes everything through the paper's PCA+K-means clustering step.
This instead correlates mutation features straight against the 6 continuous
per-tissue log2(selection) values -- no PCA, no K-means, no cluster labels
at all. Two views are produced per organ:

1. Univariate: Spearman correlation of each feature alone against that
   organ's selection (simple, but features can confound each other, e.g.
   a VR region that happens to also be enriched for charge-losing mutations).
2. Multivariate: standardized OLS coefficient for each feature, holding all
   other mutation features fixed, per organ. This is the "controlled for
   confounds" version of the same question.

Run with cwd = <repo>/x02_analysis, using venv_pcovc (Python >= 3.10).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input
from pcovc_mutation_features import build_mutation_features

OUT_DIR = './pcovc_analysis'
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']


def main():
    X_tissue = build_mouse_pca_input()  # index=(abs_pos, aa, wt_bc, lib_type), cols=6 tissues, log2(selection)
    feat, delta_cols = build_mutation_features(X_tissue.index)
    feat.index = X_tissue.index
    # no stop-codon variants survive into the mouse biodistribution dataset
    # (filtered upstream), so is_stop is constant here -- drop it.
    feat = feat.drop(columns=['is_stop'])
    numeric_cols = delta_cols
    dummy_cols = [c for c in feat.columns if c not in numeric_cols]

    # --- 1. univariate Spearman correlation, feature vs each organ ---
    corr = pd.DataFrame(index=feat.columns, columns=TISSUES, dtype=float)
    pval = pd.DataFrame(index=feat.columns, columns=TISSUES, dtype=float)
    for feature in feat.columns:
        for tissue in TISSUES:
            r, p = spearmanr(feat[feature].astype(float), X_tissue[tissue])
            corr.loc[feature, tissue] = r
            pval.loc[feature, tissue] = p
    corr.to_csv(os.path.join(OUT_DIR, 'pcovc_mutation_organ_spearman.csv'))
    pval.to_csv(os.path.join(OUT_DIR, 'pcovc_mutation_organ_spearman_pvalues.csv'))

    n_tests = corr.size
    sig = pval < (0.05 / n_tests)  # Bonferroni across the whole feature x organ grid
    print(f"Spearman correlation, {n_tests} feature x organ tests, "
          f"{sig.values.sum()} significant at Bonferroni-corrected p<0.05")

    # --- 2. multivariate: standardized OLS coefficient per organ, all features jointly ---
    # each one-hot group (vr_region_*, lib_type_*) sums to a constant (1) across
    # every row, which is exactly collinear with the intercept -- an unregularized
    # OLS fit on the full dummy set has no unique solution for those coefficients
    # (sklearn's pinv-based solver still returns *a* fit, but individual coefficients
    # become arbitrary/unstable). Drop one reference level per group -- "not in any
    # VR loop" and "substitution" -- so the remaining coefficients are interpretable
    # relative to that baseline.
    Xs = feat.drop(columns=['vr_region_none', 'lib_type_sub']).copy()
    ols_dummy_cols = [c for c in dummy_cols if c not in ('vr_region_none', 'lib_type_sub')]
    Xs[numeric_cols] = StandardScaler().fit_transform(Xs[numeric_cols])
    Xs[ols_dummy_cols] = Xs[ols_dummy_cols].astype(float)
    Xs_arr = Xs.to_numpy()

    coef = pd.DataFrame(index=Xs.columns, columns=TISSUES, dtype=float)
    r2 = {}
    for tissue in TISSUES:
        y = X_tissue[tissue].to_numpy()
        lr = LinearRegression().fit(Xs_arr, y)
        coef[tissue] = lr.coef_
        r2[tissue] = lr.score(Xs_arr, y)
    coef.to_csv(os.path.join(OUT_DIR, 'pcovc_mutation_organ_ols_coef.csv'))
    print("\nOLS R^2 per organ (mutation features only, no tissue data leakage):")
    for t, v in r2.items():
        print(f"  {t}: {v:.3f}")

    # --- figure: two heatmaps side by side ---
    fig, axes = plt.subplots(ncols=2, figsize=[13, 6])
    sns.heatmap(corr.astype(float), cmap='RdBu_r', center=0, ax=axes[0],
                cbar_kws={'label': 'Spearman r'}, annot=False)
    axes[0].set_title('Univariate: feature vs organ selection')
    sig_mask = sig.reindex(index=corr.index, columns=corr.columns)
    for i, feature in enumerate(corr.index):
        for j, tissue in enumerate(TISSUES):
            if sig_mask.loc[feature, tissue]:
                axes[0].text(j + 0.5, i + 0.5, '*', ha='center', va='center', color='black', fontsize=12)

    sns.heatmap(coef.astype(float), cmap='RdBu_r', center=0, ax=axes[1],
                cbar_kws={'label': 'standardized OLS coef'}, annot=False)
    axes[1].set_title('Multivariate: coefficient holding other features fixed')

    fig.suptitle('Mutation features vs per-organ selection (* = Bonferroni-significant, left panel)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pcovc_mutation_organ_correlation.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {OUT_DIR}/pcovc_mutation_organ_correlation.png")


if __name__ == '__main__':
    main()
