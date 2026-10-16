"""
Extends pcovc_mutation_organ_correlation.py's mutation-feature analysis
beyond its original 4 hand-picked physicochemical properties (Hydrophobicity,
Residue_volume, Polarity, net_charge).

Motivation, grounded in the source material for the Part-2-deep-dive deck:
Ogden et al. 2019's own accessible main text explicitly analyzes only ONE
biochemical property category (charge, positive vs negative) -- their own
code imports the full AAindex table (137 property scales per amino acid,
`data/meta/aa_index_0_center_norm_df.csv`) but no evidence was found that it
is actually used anywhere in the visible analysis pipeline. Our own prior
mutation-feature scripts used 4 of those 137 scales, chosen by hand. This
script asks: does using a properly de-redundant, broader panel of AAindex
properties explain meaningfully more per-organ selection variance than the
original 4, or were the original 4 already capturing most of what a
hand-picked "hydrophobicity/volume/polarity/charge" story can offer?

Method: AAindex is famously redundant (many of its 137 scales measure nearly
the same underlying property under different names/units). Rather than
picking more property names by eye, this hierarchically clusters ALL 137
scales by their correlation across the 20 canonical amino acids, cuts the
dendrogram to retain ~12 clusters, and keeps one representative scale
(closest to each cluster's medoid) per cluster -- a low-redundancy,
data-driven expansion instead of an arbitrarily larger hand-picked list.

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input
from pcovc_mutation_features import vr_region, AA_ORDER

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
ORIGINAL_4 = ['Hydrophobicity', 'Residue_volume', 'Polarity']  # net_charge is derived, handled separately
N_CLUSTERS = 12
RNG = 0


def select_low_redundancy_properties(aa_index, n_clusters=N_CLUSTERS, seed=RNG):
    """Hierarchically cluster all 137 AAindex property columns by their
    correlation across the 20 amino acids, cut to n_clusters, keep the
    column closest to each cluster's centroid (in correlation space)."""
    corr = aa_index.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - np.abs(corr)  # correlation distance; abs() so +/- versions of the same axis cluster together
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # enforce exact symmetry (float roundoff)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')

    reps = []
    for cid in sorted(set(cluster_ids)):
        members = [c for c, k in zip(aa_index.columns, cluster_ids) if k == cid]
        if len(members) == 1:
            reps.append(members[0])
            continue
        sub_corr = aa_index[members].corr().abs()
        medoid = sub_corr.mean(axis=1).idxmax()  # column most correlated with the rest of its cluster
        reps.append(medoid)
    return reps, cluster_ids


def main():
    X_tissue = build_mouse_pca_input()
    aa_index = pd.read_csv('../data/meta/aa_index_0_center_norm_df.csv', index_col=0)
    aa_index.index = AA_ORDER
    print(f"AAindex table: {aa_index.shape[1]} total properties across {aa_index.shape[0]} amino acids")

    reps, cluster_ids = select_low_redundancy_properties(aa_index)
    print(f"\nSelected {len(reps)} low-redundancy representative properties "
          f"(from {N_CLUSTERS}-cluster cut of all {aa_index.shape[1]}):")
    for r in reps:
        print(f"  - {r}")

    # net_charge is a derived combination (Positive_charge - Negative_charge),
    # not a raw AAindex column -- keep it explicitly alongside the cluster reps
    # since it's the one property Ogden et al. 2019's own main text discusses.
    # Also explicitly union in the original hand-picked 3 raw columns (even if
    # the clustering didn't happen to select them as a representative) so the
    # original-4-property model and the expanded model can be built from the
    # exact same feature table below, for a clean apples-to-apples R^2 comparison.
    net_charge = aa_index['Positive_charge'] - aa_index['Negative_charge']
    all_cols = list(dict.fromkeys(reps + ORIGINAL_4))  # union, preserve order, no dupes
    props = aa_index[all_cols].copy()
    props['net_charge'] = net_charge
    expanded_cols_raw = reps + ['net_charge']  # the actual "expanded, low-redundancy" panel
    original_cols_raw = ORIGINAL_4 + ['net_charge']

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
    all_delta_cols = [f'{c}_delta' for c in props.columns]
    feat[all_delta_cols] = feat[all_delta_cols].fillna(0.0)
    feat = pd.get_dummies(feat, columns=['vr_region', 'lib_type'], drop_first=False)
    feat.index = X_tissue.index

    expanded_delta_cols = [f'{c}_delta' for c in expanded_cols_raw]
    original_delta_cols = [f'{c}_delta' for c in original_cols_raw]

    # --- univariate Spearman, expanded (12-property, de-redundant) panel ---
    corr_df = pd.DataFrame(index=expanded_delta_cols, columns=TISSUES, dtype=float)
    pval_df = pd.DataFrame(index=expanded_delta_cols, columns=TISSUES, dtype=float)
    for feature in expanded_delta_cols:
        for tissue in TISSUES:
            r, p = spearmanr(feat[feature].astype(float), X_tissue[tissue])
            corr_df.loc[feature, tissue] = r
            pval_df.loc[feature, tissue] = p
    corr_df.to_csv(os.path.join(OUT_DIR, 'expanded_mutation_organ_spearman.csv'))

    n_tests = corr_df.size
    sig = pval_df < (0.05 / n_tests)
    print(f"\nExpanded panel: {len(expanded_delta_cols)} properties x {len(TISSUES)} organs = "
          f"{n_tests} tests, {sig.values.sum()} significant at Bonferroni p<0.05")
    print("\nMost significant property x organ pairs (top 10 by |r|):")
    flat = corr_df.stack().rename('r').reset_index()
    flat.columns = ['property', 'organ', 'r']
    flat['sig'] = [sig.loc[p, o] for p, o in zip(flat['property'], flat['organ'])]
    print(flat.reindex(flat.r.abs().sort_values(ascending=False).index).head(10).to_string(index=False))

    # --- multivariate R^2 comparison: original-4-property model vs expanded (12+1) model ---
    dummy_cols = [c for c in feat.columns if c not in all_delta_cols]
    ref_dummy = [c for c in dummy_cols if c.endswith('_none') or c.endswith('_sub')]

    def r2_per_organ(cols_numeric):
        Xs = feat[cols_numeric + [c for c in dummy_cols if c not in ref_dummy]].copy()
        Xs[cols_numeric] = StandardScaler().fit_transform(Xs[cols_numeric])
        Xs = Xs.astype(float).to_numpy()
        out = {}
        for tissue in TISSUES:
            y = X_tissue[tissue].to_numpy()
            out[tissue] = LinearRegression().fit(Xs, y).score(Xs, y)
        return out

    r2_original = r2_per_organ(original_delta_cols)
    r2_expanded = r2_per_organ(expanded_delta_cols)

    comparison = pd.DataFrame({'original_4_properties': r2_original, 'expanded_panel': r2_expanded})
    comparison['delta'] = comparison['expanded_panel'] - comparison['original_4_properties']
    comparison.to_csv(os.path.join(OUT_DIR, 'expanded_vs_original_r2.csv'))
    print(f"\nR^2 per organ, original {len(original_delta_cols)}-property model vs "
          f"expanded {len(expanded_delta_cols)}-property model:")
    print(comparison.round(3).to_string())

    # --- figure ---
    fig, axes = plt.subplots(ncols=2, figsize=[14, 6])
    sns.heatmap(corr_df.astype(float), cmap='RdBu_r', center=0, ax=axes[0],
                cbar_kws={'label': 'Spearman r'})
    sig_mask = sig.reindex(index=corr_df.index, columns=corr_df.columns)
    for i, feature in enumerate(corr_df.index):
        for j, tissue in enumerate(TISSUES):
            if sig_mask.loc[feature, tissue]:
                axes[0].text(j + 0.5, i + 0.5, '*', ha='center', va='center', fontsize=11)
    axes[0].set_title(f'Expanded panel ({len(expanded_delta_cols)} low-redundancy AAindex properties)\nvs per-organ selection')

    comparison.plot(kind='bar', y=['original_4_properties', 'expanded_panel'], ax=axes[1],
                     color=['tab:gray', 'tab:orange'])
    axes[1].set_ylabel('R^2 (organ selection explained)')
    axes[1].set_title('Original 4-property model vs. expanded de-redundant panel')
    axes[1].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'expanded_mutation_organ_correlation.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {OUT_DIR}/expanded_mutation_organ_correlation.png")


if __name__ == '__main__':
    main()
