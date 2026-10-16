"""
Compare PCA + K-means (original paper, Ogden et al. 2019 Science, Fig 3) against
PCovC (Jorgensen, Lin, Vasavada & Cersonsky 2025, Mach. Learn.: Sci. Technol.,
https://doi.org/10.1088/2632-2153/ae466e) for visualizing AAV capsid mutant
tissue-tropism groups from the mouse biodistribution screen.

PCA+K-means is unsupervised: it embeds the 6-tissue selection profile in 2D
with no knowledge of tropism-group identity, then clusters post-hoc. PCovC is
supervised: it optimizes the 2D embedding to *simultaneously* reconstruct the
input data and separate known classes, controlled by a mixing parameter alpha
(alpha=1 recovers plain PCA, alpha=0 is purely classification-driven).

Class labels used here are the paper's own 3 tropism clusters (+ "other" for
unclustered mutants) -- the same `labels_important` assignment the original
notebook derives from K-means on the PCA scores. This keeps the comparison
apples-to-apples: same input features, same target groups, different
embedding objective.

Must be run with cwd = <repo>/x02_analysis (same as the original notebooks,
so the repo's relative data paths resolve), using the venv_pcovc environment
(Python >= 3.10, needs skmatter's PCovC which isn't in the PyPI release yet).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from skmatter.decomposition import PCovC

sys.path.append('../x01_process_data/')
sys.path.append('../settings/')
import x02_load_dataframes
import x03_compute_selections
from common_functions import codon_to_aa_selector
import common as a04_common

OUT_DIR = './pcovc_analysis'
pd_idx = pd.IndexSlice


def build_mouse_pca_input():
    """Reproduce a04_analyze_mouse_data.py exactly up through the PCA input matrix."""
    package_counts = x02_load_dataframes.load_packaging_df()
    package_counts.sort_index(inplace=True)
    mouse_counts = x02_load_dataframes.load_mouse_df()

    mouse_counts_filtered = mouse_counts.where(
        (mouse_counts > 10) & (mouse_counts < 31000), np.nan)

    new_mouse_selection = x03_compute_selections.compute_mouse_selections(
        package_counts=package_counts,
        mouse_counts=mouse_counts_filtered,
        wt_norm=True, drop=True, drop_tile=False, return_freq=False)

    package_selection = x03_compute_selections.compute_packaging_selection(
        package_counts, level='aa')

    mouse_aa_selection = new_mouse_selection['aa_selection']
    mouse_aa_sel_package_filtered = codon_to_aa_selector(
        package_selection[package_selection[('CMV', '0')] > .5], mouse_aa_selection)

    mini_tissues = mouse_aa_sel_package_filtered.loc[:, pd_idx[:, :, :, :, 'mini']]
    qiagen_tissues = mouse_aa_sel_package_filtered.loc[:, pd_idx[['blood', 'liver', 'spleen'], :, :, :]]
    mouse_data_for_corr_plot = pd.concat([mini_tissues, qiagen_tissues], axis=1)

    mouse_data_for_pca = mouse_data_for_corr_plot.T.groupby(level='organ').median().replace(
        [np.inf, -np.inf, 0], np.nan).apply(np.log2).T.dropna()
    mouse_data_for_pca_selection_filtered = mouse_data_for_pca[
        ((mouse_data_for_pca > 1).any(axis=1)) | ((mouse_data_for_pca < -1).any(axis=1))]

    return mouse_data_for_pca_selection_filtered


def build_paper_labels(X):
    """Reproduce the paper's PCA + K-means(9) -> 3-cluster ('labels_important') assignment.

    NOTE: raw KMeans cluster indices are not stable across scikit-learn versions
    even at a fixed random_state (the paper pinned scikit-learn==0.21.3). The
    original notebook's hardcoded raw-index -> cluster-id dicts assumed that
    version's index assignment and point at the wrong clusters under a current
    install; cluster identity is instead recovered from the data itself via
    a04_common.identify_tropism_clusters (see x02_analysis/common.py). The PCA
    sign is likewise environment-dependent -- flipping only PC2 here matches
    the published figure's orientation under the current scikit-learn/LAPACK
    build (see the equivalent fix in a04_analyze_mouse_data.py).
    """
    pca = PCA()
    pca_fit = pca.fit_transform(X)
    pca_sign = np.ones(pca_fit.shape[1])
    pca_sign[1] = -1
    pca_df = pd.DataFrame(pca_fit * pca_sign, index=X.index)

    kmean_1 = KMeans(n_clusters=9, random_state=0).fit(pca_df)
    raw_labels = pd.Series(kmean_1.labels_, index=X.index)
    cluster_map = a04_common.identify_tropism_clusters(raw_labels, X)
    labels_important = raw_labels.map(cluster_map).fillna(0).astype(int)

    return pca_df, labels_important


def align_embedding_orientation(embedding, y):
    """PCA/PCovC components are only defined up to sign, so each embedding's
    axes come out in an arbitrary orientation independent of the others.
    Flip axes (never rotate/reorder) so cluster 1 sits to the right of
    cluster 2 and cluster 3 sits above them, matching the published figure's
    layout, so all panels are visually comparable to it and to each other."""
    embedding = np.array(embedding, copy=True)
    y = np.asarray(y)
    if embedding[y == 1, 0].mean() < embedding[y == 2, 0].mean():
        embedding[:, 0] *= -1
    if embedding[y == 3, 1].mean() < embedding[np.isin(y, [1, 2]), 1].mean():
        embedding[:, 1] *= -1
    return embedding


def main():
    X = build_mouse_pca_input()
    pca_df, y = build_paper_labels(X)
    print(f"X shape: {X.shape}, class counts:\n{y.value_counts().sort_index()}")

    Xs = StandardScaler().fit_transform(X.to_numpy())

    # unsupervised PCA (same as paper, on standardized features so it's
    # comparable to PCovC's z-scored input -- the paper itself ran PCA
    # unstandardized, but that doesn't change what we're testing here)
    pca_2d = PCA(n_components=2).fit_transform(Xs)
    pca_2d = align_embedding_orientation(pca_2d, y.to_numpy())

    # PCovC at a few mixing values: alpha=1 (~pure PCA, for a sanity check
    # PCovC collapses to PCA as expected), alpha=0.5 (balanced), alpha=0
    # (embedding chosen almost purely to separate the known classes)
    mixings = [1.0, 0.5, 0.0]
    pcovc_embeddings = {}
    for alpha in mixings:
        model = PCovC(mixing=alpha, n_components=2, random_state=0, scale_z=True)
        emb = model.fit_transform(Xs, y.to_numpy())
        pcovc_embeddings[alpha] = align_embedding_orientation(emb, y.to_numpy())

    # --- quantitative comparison: how well does each 2D embedding separate
    # the paper's known tropism groups? silhouette score against the known
    # labels, and 5-fold KNN classification accuracy within the embedding.
    def score(embedding, y):
        sil = silhouette_score(embedding, y)
        knn_acc = cross_val_score(KNeighborsClassifier(n_neighbors=15), embedding, y, cv=5).mean()
        return sil, knn_acc

    results = []
    sil, acc = score(pca_2d, y)
    results.append(('PCA (unsupervised)', sil, acc))
    for alpha in mixings:
        sil, acc = score(pcovc_embeddings[alpha], y)
        results.append((f'PCovC (mixing={alpha})', sil, acc))

    results_df = pd.DataFrame(results, columns=['method', 'silhouette', 'knn_cv_accuracy'])
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUT_DIR, 'pcovc_vs_pca_scores.csv'), index=False)

    # --- figure: PCA+Kmeans (paper) vs PCovC embeddings, colored by the same labels
    label_names = {0: 'other', 1: 'cluster 1', 2: 'cluster 2', 3: 'cluster 3'}
    palette = [sns.color_palette('deep', 9)[i] for i in (7, 1, 2, 4)]  # matches a04's palette

    panels = [('PCA (unsupervised)\n+ K-means clusters (paper)', pca_2d)] + \
              [(f'PCovC, mixing={alpha}\n(1=pure PCA, 0=pure classification)', pcovc_embeddings[alpha])
               for alpha in mixings]

    fig, axes = plt.subplots(ncols=len(panels), figsize=[3.2 * len(panels), 3.2])
    for ax, (title, emb) in zip(axes, panels):
        plot_df = pd.DataFrame(emb[:, :2], columns=['x', 'y'])
        plot_df['label'] = y.map(label_names).to_numpy()
        sns.scatterplot(data=plot_df, x='x', y='y', hue='label',
                         hue_order=[label_names[i] for i in (0, 1, 2, 3)],
                         palette=palette, s=6, lw=0, alpha=.8, ax=ax, legend=(ax is axes[-1]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pcovc_vs_pca_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"Saved figure to {OUT_DIR}/pcovc_vs_pca_comparison.png")


if __name__ == '__main__':
    main()
