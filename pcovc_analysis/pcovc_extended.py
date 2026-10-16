"""
Extends pcovc_biodistribution.py with three follow-ups:

1. De-circularized labels: instead of the paper's own PCA+K-means clusters
   (which PCovC could trivially "win" against just by being fit on labels
   that are themselves PCA-shaped), also test against an independent label
   -- for each mutant, which of the 6 tissues shows the strongest |log2
   selection| deviation from WT. This doesn't reference PCA or K-means at
   all.

2. Uncertainty quantification: bootstrap CIs on silhouette (resampling
   points against the fixed, already-fit embedding) and repeated stratified
   CV for KNN accuracy (20 repeats x 5 folds = 100 fold estimates), for
   both label schemes, so we can tell if PCovC's edge is real or noise.

3. Interpretability: PCovC's whole pitch vs. plain PCA is that its loadings
   stay interpretable while incorporating class information. Pull
   .components_ from both PCA and PCovC(mixing=0.5) and compare which
   tissues drive each component.

Run with cwd = <repo>/x02_analysis, using venv_pcovc (Python >= 3.10).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from skmatter.decomposition import PCovC

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input, build_paper_labels

OUT_DIR = './pcovc_analysis'
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
RNG = 0


def build_tissue_labels(X):
    """Independent label: which tissue shows the strongest |log2 selection|
    deviation for each mutant. Doesn't reference PCA or K-means at all."""
    return X.abs().idxmax(axis=1)


def bootstrap_silhouette_ci(embedding, y, n_boot=1000, seed=RNG):
    rng = np.random.default_rng(seed)
    n = embedding.shape[0]
    scores = []
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # silhouette needs >=2 classes present in the resample
        if len(np.unique(y_arr[idx])) < 2:
            continue
        scores.append(silhouette_score(embedding[idx], y_arr[idx]))
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def repeated_cv_knn_ci(embedding, y, n_splits=5, n_repeats=20, seed=RNG):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=15), embedding, y, cv=cv)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def evaluate(embedding, y, method_name, label_scheme):
    sil_mean, sil_lo, sil_hi = bootstrap_silhouette_ci(embedding, y)
    knn_mean, knn_lo, knn_hi = repeated_cv_knn_ci(embedding, y)
    return {
        'label_scheme': label_scheme,
        'method': method_name,
        'silhouette_mean': sil_mean, 'silhouette_lo': sil_lo, 'silhouette_hi': sil_hi,
        'knn_acc_mean': knn_mean, 'knn_acc_lo': knn_lo, 'knn_acc_hi': knn_hi,
    }


def main():
    X = build_mouse_pca_input()
    pca_df, y_paper = build_paper_labels(X)
    y_tissue = build_tissue_labels(X)

    print("paper-cluster label counts:\n", y_paper.value_counts().sort_index())
    print("\ntissue-of-max-deviation label counts:\n", y_tissue.value_counts())

    Xs = StandardScaler().fit_transform(X.to_numpy())

    label_schemes = {'paper_clusters (PCA+Kmeans-derived)': y_paper.to_numpy(),
                      'tissue_of_max_deviation (independent)': y_tissue.to_numpy()}

    pca_2d = PCA(n_components=2, random_state=RNG).fit(Xs)
    pca_embedding = pca_2d.transform(Xs)

    results = []
    loadings = {}

    for scheme_name, y in label_schemes.items():
        # PCA doesn't use labels, but re-evaluate per scheme since silhouette/knn
        # depend on which labels we're scoring against
        results.append(evaluate(pca_embedding, y, 'PCA (unsupervised)', scheme_name))

        pcovc = PCovC(mixing=0.5, n_components=2, random_state=RNG, scale_z=True)
        pcovc_embedding = pcovc.fit_transform(Xs, y)
        results.append(evaluate(pcovc_embedding, y, 'PCovC (mixing=0.5)', scheme_name))

        loadings[scheme_name] = pcovc.components_  # (2, 6)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUT_DIR, 'pcovc_extended_scores.csv'), index=False)

    # --- interpretability: PCA loadings vs PCovC loadings (per label scheme) ---
    loadings_rows = []
    for comp_i in range(2):
        for t_i, tissue in enumerate(TISSUES):
            loadings_rows.append({'component': f'PC{comp_i+1}', 'tissue': tissue,
                                   'method': 'PCA', 'label_scheme': '(unsupervised)',
                                   'loading': pca_2d.components_[comp_i, t_i]})
            for scheme_name, comps in loadings.items():
                loadings_rows.append({'component': f'PC{comp_i+1}', 'tissue': tissue,
                                       'method': 'PCovC (mixing=0.5)', 'label_scheme': scheme_name,
                                       'loading': comps[comp_i, t_i]})
    loadings_df = pd.DataFrame(loadings_rows)
    loadings_df.to_csv(os.path.join(OUT_DIR, 'pcovc_loadings.csv'), index=False)

    fig, axes = plt.subplots(ncols=2, figsize=[11, 4], sharey=True)
    for comp_i, ax in zip(range(2), axes):
        sub = loadings_df[loadings_df['component'] == f'PC{comp_i+1}'].copy()
        sub['series'] = sub['method'] + ' | ' + sub['label_scheme']
        sns.barplot(data=sub, x='tissue', y='loading', hue='series', ax=ax)
        ax.set_title(f'Component {comp_i+1} loadings')
        ax.axhline(0, color='black', lw=.5)
        ax.tick_params(axis='x', rotation=30)
        if comp_i == 0:
            ax.legend(fontsize=6, frameon=False)
        else:
            ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pcovc_loadings_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved loadings figure to {OUT_DIR}/pcovc_loadings_comparison.png")

    # --- CI comparison figure: silhouette and KNN accuracy, both label schemes ---
    fig, axes = plt.subplots(ncols=2, figsize=[9, 3.5])
    for ax, metric in zip(axes, ['silhouette', 'knn_acc']):
        plot_df = results_df.copy()
        plot_df['err_lo'] = plot_df[f'{metric}_mean'] - plot_df[f'{metric}_lo']
        plot_df['err_hi'] = plot_df[f'{metric}_hi'] - plot_df[f'{metric}_mean']
        for i, (_, row) in enumerate(plot_df.iterrows()):
            x = i
            ax.errorbar(x, row[f'{metric}_mean'],
                        yerr=[[row['err_lo']], [row['err_hi']]],
                        fmt='o', capsize=3,
                        color='tab:blue' if 'PCA' in row['method'] else 'tab:orange')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels([f"{r.method}\n{r.label_scheme}" for _, r in plot_df.iterrows()],
                            rotation=60, ha='right', fontsize=6)
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pcovc_uncertainty_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"Saved uncertainty figure to {OUT_DIR}/pcovc_uncertainty_comparison.png")


if __name__ == '__main__':
    main()
