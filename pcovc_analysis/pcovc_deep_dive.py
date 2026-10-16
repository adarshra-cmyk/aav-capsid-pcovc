"""
pcovc_deep_dive.py -- pushes the PCovC-vs-PCA comparison well past
pcovc_biodistribution.py / pcovc_extended.py, for a standalone "Part 2"
deep-dive slide deck. Five extensions:

1. NESTED CV FIX (the most important one). pcovc_extended.py's evaluate()
   scores K-NN accuracy on an embedding produced by
   `pcovc.fit_transform(Xs, y)` using ALL labels, then cross-validates KNN
   *on that fixed embedding*. But PCovC's own embedding-fitting step already
   used every fold's labels (including the ones later held out for KNN
   scoring) -- so every previously-reported PCovC KNN accuracy had already
   leaked test-fold label information through the embedding itself before
   the "held-out" evaluation even started. PCA is unaffected (it never
   looks at y), so this made the PCovC-vs-PCA comparison specifically
   biased in PCovC's favor, on top of the (already-flagged) label-circularity
   issue. This script refits PCovC from scratch on each training fold only,
   and scores by transforming the untouched held-out fold -- true nested CV.
   This is a real instance of exactly the "double dipping" pattern described
   in Kriegeskorte et al. 2009 (Nat Neurosci, circular analysis) and the
   PLS-DA overoptimism literature (Westerhuis 2008; Szymanska et al. 2012,
   Metabolomics; "Overoptimism in cross-validation...", Anal Bioanal Chem
   2018) -- PCovC is structurally a supervised-projection method in the same
   family as PLS-DA, and that literature's central warning (label-aware
   projections look good on the data used to fit them; only genuinely
   held-out evaluation is trustworthy) applies directly here.

2. A full mixing-parameter sweep (0.0 to 1.0), not just the 3 cherry-picked
   values used previously, under the corrected nested-CV protocol.

3. A label-permutation test: shuffle y (many times), refit PCovC + KNN under
   nested CV each time, and compare the real-label result against this null
   distribution directly -- a genuine significance test, not eyeballed CI
   overlap. This is the standard fix in the selective-inference / "testing
   after clustering" literature (Gao, Bien & Witten 2022 JASA; SigClust,
   Liu et al.) for exactly this kind of "is the apparent structure real"
   question.

4. Classifier-choice sensitivity: PCovC's internal Z (class confidence)
   classifier defaults to LogisticRegression, but skmatter also supports
   LDA, LinearSVC, RidgeClassifier, Perceptron -- does the "modest gain"
   story hold up across all of them, or is it an artifact of one specific
   choice?

5. A decision-boundary visualization using PCovC's own fitted classifier_ on
   the 2D embedding -- previously only external KNN accuracy was reported;
   the method's own internal decision boundary was never actually shown.

Run with cwd = <repo>/x02_analysis, using venv_pcovc (Python >= 3.10)
-- but this script only needs numpy/pandas/sklearn/skmatter, no repo-internal
imports beyond pcovc_biodistribution's data loaders, so it also runs fine
from any cwd as long as PYTHONPATH/sys.path includes x02_analysis (handled
below) and pcovc_analysis/.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from skmatter.decomposition import PCovC

REPO_X02 = './repo/x02_analysis'
if REPO_X02 not in sys.path:
    sys.path.insert(0, REPO_X02)
sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input, build_paper_labels
from pcovc_extended import build_tissue_labels

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
RNG = 0
N_SPLITS = 5
N_REPEATS = 20  # repeated stratified CV -> 100 fold estimates, same as pcovc_extended.py
KNN_K = 15


# ─────────────────────────────────────────────────────────────────────────────
# 1. NESTED CV: refit the embedding per fold, not once on everything
# ─────────────────────────────────────────────────────────────────────────────
def _repeated_stratified_splits(y, n_splits, n_repeats, seed):
    """Manual repeated stratified K-fold (avoids importing RepeatedStratifiedKFold
    just to get the same splits every time we need them across many callers)."""
    rng = np.random.default_rng(seed)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            yield train_idx, test_idx


def nested_cv_scores(Xs, y, embedder_fn, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                      seed=RNG, knn_k=KNN_K):
    """
    embedder_fn(X_train, y_train) -> fitted object with .transform(X) (or a
    plain callable transform for PCA/PCovC alike). Refits per fold; KNN
    accuracy is scored ONLY on the held-out fold's transform, never seen
    during the embedder's own fit. Also returns held-out-fold silhouette
    (using the true held-out labels against the held-out embedding -- this
    metric was already computed on out-of-model-fit data implicitly for PCA
    in pcovc_extended.py, since PCA doesn't use y, but for PCovC it was not;
    this makes both methods' silhouette genuinely nested too).
    Returns (knn_acc_scores, silhouette_scores) arrays, one entry per fold.
    """
    knn_scores, sil_scores = [], []
    for train_idx, test_idx in _repeated_stratified_splits(y, n_splits, n_repeats, seed):
        X_tr, X_te = Xs[train_idx], Xs[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = embedder_fn(X_tr, y_tr)
        emb_tr = model.transform(X_tr)
        emb_te = model.transform(X_te)

        knn = KNeighborsClassifier(n_neighbors=knn_k).fit(emb_tr, y_tr)
        knn_scores.append(knn.score(emb_te, y_te))

        if len(np.unique(y_te)) >= 2 and len(y_te) > knn_k:
            sil_scores.append(silhouette_score(emb_te, y_te))

    return np.array(knn_scores), np.array(sil_scores)


def summarize(arr):
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    return (arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5))


def make_pca_embedder():
    def fit(X_tr, y_tr):
        return PCA(n_components=2, random_state=RNG).fit(X_tr)
    return fit


def make_pcovc_embedder(mixing=0.5, classifier=None):
    def fit(X_tr, y_tr):
        return PCovC(mixing=mixing, n_components=2, random_state=RNG,
                     scale_z=True, classifier=classifier).fit(X_tr, y_tr)
    return fit


# ─────────────────────────────────────────────────────────────────────────────
# 2 + 3. Mixing sweep and permutation test, both under nested CV
# ─────────────────────────────────────────────────────────────────────────────
def run_mixing_sweep(Xs, label_schemes, mixings=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    rows = []
    for scheme_name, y in label_schemes.items():
        t0 = time.time()
        pca_knn, pca_sil = nested_cv_scores(Xs, y, make_pca_embedder())
        rows.append(dict(label_scheme=scheme_name, method='PCA', mixing=np.nan,
                          knn_mean=summarize(pca_knn)[0], knn_lo=summarize(pca_knn)[1], knn_hi=summarize(pca_knn)[2],
                          sil_mean=summarize(pca_sil)[0], sil_lo=summarize(pca_sil)[1], sil_hi=summarize(pca_sil)[2]))
        for m in mixings:
            knn, sil = nested_cv_scores(Xs, y, make_pcovc_embedder(mixing=m))
            rows.append(dict(label_scheme=scheme_name, method='PCovC', mixing=m,
                              knn_mean=summarize(knn)[0], knn_lo=summarize(knn)[1], knn_hi=summarize(knn)[2],
                              sil_mean=summarize(sil)[0], sil_lo=summarize(sil)[1], sil_hi=summarize(sil)[2]))
        print(f"  [{scheme_name}] mixing sweep done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'mixing_sweep_nested_cv.csv'), index=False)
    return df


def run_permutation_test(Xs, label_schemes, mixing=0.5, n_perm=500, seed=RNG):
    """
    Null distribution: shuffle y (breaking any real X-y relationship, but
    preserving class balance), refit PCovC + nested-CV KNN each time. Compare
    the observed (real-label) result's rank within this null.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for scheme_name, y in label_schemes.items():
        t0 = time.time()
        obs_knn, _ = nested_cv_scores(Xs, y, make_pcovc_embedder(mixing=mixing))
        obs_mean = obs_knn.mean()

        null_means = np.empty(n_perm)
        for i in range(n_perm):
            y_perm = rng.permutation(y)
            knn_perm, _ = nested_cv_scores(Xs, y_perm, make_pcovc_embedder(mixing=mixing),
                                            n_repeats=2)  # fewer repeats per permutation draw; n_perm draws already gives the null its power
            null_means[i] = knn_perm.mean()

        p_value = (np.sum(null_means >= obs_mean) + 1) / (n_perm + 1)
        rows.append(dict(label_scheme=scheme_name, observed_knn_mean=obs_mean,
                          null_mean=null_means.mean(), null_std=null_means.std(),
                          null_p95=np.percentile(null_means, 95), p_value=p_value))
        np.save(os.path.join(OUT_DIR, f'null_dist_{scheme_name.split()[0]}.npy'), null_means)
        print(f"  [{scheme_name}] permutation test done in {time.time()-t0:.1f}s, p={p_value:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'permutation_test.csv'), index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Classifier-choice sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def run_classifier_sensitivity(Xs, label_schemes, mixing=0.5):
    classifiers = {
        'LogisticRegression (default)': LogisticRegression(max_iter=2000),
        'LDA': LinearDiscriminantAnalysis(),
        'LinearSVC': LinearSVC(max_iter=5000),
        'RidgeClassifier': RidgeClassifier(),
    }
    rows = []
    for scheme_name, y in label_schemes.items():
        for clf_name, clf in classifiers.items():
            knn, sil = nested_cv_scores(Xs, y, make_pcovc_embedder(mixing=mixing, classifier=clf))
            rows.append(dict(label_scheme=scheme_name, classifier=clf_name,
                              knn_mean=summarize(knn)[0], knn_lo=summarize(knn)[1], knn_hi=summarize(knn)[2],
                              sil_mean=summarize(sil)[0], sil_lo=summarize(sil)[1], sil_hi=summarize(sil)[2]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'classifier_sensitivity.csv'), index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decision-boundary visualization (PCovC's own classifier_, not external KNN)
# ─────────────────────────────────────────────────────────────────────────────
def plot_decision_boundary(Xs, y, mixing=0.5, label_scheme_name='', fname='decision_boundary.png'):
    model = PCovC(mixing=mixing, n_components=2, random_state=RNG, scale_z=True).fit(Xs, y)
    emb = model.transform(Xs)

    x_min, x_max = emb[:, 0].min() - 1, emb[:, 0].max() + 1
    y_min, y_max = emb[:, 1].min() - 1, emb[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds_raw = model.predict(T=grid)

    # classes can be non-numeric (e.g. tissue-name strings for the
    # independent label scheme), which contourf can't plot directly --
    # encode to integer codes for the contour, keep original labels for
    # the legend.
    classes = sorted(np.unique(y))
    class_to_code = {c: i for i, c in enumerate(classes)}
    preds = np.vectorize(class_to_code.get)(preds_raw).reshape(xx.shape)

    palette = sns.color_palette('deep', max(len(classes), 3))
    cmap = mcolors.ListedColormap(palette[:len(classes)])
    fig, ax = plt.subplots(figsize=[6, 5.2])
    ax.contourf(xx, yy, preds, levels=np.arange(len(classes) + 1) - 0.5, cmap=cmap, alpha=0.25)
    for i, c in enumerate(classes):
        ax.scatter(emb[y == c, 0], emb[y == c, 1], s=6, lw=0, alpha=0.85,
                   color=cmap.colors[i], label=f'{c}')
    ax.set_title(f'PCovC (mixing={mixing}) decision regions\n{label_scheme_name}', fontsize=10)
    ax.set_xlabel('component 1'); ax.set_ylabel('component 2')
    ax.legend(fontsize=7, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    X = build_mouse_pca_input()
    pca_df, y_paper = build_paper_labels(X)
    y_tissue = build_tissue_labels(X)
    Xs = StandardScaler().fit_transform(X.to_numpy())

    label_schemes = {'paper_clusters (PCA+Kmeans-derived)': y_paper.to_numpy(),
                      'tissue_of_max_deviation (independent)': y_tissue.to_numpy()}

    print("=" * 70)
    print("1. NESTED CV (corrected) at mixing=0.5, vs PCA -- apples to apples")
    print("=" * 70)
    nested_rows = []
    for scheme_name, y in label_schemes.items():
        pca_knn, pca_sil = nested_cv_scores(Xs, y, make_pca_embedder())
        pcovc_knn, pcovc_sil = nested_cv_scores(Xs, y, make_pcovc_embedder(mixing=0.5))
        for method, knn, sil in [('PCA', pca_knn, pca_sil), ('PCovC (mixing=0.5)', pcovc_knn, pcovc_sil)]:
            km, klo, khi = summarize(knn)
            sm, slo, shi = summarize(sil)
            nested_rows.append(dict(label_scheme=scheme_name, method=method,
                                     knn_mean=km, knn_lo=klo, knn_hi=khi,
                                     sil_mean=sm, sil_lo=slo, sil_hi=shi))
            print(f"  {scheme_name:42s} {method:20s} KNN={km:.4f} [{klo:.4f},{khi:.4f}]  "
                  f"sil={sm:.4f} [{slo:.4f},{shi:.4f}]")
    pd.DataFrame(nested_rows).to_csv(os.path.join(OUT_DIR, 'nested_cv_headline.csv'), index=False)

    print("\n" + "=" * 70)
    print("2. Mixing-parameter sweep (nested CV)")
    print("=" * 70)
    sweep_df = run_mixing_sweep(Xs, label_schemes)
    print(sweep_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("3. Label-permutation test (nested CV inside each permutation draw)")
    print("=" * 70)
    perm_df = run_permutation_test(Xs, label_schemes, n_perm=500)
    print(perm_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("4. Classifier-choice sensitivity (nested CV)")
    print("=" * 70)
    clf_df = run_classifier_sensitivity(Xs, label_schemes)
    print(clf_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("5. Decision-boundary plots")
    print("=" * 70)
    for scheme_name, y in label_schemes.items():
        fname = f"decision_boundary_{scheme_name.split()[0]}.png"
        plot_decision_boundary(Xs, y, mixing=0.5, label_scheme_name=scheme_name, fname=fname)
        print(f"  saved {fname}")

    # --- summary figures ---
    fig, axes = plt.subplots(ncols=2, figsize=[12, 4.5], sharex=True)
    for ax, scheme_name in zip(axes, label_schemes):
        sub = sweep_df[sweep_df.label_scheme == scheme_name]
        pcovc_sub = sub[sub.method == 'PCovC']
        pca_row = sub[sub.method == 'PCA'].iloc[0]
        ax.plot(pcovc_sub.mixing, pcovc_sub.knn_mean, 'o-', color='tab:orange', label='PCovC')
        ax.fill_between(pcovc_sub.mixing, pcovc_sub.knn_lo, pcovc_sub.knn_hi, color='tab:orange', alpha=0.2)
        ax.axhline(pca_row.knn_mean, color='tab:blue', ls='--', label='PCA (mixing-independent)')
        ax.axhspan(pca_row.knn_lo, pca_row.knn_hi, color='tab:blue', alpha=0.15)
        ax.set_xlabel('mixing (1=pure PCA, 0=pure classification)')
        ax.set_ylabel('nested-CV KNN accuracy')
        ax.set_title(scheme_name, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mixing_sweep_summary.png'), dpi=200, bbox_inches='tight')

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == '__main__':
    main()
