"""
Tests whether PCovC's pxz_ (mutation feature -> cluster confidence weight)
depends on the mixing parameter, and specifically whether VR-region identity
appears among the top-ranked features only at low (classification-driven)
mixing, not at mixing=1.0 (which reduces PCovC to a classifier fit on a
plain PCA embedding).

Two checks:
1. Sweep mixing from 1.0 (pure PCA) to 0.0 (pure classification) and count
   how many of the top-8-ranked |pxz_| features (for a given class) are
   VR-region dummy variables, on both label schemes (the paper's tropism
   clusters and the independent tissue-of-max-deviation label).
2. Bootstrap check at the mixing=1.0 endpoint: across resamples, does a
   VR-region feature ever break into the top 8, or is its absence a stable
   property of the PCA-only embedding rather than a single point estimate?

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skmatter.decomposition import PCovC

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input, build_paper_labels
from pcovc_extended import build_tissue_labels
import pcovc_mutation_features_expanded as pmfe

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
RNG = 0
MIXINGS = [1.0, 0.75, 0.5, 0.25, 0.0]
N_BOOT = 300


def build_feature_matrix():
    X_tissue = build_mouse_pca_input()
    feat, delta_cols, reps = pmfe.build_expanded_mutation_features(X_tissue.index)
    numeric_cols = delta_cols
    dummy_cols = [c for c in feat.columns if c not in numeric_cols]
    Xs_df = feat.copy()
    Xs_df[numeric_cols] = StandardScaler().fit_transform(Xs_df[numeric_cols])
    Xs_df[dummy_cols] = Xs_df[dummy_cols].astype(float)
    feature_names = Xs_df.columns.tolist()
    return X_tissue, Xs_df.to_numpy(), feature_names


def top_features(Xs, y_arr, feature_names, mixing, target_class, n=8):
    m = PCovC(mixing=mixing, n_components=2, random_state=RNG, scale_z=True,
              classifier=LogisticRegression(max_iter=2000, class_weight='balanced')).fit(Xs, y_arr)
    idx = list(m.classes_).index(target_class)
    s = pd.Series(m.pxz_[:, idx], index=feature_names)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(n)


def n_vr_in_top(Xs, y_arr, feature_names, mixing, target_class, n=8):
    top = top_features(Xs, y_arr, feature_names, mixing, target_class, n=n)
    return sum(1 for f in top.index if f.startswith('vr_region'))


def main():
    X_tissue, Xs, feature_names = build_feature_matrix()
    _, y_paper = build_paper_labels(X_tissue)
    y_tissue = build_tissue_labels(X_tissue)

    label_schemes = {
        'paper_clusters': (y_paper.to_numpy(), 1),
        'independent_tissue': (y_tissue.to_numpy(), sorted(y_tissue.unique())[0]),
    }

    print("=" * 70)
    print("1. Mixing sweep: how many of the top-8 |pxz_| features are VR-region dummies?")
    print("=" * 70)
    rows = []
    for scheme_name, (y_arr, target_class) in label_schemes.items():
        for mixing in MIXINGS:
            n_vr = n_vr_in_top(Xs, y_arr, feature_names, mixing, target_class)
            rows.append(dict(label_scheme=scheme_name, mixing=mixing, n_vr_in_top8=n_vr))
            print(f"  [{scheme_name}] mixing={mixing:.2f}  VR-region features in top 8: {n_vr}")
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(OUT_DIR, 'mixing_sensitivity_sweep.csv'), index=False)

    print("\n" + "=" * 70)
    print("2. Bootstrap check at mixing=1.0 (pure PCA), paper_clusters scheme, class 1")
    print("=" * 70)
    y_arr, target_class = label_schemes['paper_clusters']
    n = len(y_arr)
    rng = np.random.default_rng(RNG)
    vr_idx = [i for i, f in enumerate(feature_names) if f.startswith('vr_region')]
    best_vr_rank = []
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y_b = y_arr[idx]
        if len(np.unique(y_b)) < len(np.unique(y_arr)):
            continue
        m = PCovC(mixing=1.0, n_components=2, random_state=RNG, scale_z=True,
                  classifier=LogisticRegression(max_iter=2000, class_weight='balanced')).fit(Xs[idx], y_b)
        cls_idx = list(m.classes_).index(target_class)
        ranks = pd.Series(np.abs(m.pxz_[:, cls_idx])).rank(ascending=False)
        best_vr_rank.append(ranks[vr_idx].min())
    best_vr_rank = np.array(best_vr_rank)
    frac_top8 = (best_vr_rank <= 8).mean()
    print(f"  {len(best_vr_rank)} usable resamples")
    print(f"  best VR-region feature rank: mean={best_vr_rank.mean():.1f}, median={np.median(best_vr_rank):.0f} "
          f"of {len(feature_names)} features")
    print(f"  fraction of resamples where a VR-region feature makes the top 8: {frac_top8:.2%}")
    pd.Series(best_vr_rank, name='best_vr_rank').to_csv(
        os.path.join(OUT_DIR, 'mixing_sensitivity_bootstrap_ranks.csv'), index=False)

    # --- figure ---
    fig, ax = plt.subplots(figsize=[7, 4.5])
    for scheme_name, marker in [('paper_clusters', 'o'), ('independent_tissue', 's')]:
        sub = sweep_df[sweep_df.label_scheme == scheme_name]
        ax.plot(sub.mixing, sub.n_vr_in_top8, marker=marker, label=scheme_name.replace('_', ' '))
    ax.set_xlabel('mixing (1.0 = pure PCA, 0.0 = pure classification)')
    ax.set_ylabel('VR-region features in top 8 by |pxz_|')
    ax.set_title('VR-region signal in pxz_, by mixing parameter')
    ax.legend(frameon=False)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mixing_sensitivity.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT_DIR}/mixing_sensitivity.png")


if __name__ == '__main__':
    main()
