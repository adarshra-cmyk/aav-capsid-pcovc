"""
Which mutation features predict tropism-cluster membership?

The main pcovc_biodistribution.py comparison uses the 6-tissue selection
profile as X and the paper's tropism clusters as y -- that's circular for
interpretability (the clusters *are* tissue profiles, so "which tissue
predicts the cluster" is trivial). This script reframes the problem: X is
now a set of per-mutation features (which variable region it falls in,
insertion vs substitution, and the biochemical property change from WT),
independent of any tissue-selection measurement, and y is still the paper's
3 tropism clusters. PCovC's pxz_ (and the underlying linear classifier's
coefficients) then directly answer "which mutation feature predicts which
organ-tropism cluster."

Run with cwd = <repo>/x02_analysis, using venv_pcovc (Python >= 3.10).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from skmatter.decomposition import PCovC

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input, build_paper_labels

OUT_DIR = './pcovc_analysis'
RNG = 0

# variable-region boundaries, amino-acid coordinates (a01_analyze_packaging_data.py)
VR_BOUNDS = {
    'VR-I': (262, 269), 'VR-II': (326, 331), 'VR-IV': (449, 469),
    'VR-V': (487, 504), 'VR-VI': (525, 540), 'VR-VII': (544, 556),
    'VR-VIII': (579, 594), 'VR-IX': (704, 711),
}

# AA_INDEX_DF (data/meta/aa_index_0_center_norm_df.csv) is unlabeled by row;
# confirmed by correlating its Molecular_weight column against real residue
# masses (spearman r=1.0) to be BLOSUM/PAM row order.
AA_ORDER = list('ARNDCQEGHILKMFPSTWYV')
PROP_COLS = ['Hydrophobicity', 'Residue_volume', 'Polarity']


def vr_region(pos):
    for name, (lo, hi) in VR_BOUNDS.items():
        if lo <= pos <= hi:
            return name
    return 'none'


def build_mutation_features(X_index, lookup_path='../data/meta/AAV2scan_chip_lookup_table.txt',
                             aa_index_path='../data/meta/aa_index_0_center_norm_df.csv'):
    lookup = pd.read_csv(lookup_path)
    wt_by_pos = lookup[lookup['is_wt_aa'] == 1].set_index('abs_pos')['aa'].to_dict()

    aa_index = pd.read_csv(aa_index_path, index_col=0)
    aa_index.index = AA_ORDER
    props = aa_index[PROP_COLS]
    net_charge = aa_index['Positive_charge'] - aa_index['Negative_charge']
    props = props.assign(net_charge=net_charge)

    rows = []
    for abs_pos, aa, wt_bc, lib_type in X_index:
        wt_aa = wt_by_pos.get(abs_pos)
        row = {
            'abs_pos': abs_pos,
            'vr_region': vr_region(abs_pos),
            'lib_type': lib_type,
            'is_stop': aa == '*',
        }
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
    feat['is_stop'] = feat['is_stop'].astype(float)

    feat = pd.get_dummies(feat, columns=['vr_region', 'lib_type'], drop_first=False)
    feat = feat.drop(columns=['abs_pos'])
    return feat, delta_cols


def main():
    X_tissue = build_mouse_pca_input()
    _, y = build_paper_labels(X_tissue)

    feat, delta_cols = build_mutation_features(X_tissue.index)
    numeric_cols = delta_cols + ['is_stop']
    dummy_cols = [c for c in feat.columns if c not in numeric_cols]

    Xs = feat.copy()
    Xs[numeric_cols] = StandardScaler().fit_transform(Xs[numeric_cols])
    Xs[dummy_cols] = Xs[dummy_cols].astype(float)
    feature_names = Xs.columns.tolist()
    Xs = Xs.to_numpy()
    y_arr = y.to_numpy()

    print(f"X (mutation features): {Xs.shape}, features: {feature_names}")
    print(f"class counts:\n{y.value_counts().sort_index()}")

    # how predictable is tropism cluster from mutation identity alone, with
    # no tissue-selection data at all? Classes are heavily imbalanced (82%
    # "other"), so plain accuracy rewards always predicting the majority
    # class -- use balanced accuracy (macro-averaged per-class recall) and
    # class_weight='balanced' instead.
    clf_cv = LogisticRegression(max_iter=2000, class_weight='balanced')
    cv_bal_acc = cross_val_score(clf_cv, Xs, y_arr, cv=5, scoring='balanced_accuracy').mean()
    print(f"\n5-fold CV balanced accuracy: {cv_bal_acc:.3f}  (chance level for 4 classes: 0.250)")

    y_pred = cross_val_predict(clf_cv, Xs, y_arr, cv=5)
    cm = confusion_matrix(y_arr, y_pred, labels=sorted(y.unique()))
    cm_df = pd.DataFrame(cm, index=[f'true {c}' for c in sorted(y.unique())],
                          columns=[f'pred {c}' for c in sorted(y.unique())])
    print("\nconfusion matrix (5-fold CV predictions):")
    print(cm_df)
    print(f"per-class recall: {(cm.diagonal() / cm.sum(axis=1)).round(3)}")

    # PCovC: mixing=0.5, classifier default (LogisticRegression, balanced).
    # pxz_ is the direct X -> class-confidence weight matrix.
    model = PCovC(mixing=0.5, n_components=2, random_state=RNG, scale_z=True,
                   classifier=LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(Xs, y_arr)

    pxz = pd.DataFrame(model.pxz_, index=feature_names, columns=model.classes_)
    pxz.to_csv(os.path.join(OUT_DIR, 'pcovc_mutation_pxz.csv'))
    print("\npxz_ (feature -> class-confidence weight):")
    print(pxz.round(3))

    # plot: top +/- features per named cluster (skip 'other')
    named_clusters = [c for c in model.classes_ if c != 0]
    fig, axes = plt.subplots(ncols=len(named_clusters), figsize=[5 * len(named_clusters), 5], sharex=False)
    for ax, cl in zip(axes, named_clusters):
        s = pxz[cl].sort_values()
        top = pd.concat([s.head(6), s.tail(6)])
        colors = ['#4C72B0' if v < 0 else '#DD8452' for v in top]
        ax.barh(top.index, top.values, color=colors)
        ax.axvline(0, color='black', lw=.6)
        ax.set_title(f'Cluster {cl}')
        ax.set_xlabel('pxz_ weight')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pcovc_mutation_pxz.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {OUT_DIR}/pcovc_mutation_pxz.png")


if __name__ == '__main__':
    main()
