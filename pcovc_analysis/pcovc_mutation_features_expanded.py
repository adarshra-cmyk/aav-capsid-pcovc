"""
Pushes pcovc_mutation_features.py (PCovC's pxz_ mutation-to-tropism-cluster
weights) further, in the same two directions used for the organ-correlation
analysis:

1. Expanded, de-redundant property panel: the original script used 4
   hand-picked AAindex properties (Hydrophobicity, Residue_volume, Polarity,
   net_charge). This reuses the same 12-cluster, correlation-de-redundant
   AAindex selection from pcovc_mutation_organ_correlation_expanded.py (plus
   net_charge) so the mutation -> cluster interpretability story is built
   from the same expanded feature set as the mutation -> organ story,
   letting the two be compared directly.

2. Bootstrap stability: pxz_ is a point estimate from a single fit. With the
   paper-cluster labels 82% "other," it's not obvious which top +/- features
   are robust signal versus noise from resampling. This refits PCovC on
   1000 bootstrap resamples of the mutants and reports each feature's pxz_
   weight as a mean + 95% percentile interval per cluster, and flags which
   top features have intervals that exclude 0 (i.e. are stable, not noise).

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
from pcovc_mutation_features import vr_region, AA_ORDER
from pcovc_mutation_organ_correlation_expanded import select_low_redundancy_properties, N_CLUSTERS

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
RNG = 0
N_BOOT = 1000


def build_expanded_mutation_features(X_index):
    aa_index = pd.read_csv('../data/meta/aa_index_0_center_norm_df.csv', index_col=0)
    aa_index.index = AA_ORDER
    reps, _ = select_low_redundancy_properties(aa_index, n_clusters=N_CLUSTERS)
    net_charge = aa_index['Positive_charge'] - aa_index['Negative_charge']
    props = aa_index[reps].copy()
    props['net_charge'] = net_charge

    lookup = pd.read_csv('../data/meta/AAV2scan_chip_lookup_table.txt')
    wt_by_pos = lookup[lookup['is_wt_aa'] == 1].set_index('abs_pos')['aa'].to_dict()

    rows = []
    for abs_pos, aa, wt_bc, lib_type in X_index:
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
    return feat, delta_cols, reps


def main():
    X_tissue = build_mouse_pca_input()
    _, y = build_paper_labels(X_tissue)

    feat, delta_cols, reps = build_expanded_mutation_features(X_tissue.index)
    numeric_cols = delta_cols
    dummy_cols = [c for c in feat.columns if c not in numeric_cols]

    Xs_df = feat.copy()
    Xs_df[numeric_cols] = StandardScaler().fit_transform(Xs_df[numeric_cols])
    Xs_df[dummy_cols] = Xs_df[dummy_cols].astype(float)
    feature_names = Xs_df.columns.tolist()
    Xs = Xs_df.to_numpy()
    y_arr = y.to_numpy()
    classes = None

    print(f"Expanded mutation feature set: {len(feature_names)} features "
          f"({len(numeric_cols)} numeric incl. {len(reps)} de-redundant AAindex properties "
          f"+ net_charge, {len(dummy_cols)} categorical)")
    print(f"class counts:\n{y.value_counts().sort_index()}")

    # --- point estimate (full data) ---
    model = PCovC(mixing=0.5, n_components=2, random_state=RNG, scale_z=True,
                   classifier=LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(Xs, y_arr)
    classes = model.classes_
    pxz_point = pd.DataFrame(model.pxz_, index=feature_names, columns=classes)

    # --- bootstrap ---
    rng = np.random.default_rng(RNG)
    n = len(y_arr)
    boot_weights = {c: [] for c in classes}
    n_ok = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y_b = y_arr[idx]
        if len(np.unique(y_b)) < len(classes):
            continue  # skip resamples that drop a class entirely
        Xs_b = Xs[idx]
        try:
            m_b = PCovC(mixing=0.5, n_components=2, random_state=RNG, scale_z=True,
                        classifier=LogisticRegression(max_iter=2000, class_weight='balanced'))
            m_b.fit(Xs_b, y_b)
        except Exception:
            continue
        if list(m_b.classes_) != list(classes):
            continue  # class order mismatch, skip rather than misalign columns
        for i, c in enumerate(classes):
            boot_weights[c].append(m_b.pxz_[:, i] if m_b.pxz_.ndim > 1 else m_b.pxz_)
        n_ok += 1
    print(f"\nBootstrap: {n_ok}/{N_BOOT} resamples usable (rest skipped, missing a class)")

    # --- summarize: mean, 95% CI, stability flag per feature x class ---
    summary_rows = []
    for c in classes:
        arr = np.array(boot_weights[c])  # (n_ok, n_features)
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        stable = (lo > 0) | (hi < 0)  # CI excludes zero
        for feat_name, m, l, h, st, pt in zip(feature_names, mean, lo, hi, stable, pxz_point[c]):
            summary_rows.append(dict(cluster=c, feature=feat_name, point_estimate=pt,
                                      boot_mean=m, boot_lo=l, boot_hi=h, stable=st))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, 'expanded_pxz_bootstrap.csv'), index=False)

    print("\nStable (bootstrap CI excludes 0) features per named cluster, ranked by |mean|:")
    named_clusters = [c for c in classes if c != 0]
    for c in named_clusters:
        sub = summary[(summary.cluster == c) & (summary.stable)].copy()
        sub = sub.reindex(sub.boot_mean.abs().sort_values(ascending=False).index)
        print(f"\n  Cluster {c} ({len(sub)} stable features):")
        print(sub[['feature', 'boot_mean', 'boot_lo', 'boot_hi']].head(8).to_string(index=False))

    # --- figure: top stable +/- features per cluster, with bootstrap CI ---
    def _short_label(name, max_len=26):
        name = name.replace('_delta', '').replace('_', ' ')
        return name if len(name) <= max_len else name[:max_len - 1] + '…'

    # Stack panels vertically instead of side-by-side: the AAindex property
    # names are too long (up to ~80 chars) for 3 narrow side-by-side panels
    # to lay out without matplotlib's tight_layout collapsing the plot area
    # to near-zero width (confirmed by isolating a single panel, which
    # renders fine on its own at normal width).
    fig, axes = plt.subplots(nrows=len(named_clusters), figsize=[8, 3.6 * len(named_clusters)],
                              constrained_layout=True)
    if len(named_clusters) == 1:
        axes = [axes]
    for ax, c in zip(axes, named_clusters):
        sub = summary[summary.cluster == c].copy()
        sub = sub.reindex(sub.boot_mean.abs().sort_values(ascending=False).index).head(10)
        sub = sub.iloc[::-1]
        labels = [_short_label(f) for f in sub.feature]
        colors = []
        for m, st in zip(sub.boot_mean, sub.stable):
            if not st:
                colors.append('#B0B0B0')
            elif m < 0:
                colors.append('#4C72B0')
            else:
                colors.append('#DD8452')
        xerr = np.array([sub.boot_mean - sub.boot_lo, sub.boot_hi - sub.boot_mean])
        ax.barh(labels, sub.boot_mean, xerr=xerr, color=colors, capsize=2, ecolor='black', error_kw={'linewidth': 0.8})
        ax.axvline(0, color='black', lw=.6)
        ax.set_title(f'Cluster {c}  (gray = bootstrap CI includes 0, i.e. unstable)', fontsize=10)
        ax.set_xlabel('pxz_ weight (bootstrap mean +/- 95% CI)', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
    fig.savefig(os.path.join(OUT_DIR, 'expanded_pxz_bootstrap.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {OUT_DIR}/expanded_pxz_bootstrap.png")


if __name__ == '__main__':
    main()
