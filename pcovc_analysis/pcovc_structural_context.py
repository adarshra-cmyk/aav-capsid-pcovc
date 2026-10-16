"""
Structural context for the tropism-cluster interpretability findings, using
data already in the original repo (no external fetch needed):
`data/meta/viperdb_contatcs..csv`, inter-subunit residue contacts across the
AAV2 capsid's icosahedral symmetry axes (2-fold, 3-fold, 5-fold interfaces),
loaded via the repo's own `common.load_axis_contacts()` (already used for
the original paper's structural discussion in a03).

Ogden et al. 2019 report (Fig 3E) that tropism clusters 1 and 2 mutations
sit in tight surface patches near the capsid's 3-fold axis, while cluster 3
mutations are dispersed in buried positions. This script tests that finding
independently, position by position, against:
  (a) our own PCovC-derived cluster assignments (paper_clusters scheme)
  (b) a continuous, cluster-free measure of tropism-switching strength
      (mean |log2 selection deviation| per position, across all 6 tissues)
and adds two further pieces of interpretability not yet in any figure:
  (c) a tissue-tissue correlation matrix (which organs behave similarly)
  (d) a linear-sequence "hotspot" track of tropism-switching strength by
      position, annotated with VR-region boundaries and axis membership

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, './pcovc_analysis')
import common as a04_common
from pcovc_biodistribution import build_mouse_pca_input, build_paper_labels
from pcovc_mutation_features import VR_BOUNDS, vr_region

OUT_DIR = './pcovc_analysis/deep_dive'
os.makedirs(OUT_DIR, exist_ok=True)
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
AXIS_COLS = {'A1-A6 (I-2)': '2-fold', 'A1-A7 (I-3)': '3-fold',
             'A1-A2 (I-5)': '5-fold', 'A1-A12 (I-5)': '5-fold'}


def build_position_axis_membership():
    """Returns {abs_pos: set of axis types ('2-fold','3-fold','5-fold') the
    position participates in via any inter-subunit contact}."""
    contacts = a04_common.load_axis_contacts()
    membership = {}
    for _, row in contacts.iterrows():
        for axis_col, axis_name in AXIS_COLS.items():
            if pd.notna(row[axis_col]):
                for pos in (row['res1_vp'], row['res2_vp']):
                    membership.setdefault(pos, set()).add(axis_name)
    return membership


def main():
    X_tissue = build_mouse_pca_input()
    _, y = build_paper_labels(X_tissue)
    axis_membership = build_position_axis_membership()

    # ── (a) + (b): per-position tropism-switching strength ──
    pos_index = X_tissue.index.get_level_values('abs_pos')
    per_pos_rows = []
    for pos in sorted(set(pos_index)):
        mask = pos_index == pos
        mean_abs_sel = X_tissue.loc[mask, TISSUES].abs().mean().mean()
        y_here = y.loc[mask]
        frac_named = (y_here != 0).mean()
        cluster_counts = y_here.value_counts()
        axes = axis_membership.get(pos, set())
        per_pos_rows.append(dict(
            abs_pos=pos, vr_region=vr_region(pos),
            mean_abs_selection=mean_abs_sel, frac_named_cluster=frac_named,
            n_mutants=mask.sum(),
            axis_2fold='2-fold' in axes, axis_3fold='3-fold' in axes, axis_5fold='5-fold' in axes,
            any_axis=len(axes) > 0,
            dominant_cluster=cluster_counts.index[0] if len(cluster_counts) else np.nan,
        ))
    per_pos = pd.DataFrame(per_pos_rows)
    per_pos.to_csv(os.path.join(OUT_DIR, 'per_position_structural_context.csv'), index=False)

    print(f"Positions analyzed: {len(per_pos)}, of which {per_pos.any_axis.sum()} "
          f"participate in a resolved inter-subunit contact (VIPERdb)")
    print(per_pos.groupby(['axis_2fold', 'axis_3fold', 'axis_5fold']).size())

    # --- test: does axis proximity predict tropism-switching strength? ---
    print("\nMean |log2 selection| by axis membership (independent of any clustering):")
    for axis in ['axis_2fold', 'axis_3fold', 'axis_5fold']:
        on = per_pos.loc[per_pos[axis], 'mean_abs_selection']
        off = per_pos.loc[~per_pos[axis], 'mean_abs_selection']
        u, p = stats.mannwhitneyu(on, off, alternative='two-sided')
        print(f"  {axis:12s}  on-axis mean={on.mean():.3f} (n={len(on)})  "
              f"off-axis mean={off.mean():.3f} (n={len(off)})  Mann-Whitney p={p:.4f}")

    print("\nFraction of mutations assigned to a NAMED tropism cluster (not 'other'), by axis:")
    for axis in ['axis_2fold', 'axis_3fold', 'axis_5fold']:
        on = per_pos.loc[per_pos[axis], 'frac_named_cluster']
        off = per_pos.loc[~per_pos[axis], 'frac_named_cluster']
        u, p = stats.mannwhitneyu(on, off, alternative='two-sided')
        print(f"  {axis:12s}  on-axis mean={on.mean():.3f} (n={len(on)})  "
              f"off-axis mean={off.mean():.3f} (n={len(off)})  Mann-Whitney p={p:.4f}")

    # ── (c) tissue-tissue correlation matrix ──
    tissue_corr = X_tissue[TISSUES].corr(method='spearman')
    tissue_corr.to_csv(os.path.join(OUT_DIR, 'tissue_tissue_correlation.csv'))
    print("\nTissue-tissue Spearman correlation (log2 selection, across all 2,007 mutants):")
    print(tissue_corr.round(2).to_string())

    # ── figures ──
    fig, axes = plt.subplots(ncols=2, figsize=[13, 5.2])
    sns.heatmap(tissue_corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', ax=axes[0])
    axes[0].set_title('Tissue-tissue Spearman correlation\n(log2 selection, n=2,007 mutants)')

    axis_summary = per_pos.melt(id_vars=['mean_abs_selection', 'frac_named_cluster'],
                                 value_vars=['axis_2fold', 'axis_3fold', 'axis_5fold'],
                                 var_name='axis', value_name='on_axis')
    axis_summary['axis'] = axis_summary['axis'].str.replace('axis_', '').str.replace('fold', '-fold')
    sns.barplot(data=axis_summary, x='axis', y='mean_abs_selection', hue='on_axis', ax=axes[1],
                palette={True: '#DD8452', False: '#B0B0B0'})
    axes[1].set_title('Mean |log2 selection| by symmetry-axis membership')
    axes[1].set_ylabel('mean |log2 selection| across 6 tissues')
    axes[1].legend(title='on axis?', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'structural_context_summary.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT_DIR}/structural_context_summary.png")

    # linear-sequence hotspot track
    fig2, ax = plt.subplots(figsize=[14, 4])
    ax.plot(per_pos.abs_pos, per_pos.mean_abs_selection, color='black', lw=0.8, alpha=0.7)
    for name, (lo, hi) in VR_BOUNDS.items():
        ax.axvspan(lo, hi, color='tab:orange', alpha=0.25)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95, name, ha='center', fontsize=7, rotation=90, va='top')
    for axis_col, color in [('axis_3fold', 'tab:red'), ('axis_2fold', 'tab:blue'), ('axis_5fold', 'tab:green')]:
        hits = per_pos.loc[per_pos[axis_col], 'abs_pos']
        ax.scatter(hits, [ax.get_ylim()[1] * 1.02] * len(hits), marker='|', color=color, s=40,
                   label=axis_col.replace('axis_', '').replace('fold', '-fold'))
    ax.set_xlabel('capsid position (VP1 numbering)')
    ax.set_ylabel('mean |log2 selection|\n(tropism-switching strength)')
    ax.set_title('Tropism-switching strength by capsid position, VR regions (orange) '
                 'and symmetry-axis contacts (tick marks)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'position_hotspot_track.png'), dpi=200, bbox_inches='tight')
    print(f"Saved {OUT_DIR}/position_hotspot_track.png")


if __name__ == '__main__':
    main()
