"""
Extends pcovc_residue_network_centrality.py's deletion-only betweenness
analysis to the full ~2,007-mutation dataset (subs/ins/dels), and directly
tests whether the centrality-fitness relationship is deletion-SPECIFIC
(interaction model) rather than just reporting three side-by-side
correlations.

Motivation (Adarsh, 2026-09-03): the deletion-only result (rho=0.297, 95%
CI [-0.058, 0.553], p=0.075, n=37) was suggestive but underpowered. The
open scientific question is whether "high-centrality residues are more
sensitive to mutation" holds generally (any mutation type) or is actually
specific to deletions (consistent with the original hypothesis that
deletions disrupt structure more directly than point substitutions).

Reuses the already-computed per-position betweenness centrality
(residue_network_per_position_centrality.csv) -- centrality is a static
structural property of a sequence position, independent of which
mutations are later correlated against it, so no network recomputation is
needed here.

Method:
  1. Per-lib_type (sub/ins/del) Spearman correlation, betweenness vs
     |log2 selection|, mean-across-tissues + all 6 tissues individually,
     bootstrap CI (matches pcovc_residue_network_centrality.py's
     convention).
  2. The actual test of deletion-specificity: an OLS interaction model,
     |log2 selection| ~ log(centrality)_z * lib_type (sub = reference
     category, since it's the largest group), bootstrapped over mutants
     (matches pcovc_organ_ols_bootstrap.py's row-resample convention). The
     del:log_centrality and ins:log_centrality interaction coefficients
     directly answer "does the centrality slope for deletions differ from
     substitutions' slope", which three separate correlations cannot.
     Fit both on the cross-tissue mean and per-tissue for robustness.

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input

OUT_DIR = './pcovc_analysis/deep_dive'
PER_POS_CSV = os.path.join(OUT_DIR, 'residue_network_per_position_centrality.csv')
os.makedirs(OUT_DIR, exist_ok=True)

TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
N_BOOT = 1000
RNG = 0
LOG_EPS = 1e-6  # 6/519 positions have exactly 0 betweenness (k=300-pivot sampling noise
                 # at the true periphery); this eps is ~1000x smaller than the median
                 # nonzero value (3.6e-4), so it separates true-zero/near-zero positions
                 # from the bulk without letting them dominate the log scale.


def spearman_boot(x, y, n_boot=N_BOOT, rng_seed=RNG):
    x, y = np.asarray(x), np.asarray(y)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, p, lo, hi, len(boots)


def make_lookup(cent_by_pos):
    """Substitutions/deletions sit at integer abs_pos (direct lookup).
    Insertions sit at half-integer abs_pos (e.g. 21.5 = inserted between
    residues 21 and 22) -- map to the mean centrality of the two flanking
    residues when both are structurally resolved, else NaN (matches
    pcovc_residue_network_centrality.py's handling)."""
    def lookup(pos):
        if float(pos).is_integer():
            return cent_by_pos.get(pos, np.nan)
        lo, hi = np.floor(pos), np.ceil(pos)
        c_lo, c_hi = cent_by_pos.get(lo, np.nan), cent_by_pos.get(hi, np.nan)
        if np.isnan(c_lo) or np.isnan(c_hi):
            return np.nan
        return (c_lo + c_hi) / 2
    return lookup


def fit_interaction_model(df, y_col):
    """OLS: y ~ log_centrality_z + is_del + is_ins + log_centrality_z:is_del
    + log_centrality_z:is_ins  (sub = reference). Bootstrapped over rows.
    Returns point estimates + 95% CI for every coefficient."""
    Xcols = ['log_centrality_z', 'is_del', 'is_ins', 'lc_x_del', 'lc_x_ins']
    X = df[Xcols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    n = len(df)

    point = LinearRegression().fit(X, y).coef_
    intercept_point = LinearRegression().fit(X, y).intercept_

    rng = np.random.default_rng(RNG)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        boots.append(LinearRegression().fit(Xb, yb).coef_)
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    mean = boots.mean(axis=0)
    stable = (lo > 0) | (hi < 0)

    return pd.DataFrame({
        'term': Xcols, 'point_estimate': point, 'boot_mean': mean,
        'boot_lo': lo, 'boot_hi': hi, 'stable': stable,
    }), intercept_point


def main():
    per_pos = pd.read_csv(PER_POS_CSV)
    cent_by_pos = per_pos.set_index('abs_pos')['betweenness_mean']
    lookup = make_lookup(cent_by_pos)

    X_tissue = build_mouse_pca_input()
    mean_abs_sel = X_tissue[TISSUES].abs().mean(axis=1)
    idx_df = pd.DataFrame(list(X_tissue.index), columns=X_tissue.index.names)
    idx_df['mean_abs_selection'] = mean_abs_sel.values
    for t in TISSUES:
        idx_df[t] = X_tissue[t].abs().values
    idx_df['betweenness'] = idx_df['abs_pos'].apply(lookup)
    idx_df['has_structure'] = idx_df['betweenness'].notna()

    covered = idx_df[idx_df.has_structure].copy()
    print(f"Total mutations: {len(idx_df)}; with 1LP3 structural coverage: {len(covered)}")
    print(covered.lib_type.value_counts().to_string())

    # ---- 1. per-lib_type correlations (mean-across-tissue + per tissue) ----
    print(f"\nBetweenness centrality vs |log2 selection|, by mutation type "
          f"(Spearman, {N_BOOT}-resample bootstrap CI):")
    corr_results = []
    for lt in ['sub', 'ins', 'del']:
        sub = covered[covered.lib_type == lt]
        rho, p, lo, hi, nb = spearman_boot(sub['betweenness'], sub['mean_abs_selection'])
        print(f"  {lt:4s} mean-across-tissues: rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}] "
              f"p={p:.4g}  (n={len(sub)})")
        corr_results.append(dict(lib_type=lt, comparison='mean_abs_selection', rho=rho,
                                  p=p, ci_lo=lo, ci_hi=hi, n=len(sub)))
        for t in TISSUES:
            rho, p, lo, hi, nb = spearman_boot(sub['betweenness'], sub[t])
            corr_results.append(dict(lib_type=lt, comparison=t, rho=rho, p=p,
                                      ci_lo=lo, ci_hi=hi, n=len(sub)))
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(os.path.join(OUT_DIR, 'centrality_by_lib_type_correlations.csv'), index=False)
    print("\nPer-tissue range by lib_type (min..max rho across the 6 tissues):")
    for lt in ['sub', 'ins', 'del']:
        sub = corr_df[(corr_df.lib_type == lt) & (corr_df.comparison != 'mean_abs_selection')]
        print(f"  {lt:4s}: {sub.rho.min():.3f} .. {sub.rho.max():.3f} "
              f"({(sub.p < 0.05).sum()}/6 tissues p<0.05)")

    # ---- 2. interaction model: is the centrality slope deletion-specific? ----
    covered['log_centrality'] = np.log(covered['betweenness'] + LOG_EPS)
    lc_mean, lc_std = covered['log_centrality'].mean(), covered['log_centrality'].std()
    covered['log_centrality_z'] = (covered['log_centrality'] - lc_mean) / lc_std
    covered['is_del'] = (covered.lib_type == 'del').astype(float)
    covered['is_ins'] = (covered.lib_type == 'ins').astype(float)
    covered['lc_x_del'] = covered['log_centrality_z'] * covered['is_del']
    covered['lc_x_ins'] = covered['log_centrality_z'] * covered['is_ins']

    print(f"\nInteraction model (sub = reference category, n={len(covered)}): "
          f"|log2 selection| ~ log_centrality_z + is_del + is_ins + "
          f"log_centrality_z:is_del + log_centrality_z:is_ins")
    interaction_rows = []
    for y_col in ['mean_abs_selection'] + TISSUES:
        coef_df, intercept = fit_interaction_model(covered, y_col)
        coef_df['target'] = y_col
        interaction_rows.append(coef_df)
        if y_col == 'mean_abs_selection':
            print(f"\n  target = {y_col}:")
            print(coef_df.to_string(index=False))
    interaction_df = pd.concat(interaction_rows, ignore_index=True)
    interaction_df.to_csv(os.path.join(OUT_DIR, 'centrality_lib_type_interaction_model.csv'), index=False)

    main_row = interaction_df[interaction_df.target == 'mean_abs_selection']
    del_int = main_row[main_row.term == 'lc_x_del'].iloc[0]
    ins_int = main_row[main_row.term == 'lc_x_ins'].iloc[0]
    main_slope = main_row[main_row.term == 'log_centrality_z'].iloc[0]
    print(f"\n  Direct answer -- is the centrality slope deletion-specific?")
    print(f"    substitution slope (reference): {main_slope.boot_mean:.4f} "
          f"[{main_slope.boot_lo:.4f}, {main_slope.boot_hi:.4f}] "
          f"({'stable' if main_slope.stable else 'CI includes 0'})")
    del_verdict = 'stable -- deletions differ from substitutions' if del_int.stable else "CI includes 0 -- NOT distinguishable from substitutions' slope"
    ins_verdict = 'stable -- insertions differ from substitutions' if ins_int.stable else "CI includes 0 -- NOT distinguishable from substitutions' slope"
    print(f"    deletion EXTRA slope (del - sub): {del_int.boot_mean:.4f} "
          f"[{del_int.boot_lo:.4f}, {del_int.boot_hi:.4f}] ({del_verdict})")
    print(f"    insertion EXTRA slope (ins - sub): {ins_int.boot_mean:.4f} "
          f"[{ins_int.boot_lo:.4f}, {ins_int.boot_hi:.4f}] ({ins_verdict})")

    n_tissue_stable_del = interaction_df[(interaction_df.term == 'lc_x_del') &
                                          (interaction_df.target != 'mean_abs_selection')].stable.sum()
    n_tissue_stable_ins = interaction_df[(interaction_df.term == 'lc_x_ins') &
                                          (interaction_df.target != 'mean_abs_selection')].stable.sum()
    print(f"\n  Per-tissue (6 models): del interaction stable in {n_tissue_stable_del}/6, "
          f"ins interaction stable in {n_tissue_stable_ins}/6")

    # ---- figure ----
    fig, axes = plt.subplots(ncols=2, figsize=[13, 5])
    colors = {'sub': 'tab:blue', 'ins': 'tab:green', 'del': 'tab:red'}
    for lt in ['sub', 'ins', 'del']:
        sub = covered[covered.lib_type == lt]
        axes[0].scatter(sub['betweenness'], sub['mean_abs_selection'], alpha=0.25,
                         s=14, color=colors[lt], label=f'{lt} (n={len(sub)})')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('betweenness centrality (log scale)')
    axes[0].set_ylabel('mean |log2 selection| across 6 tissues')
    axes[0].set_title('Centrality vs selection magnitude, by mutation type')
    axes[0].legend(fontsize=8)

    plot_terms = ['log_centrality_z', 'lc_x_del', 'lc_x_ins']
    plot_labels = ['sub slope\n(reference)', 'deletion\nextra slope', 'insertion\nextra slope']
    sub_main = interaction_df[(interaction_df.target == 'mean_abs_selection') &
                               (interaction_df.term.isin(plot_terms))]
    sub_main = sub_main.set_index('term').loc[plot_terms].reset_index()
    bar_colors = ['#4C72B0' if not st else ('#DD8452' if m > 0 else '#55A868')
                  for m, st in zip(sub_main.boot_mean, sub_main.stable)]
    xerr = np.array([sub_main.boot_mean - sub_main.boot_lo, sub_main.boot_hi - sub_main.boot_mean])
    axes[1].bar(plot_labels, sub_main.boot_mean, yerr=xerr, color=bar_colors, capsize=4)
    axes[1].axhline(0, color='black', lw=0.6)
    axes[1].set_ylabel('OLS coefficient (bootstrap mean +/- 95% CI)')
    axes[1].set_title('Is the centrality-selection slope\ndeletion-specific? (gray = CI includes 0)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'centrality_lib_type_interaction.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT_DIR}/centrality_lib_type_interaction.png")
    print("Done.")


if __name__ == '__main__':
    main()
