"""
Full-capsid residue interaction network (RIN): does structural centrality
predict how deleterious a mutation is, specifically for deletions?

Motivation (Adarsh, 2026-09-03): the existing structural-context analysis
(pcovc_structural_context.py) only tests symmetry-axis PROXIMITY (binary:
is a position near a 2/3/5-fold interface), derived from VIPERdb's curated
650-row contact list. That list is itself axis-biased (VIPERdb only reports
contacts VIPERdb flagged as axis-relevant), so it can't support a genuine
network-topology metric like betweenness centrality without just re-deriving
axis membership. This script instead builds a real residue interaction
network from the full 60-subunit icosahedral capsid assembly (PDB 1LP3,
Xie et al. 2002, applying the entry's own REMARK 350 BIOMT operators) and
tests whether network centrality -- specifically betweenness, a measure of
how many shortest paths through the capsid's contact network pass through a
given residue -- predicts mutational sensitivity, with a focus on deletion
mutants (the hypothesis being that deletions at highly-central, structurally
"load-bearing" residues should be more deleterious than deletions at
peripheral ones).

Prior-art check (see fork report): no existing published RIN/centrality
analysis of the AAV2 capsid was found tied to this dataset or otherwise;
this appears to be a genuinely novel angle for this system. The closest
independent precedent is Ronzitti-group work (Mol Ther Methods Clin Dev
2024, PMC11403266) showing buried (non-HVR) residues have a much higher
deleterious-mutation rate (65%) than accessible HVR residues (15%) --
consistent with, but not the same claim as, "high-centrality residues are
more sensitive to deletion."

Structure: PDB 1LP3 resolves only the VP3-common region (PDB numbering
80-598 -> VP1/abs_pos numbering 217-735, via the repo's own established
+137 offset, common.py:load_axis_contacts). The VP1/VP2-unique N-terminus
(abs_pos < 217) has no coordinates and is excluded from this analysis --
37 of the 41 deletion mutants fall in the resolved range.

Method:
  1. Parse chain A (the asymmetric unit) CA coordinates from 1LP3.pdb.
  2. Parse all 60 REMARK 350 BIOMT matrices and apply each to chain A's CA
     coordinates to generate the full 60-copy icosahedral assembly.
  3. Build a contact graph: nodes = (copy_id, pdb_resnum), edges = any pair
     of CA atoms (any copy, including cross-copy/inter-subunit pairs) within
     8.0 A of each other (KDTree query_pairs, not a naive O(n^2) loop --
     31,140 nodes makes the naive approach impractical).
  4. Compute APPROXIMATE betweenness centrality (networkx, k-pivot sampling
     a la Brandes & Pich 2007) -- exact betweenness on ~31k nodes / ~230k
     edges is computationally infeasible (O(V*E) Brandes' algorithm). Run 5
     independent seeds to report sampling variability alongside the T=1
     icosahedral symmetry check (per-position centrality should already be
     near-identical across its 60 physical copies; both are reported as
     sanity checks on the network build, distinct from the downstream
     mutation-correlation bootstrap).
  5. Aggregate to one centrality value per sequence position (mean across
     the 60 symmetry-equivalent copies), convert pdb_resnum -> abs_pos.
  6. Correlate against deletion mutants' |log2 selection| (Spearman,
     bootstrap CI over mutants, N_BOOT=1000, matching this project's
     established convention in pcovc_organ_ols_bootstrap.py), plus two
     secondary comparisons: centrality by lib_type (del vs sub vs ins), and
     centrality of high- vs low-|log2 selection| deletions.

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import cKDTree
from scipy import stats

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input

OUT_DIR = './pcovc_analysis/deep_dive'
STRUCT_DIR = './pcovc_analysis/structures'
PDB_PATH = os.path.join(STRUCT_DIR, '1LP3.pdb')
os.makedirs(OUT_DIR, exist_ok=True)

TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
# repo-established PDB(VIPERdb) -> VP1/abs_pos numbering offset, verified in
# common.py:load_axis_contacts (res_vp = res_pdb + 219 - 82)
PDB_TO_VP1_OFFSET = 137
CUTOFF_A = 8.0  # CA-CA contact cutoff, angstroms
N_BOOT = 1000
RNG = 0
BETWEENNESS_K = 300  # pivot sample size for approximate betweenness
BETWEENNESS_SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- parsing --
def parse_ca_coords(pdb_path):
    """Chain A CA coordinates from the ATOM records, keyed by PDB resnum."""
    resnum_to_xyz = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            chain = line[21]
            if atom_name != 'CA' or chain != 'A':
                continue
            resnum = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            resnum_to_xyz[resnum] = np.array([x, y, z])
    resnums = np.array(sorted(resnum_to_xyz))
    coords = np.array([resnum_to_xyz[r] for r in resnums])
    return resnums, coords


def parse_biomt(pdb_path):
    """All REMARK 350 BIOMT rotation+translation matrices."""
    mats = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('REMARK 350   BIOMT'):
                continue
            row = int(line[18:19])
            mat_id = int(line[19:24])
            vals = [float(x) for x in line[24:].split()]
            R_row, t = vals[:3], vals[3]
            entry = mats.setdefault(mat_id, {'R': np.zeros((3, 3)), 't': np.zeros(3)})
            entry['R'][row - 1] = R_row
            entry['t'][row - 1] = t
    return [mats[k] for k in sorted(mats)]


def build_full_assembly_ca(pdb_path):
    resnums, coords = parse_ca_coords(pdb_path)
    biomt = parse_biomt(pdb_path)
    assert len(biomt) == 60, f"expected 60 BIOMT matrices, got {len(biomt)}"
    all_coords, all_copy_id, all_resnum = [], [], []
    for copy_id, m in enumerate(biomt, start=1):
        transformed = coords @ m['R'].T + m['t']
        all_coords.append(transformed)
        all_copy_id.append(np.full(len(resnums), copy_id))
        all_resnum.append(resnums)
    return (np.concatenate(all_coords), np.concatenate(all_copy_id),
            np.concatenate(all_resnum))


# ---------------------------------------------------------- network build --
def build_contact_graph(coords, copy_id, resnum, cutoff=CUTOFF_A):
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    G = nx.Graph()
    nodes = list(zip(copy_id.tolist(), resnum.tolist()))
    G.add_nodes_from(nodes)
    edges = [(nodes[i], nodes[j]) for i, j in pairs]
    G.add_edges_from(edges)
    return G, nodes


def approx_betweenness(G, k=BETWEENNESS_K, seeds=BETWEENNESS_SEEDS):
    """Multiple pivot-sampled runs -> per-node mean + spread across seeds,
    reported as a sanity check on estimator noise (separate from the
    mutation-level bootstrap below)."""
    runs = []
    for s in seeds:
        t0 = time.time()
        bc = nx.betweenness_centrality(G, k=k, seed=s, normalized=True)
        print(f"  betweenness seed={s}: {time.time() - t0:.1f}s")
        runs.append(pd.Series(bc))
    df = pd.concat(runs, axis=1)
    df.columns = [f'seed{s}' for s in seeds]
    return df


# --------------------------------------------------------------- analysis --
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


def main():
    # ---- 1-2. structure + assembly ----
    print("Parsing 1LP3 chain A CA coordinates + BIOMT matrices...")
    resnums, ca = parse_ca_coords(PDB_PATH)
    print(f"  chain A resolved range: {resnums.min()}-{resnums.max()} "
          f"(PDB numbering) = {resnums.min()+PDB_TO_VP1_OFFSET}-{resnums.max()+PDB_TO_VP1_OFFSET} (VP1/abs_pos)")
    coords, copy_id, resnum = build_full_assembly_ca(PDB_PATH)
    print(f"  full assembly: {len(coords)} CA atoms across {copy_id.max()} copies")

    # ---- 3. contact graph ----
    print(f"Building contact graph (CA-CA < {CUTOFF_A} A)...")
    G, nodes = build_contact_graph(coords, copy_id, resnum)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"connected={nx.is_connected(G)}")

    # ---- 4. approximate betweenness, multi-seed ----
    print(f"Computing approximate betweenness centrality (k={BETWEENNESS_K}, "
          f"{len(BETWEENNESS_SEEDS)} seeds)...")
    bc_runs = approx_betweenness(G)
    bc_runs.index = pd.MultiIndex.from_tuples(bc_runs.index, names=['copy_id', 'pdb_resnum'])
    bc_mean_per_node = bc_runs.mean(axis=1)
    seed_cv = (bc_runs.std(axis=1) / bc_runs.mean(axis=1).replace(0, np.nan)).median()
    print(f"  median per-node coefficient of variation across seeds: {seed_cv:.3f}")

    # ---- 5. aggregate to per-position (across 60 symmetry copies) ----
    per_pos = bc_mean_per_node.groupby(level='pdb_resnum').agg(['mean', 'std', 'count'])
    per_pos.columns = ['betweenness_mean', 'betweenness_std_across_copies', 'n_copies']
    per_pos['betweenness_cv_across_copies'] = (
        per_pos['betweenness_std_across_copies'] / per_pos['betweenness_mean'])
    per_pos = per_pos.reset_index().rename(columns={'pdb_resnum': 'pdb_resnum'})
    per_pos['abs_pos'] = per_pos['pdb_resnum'] + PDB_TO_VP1_OFFSET
    print(f"\nSymmetry sanity check -- median betweenness CV *across the 60 physical "
          f"copies* of the same position (should be small if the T=1 assembly and "
          f"graph were built correctly): {per_pos['betweenness_cv_across_copies'].median():.3f}")
    per_pos.to_csv(os.path.join(OUT_DIR, 'residue_network_per_position_centrality.csv'), index=False)

    # ---- 6. mutation data + correlation ----
    X_tissue = build_mouse_pca_input()
    mean_abs_sel = X_tissue[TISSUES].abs().mean(axis=1)
    idx_df = pd.DataFrame(list(X_tissue.index), columns=X_tissue.index.names)
    idx_df['mean_abs_selection'] = mean_abs_sel.values
    for t in TISSUES:
        idx_df[t] = X_tissue[t].abs().values

    # Substitutions/deletions sit at integer abs_pos (direct lookup).
    # Insertions sit at half-integer abs_pos (e.g. 21.5 = inserted between
    # residues 21 and 22) -- map to the mean centrality of the two flanking
    # residues when both are structurally resolved, else NaN (no silent
    # partial-coverage guessing).
    cent_by_pos = per_pos.set_index('abs_pos')['betweenness_mean']

    def lookup_centrality(pos):
        if float(pos).is_integer():
            return cent_by_pos.get(pos, np.nan)
        lo, hi = np.floor(pos), np.ceil(pos)
        c_lo, c_hi = cent_by_pos.get(lo, np.nan), cent_by_pos.get(hi, np.nan)
        if np.isnan(c_lo) or np.isnan(c_hi):
            return np.nan
        return (c_lo + c_hi) / 2

    idx_df['betweenness'] = idx_df['abs_pos'].apply(lookup_centrality)
    idx_df['has_structure'] = idx_df['betweenness'].notna()

    dels = idx_df[idx_df.lib_type == 'del'].copy()
    n_del_total = len(dels)
    dels_covered = dels[dels.has_structure]
    print(f"\nDeletion mutants: {n_del_total} total, {len(dels_covered)} with 1LP3 "
          f"structural coverage (abs_pos in {resnums.min()+PDB_TO_VP1_OFFSET}-"
          f"{resnums.max()+PDB_TO_VP1_OFFSET})")

    results = []
    print("\nBetweenness centrality vs |log2 selection|, deletions only "
          f"(Spearman, {N_BOOT}-resample bootstrap CI):")
    rho, p, lo, hi, nb = spearman_boot(dels_covered['betweenness'], dels_covered['mean_abs_selection'])
    print(f"  mean-across-tissues:  rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}]  "
          f"p={p:.4f}  (n={len(dels_covered)}, {nb}/{N_BOOT} valid resamples)")
    results.append(dict(comparison='mean_abs_selection', rho=rho, p=p, ci_lo=lo, ci_hi=hi,
                         n=len(dels_covered)))
    for t in TISSUES:
        rho, p, lo, hi, nb = spearman_boot(dels_covered['betweenness'], dels_covered[t])
        print(f"  {t:8s}:  rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}]  p={p:.4f}")
        results.append(dict(comparison=t, rho=rho, p=p, ci_lo=lo, ci_hi=hi, n=len(dels_covered)))
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, 'residue_network_centrality_vs_deletion_selection.csv'), index=False)

    # ---- secondary: centrality by lib_type (positions harboring each type) ----
    print("\nSecondary check -- betweenness of positions harboring each mutation type "
          "(Mann-Whitney U, positions with structural coverage only):")
    covered = idx_df[idx_df.has_structure]
    lib_groups = {lt: covered.loc[covered.lib_type == lt, 'betweenness'].dropna()
                  for lt in ['del', 'sub', 'ins']}
    for a, b in [('del', 'sub'), ('del', 'ins'), ('sub', 'ins')]:
        u, p = stats.mannwhitneyu(lib_groups[a], lib_groups[b], alternative='two-sided')
        print(f"  {a} (mean={lib_groups[a].mean():.4g}, n={len(lib_groups[a])}) vs "
              f"{b} (mean={lib_groups[b].mean():.4g}, n={len(lib_groups[b])}): "
              f"Mann-Whitney p={p:.4f}")

    # ---- secondary: high vs low |selection| deletions ----
    print("\nSecondary check -- betweenness, high- vs low-|selection| deletions "
          "(median split, covered deletions only):")
    med = dels_covered['mean_abs_selection'].median()
    hi_sel = dels_covered.loc[dels_covered.mean_abs_selection >= med, 'betweenness']
    lo_sel = dels_covered.loc[dels_covered.mean_abs_selection < med, 'betweenness']
    u, p = stats.mannwhitneyu(hi_sel, lo_sel, alternative='two-sided')
    print(f"  high-|sel| (n={len(hi_sel)}, mean betweenness={hi_sel.mean():.4g}) vs "
          f"low-|sel| (n={len(lo_sel)}, mean betweenness={lo_sel.mean():.4g}): "
          f"Mann-Whitney p={p:.4f}")

    dels_covered.to_csv(os.path.join(OUT_DIR, 'deletion_mutants_with_centrality.csv'), index=False)

    # ---- figure ----
    fig, axes = plt.subplots(ncols=2, figsize=[12, 5])
    axes[0].scatter(dels_covered['betweenness'], dels_covered['mean_abs_selection'],
                     alpha=0.7, color='tab:red', edgecolor='k', linewidth=0.4)
    axes[0].set_xlabel('mean betweenness centrality (60-copy avg, log scale)')
    axes[0].set_ylabel('mean |log2 selection| across 6 tissues')
    axes[0].set_xscale('log')
    axes[0].set_title(f"Deletions only (n={len(dels_covered)})\n"
                       f"Spearman rho={results_df.iloc[0]['rho']:.2f} "
                       f"[{results_df.iloc[0]['ci_lo']:.2f}, {results_df.iloc[0]['ci_hi']:.2f}]")

    box_data = [lib_groups['del'], lib_groups['sub'], lib_groups['ins']]
    axes[1].boxplot(box_data, tick_labels=['del', 'sub', 'ins'], showfliers=False)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('betweenness centrality (position harboring mutation, log scale)')
    axes[1].set_title('Centrality of mutated positions by mutation type')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'residue_network_centrality.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved {OUT_DIR}/residue_network_centrality.png")
    print("Done.")


if __name__ == '__main__':
    main()
