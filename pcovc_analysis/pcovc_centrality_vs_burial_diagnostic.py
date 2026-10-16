"""
Diagnostic: is betweenness centrality actually measuring the same thing as
"burial" (solvent accessibility)? And is Ogden et al.'s pooled qualitative
claim ("buried positions... more deleterious", stated across ALL mutation
types combined) actually driven by indels rather than substitutions?

Motivation (Adarsh, 2026-09-03): pcovc_residue_network_centrality_full_mutations.py
found substitutions have a SIGNIFICANT NEGATIVE correlation between
betweenness centrality and |log2 selection| (rho=-0.135, p=1.4e-05, n=1024)
-- higher centrality, LESS deleterious. That's the opposite of the standard
DMS-literature expectation (burial/centrality -> more substitution-deleterious)
and opposite of Ogden's own one-line pooled claim. A literature check flagged
this as needing scrutiny before treating the finding as clean, specifically
because betweenness centrality (network connectivity / how many shortest
paths pass through a residue) and solvent burial (how buried from solvent) are
conceptually different structural properties -- a residue can be buried but
topologically peripheral (dead-end pocket), or surface-adjacent but
topologically central (a hub at a symmetry interface).

This script:
  A. Computes true per-residue solvent accessibility (SASA/RSA, Shrake-Rupley,
     Bio.PDB) for chain A embedded in its local capsid neighborhood (chain A +
     its 5 CA-CA-contacting symmetry copies, identified via the same 8A
     cutoff as the network build -- NOT chain A in isolation, which would
     wrongly report inter-subunit interface residues as "exposed").
     Correlates RSA against betweenness centrality (are they redundant,
     orthogonal, or anti-correlated?), then re-runs the substitution-only
     |log2 selection| correlation using RSA instead of centrality: does the
     standard "buried = more deleterious" direction reappear once burial,
     not centrality, is the measure?
  B. Tests Ogden's own pooled claim directly on our local copy of their data:
     pool all mutation types (sub/ins/del -- this processed dataset has no
     separate stop-codon category, confirmed via lib_type.value_counts(),
     so a "stops driving it" hypothesis cannot be tested here, only
     "indels driving it") and check whether RSA vs |log2 selection| is
     negative (buried=more deleterious) in the POOLED data, then decompose by
     lib_type to see whether that pooled signal survives, disappears, or
     flips when substitutions are isolated.

Run with cwd = <repo>/x02_analysis, using venv_pcovc/bin/python3.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, Structure, Model, Chain
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input
from pcovc_residue_network_centrality import (
    parse_ca_coords, parse_biomt, build_full_assembly_ca, PDB_PATH,
    PDB_TO_VP1_OFFSET, CUTOFF_A,
)

OUT_DIR = './pcovc_analysis/deep_dive'
PER_POS_CENTRALITY_CSV = os.path.join(OUT_DIR, 'residue_network_per_position_centrality.csv')
os.makedirs(OUT_DIR, exist_ok=True)

TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
N_BOOT = 1000
RNG = 0

# Tien et al. 2013 (PLOS ONE) theoretical maximum ASA, Angstrom^2 -- standard
# reference set for computing relative solvent accessibility (RSA = ASA / max_ASA).
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}


def spearman_boot(x, y, n_boot=N_BOOT, rng_seed=RNG):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
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
    return rho, p, lo, hi


def identify_local_neighborhood():
    """Which of the 60 BIOMT copies directly CA-CA contact (<8A) the
    identity copy (chain A itself)? Same cutoff as the network build, so
    the local SASA neighborhood matches what "in contact" meant there."""
    biomt = parse_biomt(PDB_PATH)
    identity_idx = None
    for i, m in enumerate(biomt, start=1):
        if np.allclose(m['R'], np.eye(3), atol=1e-4) and np.allclose(m['t'], 0, atol=1e-3):
            identity_idx = i
    assert identity_idx is not None, "no identity BIOMT matrix found"

    coords, copy_id, resnum = build_full_assembly_ca(PDB_PATH)
    identity_mask = copy_id == identity_idx
    tree_all = cKDTree(coords)
    tree_id = cKDTree(coords[identity_mask])
    pairs = tree_id.query_ball_tree(tree_all, r=CUTOFF_A)
    neighbor_copies = set()
    for lst in pairs:
        for j in lst:
            if copy_id[j] != identity_idx:
                neighbor_copies.add(int(copy_id[j]))
    return identity_idx, sorted(neighbor_copies), biomt


def build_local_assembly_structure():
    """Bio.PDB Structure with chain A (identity) + its directly-contacting
    symmetry copies, full atoms (not just CA), for realistic SASA -- an
    isolated chain A would wrongly report inter-subunit interface residues
    as solvent-exposed."""
    identity_idx, neighbor_copies, biomt = identify_local_neighborhood()
    print(f"  identity copy: {identity_idx}; local neighborhood: "
          f"{neighbor_copies} ({len(neighbor_copies)} contacting copies)")

    parser = PDBParser(QUIET=True)
    orig_structure = parser.get_structure('1LP3', PDB_PATH)
    orig_chain = orig_structure[0]['A']

    structure = Structure.Structure('local_assembly')
    model = Model.Model(0)
    structure.add(model)

    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_copy_ids = [identity_idx] + neighbor_copies
    for letter, copy_id in zip(chain_letters, all_copy_ids):
        m = biomt[copy_id - 1]
        new_chain = Chain.Chain(letter)
        for residue in orig_chain:
            if residue.id[0] != ' ':  # skip heteroatoms/waters
                continue
            new_residue = residue.copy()
            for atom in new_residue:
                coord = atom.get_coord()
                atom.set_coord(coord @ m['R'].T + m['t'])
            new_chain.add(new_residue)
        model.add(new_chain)
    return structure, chain_letters[0]  # identity is always the first chain added


def compute_sasa():
    print("Building local 6-chain assembly (identity + contacting neighbors) for SASA...")
    structure, identity_chain_letter = build_local_assembly_structure()
    print("Running Shrake-Rupley SASA...")
    sr = ShrakeRupley()
    sr.compute(structure, level='R')

    rows = []
    identity_chain = structure[0][identity_chain_letter]
    for residue in identity_chain:
        resnum = residue.id[1]
        resname = residue.get_resname()
        asa = residue.sasa
        max_asa = MAX_ASA.get(resname, np.nan)
        rsa = asa / max_asa if not np.isnan(max_asa) else np.nan
        rows.append(dict(pdb_resnum=resnum, resname=resname, asa=asa, rsa=rsa))
    df = pd.DataFrame(rows)
    df['abs_pos'] = df['pdb_resnum'] + PDB_TO_VP1_OFFSET
    return df


def make_lookup(series_by_pos):
    def lookup(pos):
        if float(pos).is_integer():
            return series_by_pos.get(pos, np.nan)
        lo, hi = np.floor(pos), np.ceil(pos)
        v_lo, v_hi = series_by_pos.get(lo, np.nan), series_by_pos.get(hi, np.nan)
        if np.isnan(v_lo) or np.isnan(v_hi):
            return np.nan
        return (v_lo + v_hi) / 2
    return lookup


def main():
    # ---- A1: compute SASA/RSA ----
    sasa_df = compute_sasa()
    sasa_df.to_csv(os.path.join(OUT_DIR, 'per_position_sasa_rsa.csv'), index=False)
    print(f"\nSASA computed for {len(sasa_df)} chain-A residues "
          f"(abs_pos {sasa_df.abs_pos.min():.0f}-{sasa_df.abs_pos.max():.0f})")

    centrality_df = pd.read_csv(PER_POS_CENTRALITY_CSV)
    merged = centrality_df.merge(sasa_df, on='abs_pos', how='inner')
    print(f"Merged centrality+SASA coverage: {len(merged)} positions "
          f"(centrality had {len(centrality_df)}, SASA had {len(sasa_df)})")

    # ---- A2: centrality vs RSA -- redundant, orthogonal, or opposed? ----
    rho, p, lo, hi = spearman_boot(merged['betweenness_mean'], merged['rsa'])
    print(f"\nBetweenness centrality vs RSA (solvent accessibility), n={len(merged)}:")
    print(f"  rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}] p={p:.4g}")
    if rho < -0.2 and p < 0.05:
        print("  -> negatively correlated: high centrality tends to mean MORE buried "
              "(consistent-ish with both being 'structural importance', but distinct axes)")
    elif abs(rho) < 0.2 or p >= 0.05:
        print("  -> weak/no correlation: centrality and burial are largely ORTHOGONAL "
              "structural properties in this capsid, not redundant measures")
    else:
        print("  -> positively correlated: unexpected, high centrality tends to mean "
              "MORE exposed")

    centrality_by_pos = centrality_df.set_index('abs_pos')['betweenness_mean']
    rsa_by_pos = sasa_df.set_index('abs_pos')['rsa']
    lookup_cent = make_lookup(centrality_by_pos)
    lookup_rsa = make_lookup(rsa_by_pos)

    # ---- load mutation data ----
    X_tissue = build_mouse_pca_input()
    mean_abs_sel = X_tissue[TISSUES].abs().mean(axis=1)
    idx_df = pd.DataFrame(list(X_tissue.index), columns=X_tissue.index.names)
    idx_df['mean_abs_selection'] = mean_abs_sel.values
    idx_df['betweenness'] = idx_df['abs_pos'].apply(lookup_cent)
    idx_df['rsa'] = idx_df['abs_pos'].apply(lookup_rsa)
    idx_df['has_rsa'] = idx_df['rsa'].notna()
    covered = idx_df[idx_df.has_rsa].copy()
    print(f"\nMutations with RSA coverage: {len(covered)} / {len(idx_df)} "
          f"({covered.lib_type.value_counts().to_dict()})")

    # ---- A3: does using RSA instead of centrality flip the substitution result? ----
    print("\n=== A3: RSA vs |log2 selection|, by mutation type "
          f"({N_BOOT}-resample bootstrap) ===")
    print("(negative rho = 'buried [low RSA] -> more deleterious', matching the "
          "standard DMS-literature / Ogden's own pooled-claim direction)")
    rsa_results = []
    for lt in ['sub', 'ins', 'del']:
        sub = covered[covered.lib_type == lt]
        rho, p, lo, hi = spearman_boot(sub['rsa'], sub['mean_abs_selection'])
        direction = "MATCHES standard direction (buried=more deleterious)" if rho < 0 else \
                    "still OPPOSITE of standard direction"
        print(f"  {lt:4s}: rho={rho:.3f} [95% CI {lo:.3f}, {hi:.3f}] p={p:.4g} "
              f"(n={len(sub)})  -- {direction}")
        rsa_results.append(dict(lib_type=lt, rho=rho, p=p, ci_lo=lo, ci_hi=hi, n=len(sub)))
    pd.DataFrame(rsa_results).to_csv(
        os.path.join(OUT_DIR, 'rsa_by_lib_type_correlations.csv'), index=False)

    # ---- B: pooled (mimicking Ogden's aggregate claim) vs decomposed ----
    print("\n=== B: pooled (all lib_types together, mimicking Ogden's aggregate "
          "claim) vs decomposed by lib_type ===")
    print("NOTE: this processed dataset has no separate stop-codon/nonsense category "
          f"(lib_type values present: {sorted(idx_df.lib_type.unique())}) -- only an "
          "indels-vs-substitutions decomposition is testable here, not a "
          "stops-vs-substitutions one.")
    rho_pool, p_pool, lo_pool, hi_pool = spearman_boot(covered['rsa'], covered['mean_abs_selection'])
    print(f"  POOLED (n={len(covered)}): rho={rho_pool:.3f} "
          f"[95% CI {lo_pool:.3f}, {hi_pool:.3f}] p={p_pool:.4g}")
    print(f"  (for reference, substitution-only was rho={rsa_results[0]['rho']:.3f}, "
          f"p={rsa_results[0]['p']:.4g})")

    indel = covered[covered.lib_type.isin(['ins', 'del'])]
    rho_indel, p_indel, lo_indel, hi_indel = spearman_boot(indel['rsa'], indel['mean_abs_selection'])
    print(f"  INDELS ONLY (ins+del, n={len(indel)}): rho={rho_indel:.3f} "
          f"[95% CI {lo_indel:.3f}, {hi_indel:.3f}] p={p_indel:.4g}")

    verdict_lines = []
    if rho_pool < 0 and rsa_results[0]['rho'] >= 0:
        verdict_lines.append(
            "Pooled signal is negative (matches Ogden's aggregate claim direction) "
            "but substitutions ALONE are not negative -- consistent with the pooled "
            "claim being driven by indels, not substitutions.")
    elif rho_pool < 0 and rsa_results[0]['rho'] < 0:
        verdict_lines.append(
            "Pooled AND substitution-only are both negative -- the standard direction "
            "holds for substitutions once RSA (not centrality) is used; the earlier "
            "centrality-based negative-substitution result reflects centrality and "
            "burial measuring different things, not a contradiction of the standard "
            "burial-deleteriousness relationship.")
    else:
        verdict_lines.append(
            "Pooled signal itself is not clearly negative here -- the "
            "'buried=more deleterious' pattern may not replicate strongly in this "
            "processed subset at all; treat with caution.")
    print("\n=== VERDICT ===")
    for line in verdict_lines:
        print("  " + line)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(merged['rsa'], merged['betweenness_mean'], s=10, alpha=0.5)
    axes[0].set_xlabel('RSA (solvent accessibility)')
    axes[0].set_ylabel('Betweenness centrality')
    axes[0].set_title('Centrality vs RSA\n(are they redundant?)')

    for lt, c in zip(['sub', 'ins', 'del'], ['tab:blue', 'tab:orange', 'tab:green']):
        sub = covered[covered.lib_type == lt]
        axes[1].scatter(sub['rsa'], sub['mean_abs_selection'], s=10, alpha=0.4, c=c, label=lt)
    axes[1].set_xlabel('RSA (solvent accessibility)')
    axes[1].set_ylabel('mean |log2 selection|')
    axes[1].set_title('RSA vs deleteriousness, by type')
    axes[1].legend()

    axes[2].bar(['sub', 'ins', 'del'], [r['rho'] for r in rsa_results],
                yerr=[[r['rho'] - r['ci_lo'] for r in rsa_results],
                      [r['ci_hi'] - r['rho'] for r in rsa_results]],
                capsize=4, color=['tab:blue', 'tab:orange', 'tab:green'])
    axes[2].axhline(0, color='k', lw=0.8)
    axes[2].set_ylabel("Spearman rho (RSA vs |log2 sel|)")
    axes[2].set_title('RSA-deleteriousness rho by type\n(negative = standard direction)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'centrality_vs_burial_diagnostic.png'), dpi=150)
    print(f"\nSaved plot to {OUT_DIR}/centrality_vs_burial_diagnostic.png")


if __name__ == '__main__':
    main()
