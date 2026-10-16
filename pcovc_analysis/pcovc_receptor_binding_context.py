"""
Cross-references PCovC/structural-context findings against literature-defined
AAV2 receptor-binding footprints. Not present anywhere else in this project
(confirmed by grep for "receptor" across all scripts/CSVs/decks) -- this is
new analysis, not a reproduction of an existing figure.

Two receptor footprints, both reported in AAV2 VP1 numbering (matching this
project's abs_pos convention -- see per_position_sasa_rsa.csv provenance:
1LP3.pdb chain A covers VP1 residues 80-598, abs_pos = pdb_resnum + 137):

1. Heparan sulfate proteoglycan (HSPG), the primary AAV2 attachment factor.
   Basic-patch residues R484, R487, K527, K532, R585, R588; R585/R588 are the
   two residues whose individual mutation abolishes heparin-agarose binding.
   Kern et al. 2003 J Virol 77:11072 ("Identification of a Heparin-Binding
   Motif on AAV2"); Opie et al. 2003 J Virol 77:6995.

2. AAVR (KIAA0319L), the universal proteinaceous receptor. Its PKD2 domain
   binds the capsid spike at the icosahedral 3-fold axis; the AAV2 contact
   surface is built from variable regions VR-I (BC-loop, ~262-269), and
   VR-IV/V/VIII, which interdigitate to form the 3-fold protrusions.
   Zhang et al. 2019 Nat Microbiol 4:675 ("AAV2 bound to its cellular
   receptor AAVR"); Meyer et al. 2019 eLife 8:e44707.

Notably, this project's own VR_BOUNDS (pcovc_mutation_features.py, from
a01_analyze_packaging_data.py) place ALL SIX HSPG residues inside or
immediately adjacent to VR-V (487-504), VR-VI (525-540), or VR-VIII
(579-594) -- i.e. the HSPG and AAVR footprints spatially overlap on VR-V
and VR-VIII specifically, consistent with both receptors engaging the same
3-fold protrusion architecture.

Run with cwd = pcovc_analysis, python3 (no PCovC-specific deps needed).
"""
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = './pcovc_analysis/deep_dive'

HSPG_RESIDUES = [484, 487, 527, 532, 585, 588]
HSPG_PRIMARY = [585, 588]  # single-mutation-abolishes-binding residues
AAVR_VR_REGIONS = ['VR-I', 'VR-IV', 'VR-V', 'VR-VIII']
HSPG_VR_REGIONS = ['VR-V', 'VR-VI', 'VR-VIII']  # per VR_BOUNDS overlap noted above

ctx = pd.read_csv(f'{OUT_DIR}/per_position_structural_context.csv')
rsa = pd.read_csv(f'{OUT_DIR}/per_position_sasa_rsa.csv')[['abs_pos', 'rsa']]
cent = pd.read_csv(f'{OUT_DIR}/residue_network_per_position_centrality.csv')[['abs_pos', 'betweenness_mean']]

df = ctx.merge(rsa, on='abs_pos', how='left').merge(cent, on='abs_pos', how='left')
# per_position_structural_context.csv gives insertion mutants half-integer
# abs_pos labels (X.5 = inserted between residue X and X+1, confirmed by
# inspection -- 147/421 rows are non-integer). Do NOT round/truncate these:
# that silently collapses two distinct positions (e.g. 527.0 and 527.5) onto
# one integer and corrupts the per-residue stats below. Receptor residues
# from crystallography are substitution positions, so exact-match only the
# true integer rows for residue-level lookups.
df['is_integer_pos'] = (df['abs_pos'] % 1 == 0)
df['is_hspg_residue'] = df['is_integer_pos'] & df['abs_pos'].isin(HSPG_RESIDUES)
df['is_aavr_vr'] = df['vr_region'].isin(AAVR_VR_REGIONS)
df['is_hspg_vr'] = df['vr_region'].isin(HSPG_VR_REGIONS)

print("=== HSPG basic-patch residues: are they where we'd expect? ===")
found = df[df['is_hspg_residue']][['abs_pos', 'vr_region', 'rsa', 'betweenness_mean', 'mean_abs_selection', 'n_mutants']]
print(found.to_string(index=False))
missing = set(HSPG_RESIDUES) - set(found['abs_pos'])
if missing:
    print(f"NOT in scan/structure data: {sorted(missing)}")

print(f"\nMean RSA at HSPG residues: {found['rsa'].mean():.3f} "
      f"vs whole-protein mean: {df['rsa'].mean():.3f} "
      f"(surface accessibility sanity check -- receptor-binding residues should be exposed)")

print("\n=== Selection strength: HSPG-associated VRs (V/VI/VIII) vs rest ===")
g_hspg = df[df['is_hspg_vr']]['mean_abs_selection'].dropna()
g_rest = df[~df['is_hspg_vr']]['mean_abs_selection'].dropna()
u, p = stats.mannwhitneyu(g_hspg, g_rest, alternative='two-sided')
print(f"HSPG-VR (n={len(g_hspg)}) mean={g_hspg.mean():.4f}  vs  rest (n={len(g_rest)}) mean={g_rest.mean():.4f}"
      f"  Mann-Whitney p={p:.3g}")

print("\n=== Selection strength: AAVR-associated VRs (I/IV/V/VIII) vs rest ===")
g_aavr = df[df['is_aavr_vr']]['mean_abs_selection'].dropna()
g_rest2 = df[~df['is_aavr_vr']]['mean_abs_selection'].dropna()
u2, p2 = stats.mannwhitneyu(g_aavr, g_rest2, alternative='two-sided')
print(f"AAVR-VR (n={len(g_aavr)}) mean={g_aavr.mean():.4f}  vs  rest (n={len(g_rest2)}) mean={g_rest2.mean():.4f}"
      f"  Mann-Whitney p={p2:.3g}")

print("\n=== Named-tropism-cluster enrichment in receptor-footprint VRs ===")
for label, mask in [("HSPG-VR (V/VI/VIII)", df['is_hspg_vr']), ("AAVR-VR (I/IV/V/VIII)", df['is_aavr_vr'])]:
    frac_in = df[mask]['frac_named_cluster'].mean()
    frac_out = df[~mask]['frac_named_cluster'].mean()
    print(f"{label}: frac_named_cluster in={frac_in:.3f} (n={mask.sum()}) vs out={frac_out:.3f} (n={(~mask).sum()})")

out_cols = ['abs_pos', 'vr_region', 'is_hspg_residue', 'is_aavr_vr', 'is_hspg_vr',
            'rsa', 'betweenness_mean', 'mean_abs_selection', 'frac_named_cluster', 'n_mutants']
df[out_cols].to_csv(f'{OUT_DIR}/receptor_binding_context.csv', index=False)
print(f"\nWrote {OUT_DIR}/receptor_binding_context.csv")
