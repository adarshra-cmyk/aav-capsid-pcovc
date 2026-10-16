"""
Residue-level test of the chain: VR -> receptor footprint -> organ.

Question: within the variable regions, do the specific residues that the
literature assigns to a receptor footprint carry selection (overall and per
organ) that the VR label alone does not explain?

Receptor residue sets (VP1 numbering), primary sources:
  HSPG (5 residues): R484, R487, K532, R585, R588.
     Opie et al. 2003 J Virol 77:6995; Kern et al. 2003 J Virol 77:11072.
     R585 or R588 mutation eliminates heparin-agarose binding; R484, R487,
     K532 mutants keep partial binding. K527 is NOT included: Opie et al.
     state it does not appear to be involved in heparin binding.
  AAVR (19 residues within 4.5 A of AAVR PKD2 in the AAV2-AAVR structure):
     Q263 S264 G265 A266 S267 N268 H271 N382 G383 S384 Q385 R471 W502 T503
     D528 D529 Q589 K706 V708.
     Meyer et al. 2019 eLife 8:e44707, Table 2 (text: "VR-I and III are at
     the core of the AAVR footprint").

Two tests:
  1. Position level: mean |selection| (and mean signed selection per organ)
     for receptor residues vs. residues drawn at random from the SAME VRs
     (stratified permutation, 20000 draws). Integer positions, all mutant types.
  2. Mutant level: the organ-level OLS from pcovc_organ_ols_bootstrap.py with
     two added 0/1 indicators (HSPG residue, AAVR residue). Bootstrap resamples
     POSITIONS (all mutants at a position move together), 1000 draws.

Run with cwd = <repo>/x02_analysis, venv_pcovc/bin/python3.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, './pcovc_analysis')
from pcovc_biodistribution import build_mouse_pca_input
from pcovc_mutation_organ_correlation_expanded import select_low_redundancy_properties, N_CLUSTERS
from pcovc_mutation_features import vr_region, AA_ORDER

OUT = './pcovc_analysis/deep_dive'
TISSUES = ['blood', 'heart', 'kidney', 'lung', 'liver', 'spleen']
HSPG = [484, 487, 532, 585, 588]
AAVR = [263, 264, 265, 266, 267, 268, 271, 382, 383, 384, 385, 471, 502, 503, 528, 529, 589, 706, 708]
N_BOOT, N_PERM, RNG = 1000, 20000, 0


def build():
    X = build_mouse_pca_input()
    aa_index = pd.read_csv('../data/meta/aa_index_0_center_norm_df.csv', index_col=0)
    aa_index.index = AA_ORDER
    reps, _ = select_low_redundancy_properties(aa_index, n_clusters=N_CLUSTERS)
    props = aa_index[reps].copy()
    props['net_charge'] = aa_index['Positive_charge'] - aa_index['Negative_charge']
    lookup = pd.read_csv('../data/meta/AAV2scan_chip_lookup_table.txt')
    wt = lookup[lookup['is_wt_aa'] == 1].set_index('abs_pos')['aa'].to_dict()
    rows = []
    for abs_pos, aa, wt_bc, lib_type in X.index:
        row = {'abs_pos': abs_pos, 'vr_region': vr_region(abs_pos), 'lib_type': lib_type}
        w = wt.get(abs_pos)
        for col in props.columns:
            row[f'{col}_delta'] = (props.loc[aa, col] - props.loc[w, col]) if (aa in props.index and w in props.index) else 0.0
        rows.append(row)
    f = pd.DataFrame(rows)
    f[TISSUES] = X[TISSUES].to_numpy()
    f['is_int'] = f.abs_pos % 1 == 0
    f['hspg'] = (f.is_int & f.abs_pos.isin(HSPG)).astype(float)
    f['aavr'] = (f.is_int & f.abs_pos.isin(AAVR)).astype(float)
    f['absmean'] = f[TISSUES].abs().mean(axis=1)
    return f, [c for c in f.columns if c.endswith('_delta')]


def position_test(f):
    """Stratified permutation: receptor residues vs random residues from the same VR strata."""
    rng = np.random.default_rng(RNG)
    pos = f[f.is_int].groupby('abs_pos').agg(vr=('vr_region', 'first'), n=('absmean', 'size'),
                                              absmean=('absmean', 'mean'),
                                              **{t: (t, 'mean') for t in TISSUES}).reset_index()
    out = []
    for name, S in [('HSPG', HSPG), ('AAVR', AAVR)]:
        sel = pos[pos.abs_pos.isin(S)]
        # strata = VR label (VR-III / 'none' positions form the 'none' stratum)
        strata = sel.vr.value_counts().to_dict()
        pools = {v: pos[pos.vr == v] for v in strata}
        for metric in ['absmean'] + TISSUES:
            obs = sel[metric].mean()
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                vals = []
                for v, k in strata.items():
                    p = pools[v][metric].to_numpy()
                    vals.append(rng.choice(p, size=min(k, len(p)), replace=False))
                null[i] = np.concatenate(vals).mean()
            p2 = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (N_PERM + 1)
            out.append(dict(set=name, metric=metric, n_pos_with_data=len(sel), n_pos_listed=len(S),
                            observed=obs, vr_matched_null_mean=null.mean(), diff=obs - null.mean(), p=p2))
    return pd.DataFrame(out), pos


def residue_table(f):
    g = f[f.is_int & (f.hspg + f.aavr > 0)]
    t = g.groupby(['abs_pos', 'vr_region']).agg(n=('absmean', 'size'), absmean=('absmean', 'mean'),
                                                **{x: (x, 'mean') for x in TISSUES}).reset_index()
    t['set'] = np.where(t.abs_pos.isin(HSPG), 'HSPG', 'AAVR')
    return t


def cluster_boot_ols(f, delta_cols, extra):
    """OLS per organ; bootstrap over positions."""
    vr_dum = pd.get_dummies(f.vr_region).drop(columns=['none']).astype(float)
    lib = pd.get_dummies(f.lib_type).drop(columns=['sub']).astype(float)
    Xd = f[delta_cols].copy()
    Xd[:] = StandardScaler().fit_transform(Xd)
    cols = pd.concat([Xd, vr_dum.add_prefix('vr_'), lib.add_prefix('lib_'), f[extra]], axis=1)
    names = cols.columns.tolist()
    A = cols.to_numpy(float); Y = f[TISSUES].to_numpy()
    groups = f.abs_pos.to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    pt = {t: LinearRegression().fit(A, Y[:, i]).coef_ for i, t in enumerate(TISSUES)}
    rng = np.random.default_rng(RNG)
    boot = {t: [] for t in TISSUES}
    for _ in range(N_BOOT):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in gs])
        for i, t in enumerate(TISSUES):
            boot[t].append(LinearRegression().fit(A[ii], Y[ii, i]).coef_)
    rows = []
    for t in TISSUES:
        a = np.array(boot[t])
        for j, n in enumerate(names):
            lo, hi = np.percentile(a[:, j], [2.5, 97.5])
            rows.append(dict(organ=t, feature=n, point=pt[t][j], boot_mean=a[:, j].mean(), lo=lo, hi=hi,
                             stable=(lo > 0) or (hi < 0)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    f, delta_cols = build()
    print('mutants', len(f), '| HSPG mutants', int(f.hspg.sum()), '| AAVR mutants', int(f.aavr.sum()))
    print(f[f.hspg + f.aavr > 0].groupby('abs_pos').size().to_dict())
    pt, pos = position_test(f)
    pt.to_csv(f'{OUT}/receptor_residue_position_permutation.csv', index=False)
    residue_table(f).to_csv(f'{OUT}/receptor_residue_table.csv', index=False)
    print(pt.round(3).to_string())
    # model A: same as report OLS + receptor indicators, position-cluster bootstrap
    A = cluster_boot_ols(f, delta_cols, ['hspg', 'aavr'])
    A.to_csv(f'{OUT}/receptor_residue_ols_position_bootstrap.csv', index=False)
    # model 0: report OLS spec without receptor indicators, position-cluster bootstrap (for VR coefficient comparison)
    B = cluster_boot_ols(f, delta_cols, [])
    B.to_csv(f'{OUT}/vr_ols_position_bootstrap_no_receptor_terms.csv', index=False)
    print('done')
