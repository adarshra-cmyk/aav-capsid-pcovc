"""
Figures and one extra table for the receptor-residue section of the report.
Reads deep_dive/receptor_residue_*.csv (from pcovc_receptor_residue_test.py) and
recomputes the VR-VIII per-position substitution profile (saved to
deep_dive/vr8_position_profile.csv, plus non-VR substitution baseline).

Run with cwd = <repo>/x02_analysis, venv_pcovc/bin/python3.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, './pcovc_analysis')
from pcovc_receptor_residue_test import build, TISSUES, HSPG

DD = './pcovc_analysis/deep_dive/'
FIG = './report_figures/'
plt.rcParams.update({'font.size': 9})

# ---- VR-VIII per-position profile (substitutions only) + baseline ----
f, _ = build()
g = f[f.is_int & (f.lib_type == 'sub')]
prof = g[(g.abs_pos >= 579) & (g.abs_pos <= 594)].groupby('abs_pos').agg(
    n=('absmean', 'size'), **{t: (t, 'mean') for t in TISSUES}).reset_index()
prof.to_csv(DD + 'vr8_position_profile.csv', index=False)
base = g[g.vr_region == 'none'][TISSUES].mean()
base.to_csv(DD + 'non_vr_substitution_baseline.csv', header=['mean_log2_selection'])

# ---- Fig A: residue x organ heatmap ----
t = pd.read_csv(DD + 'receptor_residue_table.csv')
t['set'] = pd.Categorical(t['set'], ['HSPG', 'AAVR'])
t = t.sort_values(['set', 'abs_pos']).reset_index(drop=True)
lab = [f"{int(r.abs_pos)} {r['set']} ({'no VR' if r.vr_region == 'none' else r.vr_region}, n={int(r.n)})" for _, r in t.iterrows()]
M = t[TISSUES].to_numpy()
fig, ax = plt.subplots(figsize=(7.2, 8))
im = ax.imshow(M, cmap='RdBu', vmin=-3.5, vmax=3.5, aspect='auto')
ax.set_xticks(range(6)); ax.set_xticklabels(TISSUES)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(M.shape[0]):
    for j in range(6):
        ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=7, color='white' if abs(M[i, j]) > 2.2 else 'black')
nh = int((t['set'] == 'HSPG').sum())
ax.axhline(nh - 0.5, color='black', lw=1.5)
ax.set_title('Mean signed log2 selection at receptor-associated residues\n(all mutant types at each integer position)')
fig.colorbar(im, ax=ax, shrink=0.6, label='mean log2 selection')
fig.tight_layout(); fig.savefig(FIG + 'receptor_residue_heatmap.png', dpi=200); plt.close(fig)

# ---- Fig B: indicator coefficients with position-bootstrap intervals ----
A = pd.read_csv(DD + 'receptor_residue_ols_position_bootstrap.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, feat, title in [(axes[0], 'hspg', 'HSPG residues (5 positions)'), (axes[1], 'aavr', 'AAVR contact residues (16 positions with data)')]:
    r = A[A.feature == feat].set_index('organ').loc[TISSUES]
    x = np.arange(6)
    for i, (o, row) in enumerate(r.iterrows()):
        c = '#1f5fa8' if row.stable else '#8c8c8c'
        ax.plot([i, i], [row.lo, row.hi], color=c, lw=2)
        ax.plot(i, row.boot_mean, 'o', color=c)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TISSUES, rotation=30)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel('coefficient (log2 selection units)')
fig.suptitle('Receptor-residue indicator in the organ-level OLS (bootstrap over positions; blue = interval excludes 0)', fontsize=9)
fig.tight_layout(); fig.savefig(FIG + 'receptor_indicator_coefficients.png', dpi=200); plt.close(fig)

# ---- Fig C: VR-VIII per-position profile ----
P = prof.set_index('abs_pos')[TISSUES]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
im = ax.imshow(P.to_numpy().T, cmap='RdBu', vmin=-3.5, vmax=3.5, aspect='auto')
ax.set_yticks(range(6)); ax.set_yticklabels(TISSUES)
xl = [f"{int(p)}*" if int(p) in HSPG else f"{int(p)}" for p in P.index]
ax.set_xticks(range(len(P))); ax.set_xticklabels(xl, rotation=90)
for j in range(P.shape[0]):
    for i in range(6):
        v = P.iloc[j, i]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6.5, color='white' if abs(v) > 2.2 else 'black')
ax.set_title('VR-VIII (579 to 594): mean log2 selection per position, substitutions only\n* = HSPG residue')
fig.colorbar(im, ax=ax, shrink=0.8, label='mean log2 selection')
fig.tight_layout(); fig.savefig(FIG + 'vr8_position_profile.png', dpi=200); plt.close(fig)
print(base.round(2).to_dict())
print('ok')
