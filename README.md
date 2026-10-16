# AAV2 capsid: mutation-level PCovC analysis

Reproduction and extension of Ogden et al. (2019, *Science*), "Comprehensive AAV capsid fitness
landscape reveals a viral gene and enables machine-guided design". The analysis looks at which
capsid mutations drive tissue tropism in the paper's mouse biodistribution screen (2,007
mutants, log2 selection in blood, heart, kidney, lung, liver and spleen), using Principal
Covariates Classification (PCovC) alongside PCA, with bootstrap uncertainty throughout.

Highlights (full write-up in `aav_pcovc_findings.pdf`):

- Positions at the 3-fold symmetry axis are enriched for strong selection (mean |log2 selection|
  1.114 vs. 0.977, Mann-Whitney p = 2.9e-6), in line with the paper's description of tropism
  patches near that axis.
- Positions at the 5-fold axis show the opposite pattern (0.652 vs. 1.035, p = 0.014).
- Mutation-level features (variable region, physicochemical change, network centrality, solvent
  accessibility, receptor-associated residues) are tested against tissue selection with
  bootstrapped OLS and Spearman correlation.

## Layout

| Path | What |
|---|---|
| `figures/` | executed notebooks reproducing the paper's figures 1 to 3 and the MAAP analysis |
| `pcovc_analysis/` | PCovC, mutation-feature, structural and network-centrality analyses |
| `report_figures/`, `aav_pcovc_findings.tex/.pdf` | the write-up |

## Running

Clone the authors' repository into `./repo` and run scripts from this directory:

```
git clone https://github.com/churchlab/AAV_fitness_landscape repo
python pcovc_analysis/pcovc_deep_dive.py
```

Data and the original analysis code are from the authors' public repository. Structure: PDB
1LP3 (Xie et al. 2002).
