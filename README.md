# Spectral Exponents of the Twelve-Lead ECG

**Spectral Exponents of the Twelve-Lead ECG Reveal the Anatomy of Cardiac Conduction Disorders and a Bifurcation Between Aging and Disease**

Theodor Spiro | [ORCID](https://orcid.org/0009-0004-5382-9346)

## Summary

We apply IRASA (Irregular-Resampling Auto-Spectral Analysis) to **412,730 twelve-lead ECG recordings** across three datasets spanning three continents to extract the spectral exponent β — a single number per lead quantifying the balance between order and disorder in the cardiac electrical signal.

### Key findings

1. **Diagnostic specificity** — Each cardiac diagnostic category has a characteristic β-profile. Conduction disturbances show the strongest divergence from normal (Cohen's d = 0.99).

2. **Spectral anatomy** — The 12-lead β-vector encodes conduction system anatomy. A classifier distinguishes complete left from complete right bundle branch block with AUC = 0.982.

3. **Cross-population invariance** — The CLBBB vs CRBBB AUC replicates identically across three continents (0.982 Germany, 0.982 China, 0.979 Brazil), despite factor-of-two variation in absolute β-values across equipment.

4. **Aging bifurcation** — Healthy aging flattens the spectrum (β↓, ρ = −0.179) while disease steepens it (β↑), creating a bifurcation from the healthy operating point.

5. **Honest null result** — β does not independently predict mortality (adjusted HR = 1.02, p = 0.83). It is a diagnostic marker of conduction anatomy, not a prognostic biomarker.

## Datasets

| Dataset | Country | N | Sampling Rate |
|---------|---------|---|---------------|
| [PTB-XL](https://physionet.org/content/ptb-xl/) | Germany | 21,799 | 500 Hz |
| [Chapman-Shaoxing](https://physionet.org/content/ecg-arrhythmia/) | China | 45,152 | 500 Hz |
| [CODE-15%](https://zenodo.org/records/4916206) | Brazil | 345,779 | 400 Hz |

## Repository structure

```
├── paper/
│   ├── main.tex                        # Preprint manuscript
│   └── figures/                        # Publication figures (PDF + PNG)
├── kaggle_notebook.ipynb               # Main analysis notebook
├── criticality_analysis.py             # Core IRASA pipeline
├── diagnostic_classification.py        # CD subtype classification
├── niche_analysis.py                   # Subclinical detection + spatial fingerprints
├── aging_analysis.py                   # Aging trajectories + breakpoint analysis
├── external_validation_chapman.py      # Chapman-Shaoxing replication
├── external_validation_code15.py       # CODE-15% replication + mortality
├── generate_paper_figures.py           # Figure generation (Figures 1-5)
├── generate_supplementary.py           # Supplementary materials
└── results/                            # Computed β-features and figures
```

## Reproducing the analysis

1. Download datasets (PTB-XL from PhysioNet, Chapman-Shaoxing from PhysioNet, CODE-15% from Zenodo)
2. Run `criticality_analysis.py` to extract β-features from PTB-XL
3. Run `external_validation_chapman.py` and `external_validation_code15.py` for replication
4. Run `generate_paper_figures.py` to generate all main figures

## Citation

If you use this work, please cite the preprint (link forthcoming).

## License

MIT
