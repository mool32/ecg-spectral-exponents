[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19945065.svg)](https://doi.org/10.5281/zenodo.19945065)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Preprint](https://img.shields.io/badge/Preprint-bioRxiv-red)](https://www.biorxiv.org/)

# Spectral Exponents of the Twelve-Lead ECG

**Spectral Exponents of the Twelve-Lead ECG Reveal the Anatomy of Cardiac Conduction Disorders and a Bifurcation Between Aging and Disease**

Theodor Spiro | [ORCID 0009-0004-5382-9346](https://orcid.org/0009-0004-5382-9346) | tspiro@vaika.org

📄 **Preprint:** [`paper/main.pdf`](paper/main.pdf) — bioRxiv submission ID BIORXIV/2026/717157
🧮 **Main analysis script:** [`generate_paper_figures.py`](generate_paper_figures.py)
📊 **Kaggle notebook (reproducible):** [The Heartbeat at the Edge of Chaos](https://www.kaggle.com/code/theodorspiro/the-heartbeat-at-the-edge-of-chaos)
📦 **Archived release (Zenodo DOI):** [10.5281/zenodo.19945065](https://doi.org/10.5281/zenodo.19945065)

---

## Brief Summary

We extract the spectral exponent **β** — the slope of the aperiodic (1/f^β) component of the ECG power spectrum — from **412,730 twelve-lead recordings across three continents**. The analysis demonstrates that:

1. **β is diagnostic.** Each cardiac diagnostic category produces a characteristic β-profile; conduction disturbances show the strongest divergence from healthy controls (Cohen's *d* = 0.99).
2. **β encodes conduction anatomy.** A classifier distinguishes complete left from complete right bundle branch block with **AUC = 0.982** on PTB-XL (Germany).
3. **The result replicates exactly across populations.** AUC = 0.982 on Chapman-Shaoxing (China), AUC = 0.979 on CODE-15% (Brazil) — despite factor-of-two variation in absolute β-values across recording equipment.
4. **β captures information beyond standard morphology.** For CLBBB vs CRBBB classification, QRS duration alone gives AUC = 0.688; β-vector gives AUC = 0.988 (ΔAUC = +0.300).
5. **Aging and disease drive β in opposite directions.** Healthy aging flattens the spectrum (ρ = −0.179), overt conduction disease steepens it — creating a bifurcation from the healthy operating point.
6. **Honest null result.** β does not independently predict mortality after full adjustment (HR = 1.02, *p* = 0.83). It is a diagnostic marker of conduction anatomy, not a prognostic biomarker.

The geometry of the twelve-lead β-vector is determined by conduction system anatomy and is reproducible across equipment, ethnicity, and geography — a cross-population invariant of cardiac electrophysiology.

## Datasets

| Dataset | Country | N | Sampling Rate |
|---------|---------|------|---------------|
| [PTB-XL](https://physionet.org/content/ptb-xl/) | Germany | 21,799 | 500 Hz |
| [Chapman-Shaoxing](https://physionet.org/content/ecg-arrhythmia/) | China | 45,152 | 500 Hz |
| [CODE-15%](https://zenodo.org/records/4916206) | Brazil | 345,779 | 400 Hz |

## Repository structure

```
├── paper/
│   ├── main.tex                        # Preprint manuscript (LaTeX source)
│   ├── main.pdf                        # Compiled preprint
│   ├── cover_letter_elife.tex/pdf      # Cover letter for journal submission
│   └── figures/                        # All publication figures (PDF + PNG)
├── generate_paper_figures.py           # Main analysis: generates Figures 1–5
├── generate_supplementary.py           # Supplementary tables and figures
├── kaggle_notebook.ipynb               # Reproducible analysis notebook
├── criticality_analysis.py             # Core IRASA pipeline (PTB-XL)
├── diagnostic_classification.py        # CD subtype classification
├── niche_analysis.py                   # Subclinical detection + spatial fingerprints
├── aging_analysis.py                   # Aging trajectories + breakpoint analysis
├── external_validation_chapman.py      # Chapman-Shaoxing replication
├── external_validation_code15.py       # CODE-15% replication + mortality
└── results/                            # Cached β-features and intermediate figures
    ├── beta_features.csv               # PTB-XL β-features (all 21,799 records)
    ├── chapman_beta_features.csv       # Chapman β-features
    └── code15_beta_features.csv        # CODE-15% β-features
```

## Reproducing the analysis

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib wfdb lifelines

# 2. Download datasets:
#    - PTB-XL → ptb-xl/
#    - Chapman-Shaoxing → chapman-shaoxing/
#    - CODE-15% → code15/

# 3. Extract β-features for PTB-XL (~30 min)
python criticality_analysis.py

# 4. External validations (~1–2 h each)
python external_validation_chapman.py
python external_validation_code15.py

# 5. Generate all paper figures (uses cached features, ~2 min)
python generate_paper_figures.py
python generate_supplementary.py
```

All key results from the manuscript reproduce exactly — see commit history for verification log.

## Citation

If you use this work, please cite the preprint:

> Spiro, T. (2026). Spectral Exponents of the Twelve-Lead ECG Reveal the Anatomy of Cardiac Conduction Disorders and a Bifurcation Between Aging and Disease. *bioRxiv* [DOI to be added after acceptance].

And the archived code release:

> Spiro, T. (2026). ecg-spectral-exponents (v1.0.0). *Zenodo*. https://doi.org/10.5281/zenodo.19945065

## Contact

Theodor Spiro — tspiro@vaika.org

## License

MIT (see [LICENSE](LICENSE))
