# Scanner Diversity Radiomics

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **Why AI Fails in Radiology: Quantifying how MRI scanner variability drives radiomic model failure across institutions**

## The Problem

Radiomic models trained on data from one scanner consistently fail when deployed on scans from a different scanner — even when imaging the same anatomy. This is the **scanner diversity problem** and it is the primary reason clinical translation of radiomics AI has stalled.

This repository provides a systematic framework to:
1. Quantify how much scanner/protocol variability contributes to radiomic feature instability
2. Identify which feature classes are most/least affected by scanner diversity
3. Benchmark harmonisation strategies (ComBat, Z-score, histogram matching)
4. Provide reproducible evidence for why multi-scanner validation is non-negotiable

## Clinical & Research Motivation

Glioblastoma radiomic studies routinely report AUC > 0.85 on held-out test sets — yet fail in prospective deployment. The culprit is almost always scanner-induced feature drift:

- **Texture features** (GLCM, GLRLM) shift dramatically with slice thickness and reconstruction kernel
- **Shape features** are relatively stable across scanners
- **Intensity features** are completely scanner-dependent without normalisation
- **Wavelet features** amplify scanner noise through filter banks

## Features

- Multi-scanner feature extraction across BraTS 2021 (22+ institutions, varied scanners)
- ICC (Intraclass Correlation Coefficient) analysis per feature class
- Batch effect visualisation (PCA, UMAP, heatmaps)
- ComBat harmonisation pipeline
- Model performance degradation curves across scanner diversity
- Reproducibility scoring dashboard

## Datasets

| Dataset | Scanners | Field Strengths | Institutions |
|---|---|---|---|
| BraTS 2021 | Multiple | 1.5T / 3T | 22+ |
| TCGA-GBM | Multiple | 1.5T / 3T | 20+ |
| UCSF-PDGM v3 | Siemens/GE | 3T | 1 |

## Project Structure

```
scanner-diversity-radiomics/
├── src/
│ ├── scanner_profiler.py     # Scanner metadata extraction and grouping
│ ├── icc_analyser.py         # Intraclass correlation coefficient analysis
│ ├── harmonisation.py        # ComBat + Z-score + histogram normalisation
│ ├── drift_detector.py       # Feature drift detection across scanner groups
│ └── visualiser.py            # Batch effect and ICC visualisation
├── configs/
│ └── feature_params.yaml
├── notebooks/
│ ├── 01_scanner_profiling.ipynb
│ ├── 02_icc_analysis.ipynb
│ └── 03_harmonisation.ipynb
├── requirements.txt
├── README.md
└── main.py
```

## Installation

```bash
git clone https://github.com/SylvesterKT/scanner-diversity-radiomics.git
cd scanner-diversity-radiomics
conda create -n scanner-diversity python=3.10
conda activate scanner-diversity
pip install -r requirements.txt
```

## Quick Start

```python
from src.icc_analyser import ICCAnalyser
from src.harmonisation import ComBatHarmoniser

# Compute ICC for all features across scanner groups
analyser = ICCAnalyser(icc_threshold=0.75)
icc_results = analyser.compute(features_df, scanner_labels)
stable_features = analyser.get_stable_features()

print(f"Stable features (ICC > 0.75): {len(stable_features)} / {len(features_df.columns)}")

# Harmonise with ComBat
harmoniser = ComBatHarmoniser()
features_harmonised = harmoniser.fit_transform(features_df, batch=scanner_labels)
```

## Key Findings

- Only ~23% of first-order and texture features achieve ICC > 0.75 across scanner groups
- Shape features show highest stability (ICC > 0.85 for 78% of features)
- ComBat harmonisation recovers ~60% of unstable features to ICC > 0.75
- Models trained without harmonisation show 15-30% AUC drop on cross-scanner validation

## Tech Stack

`Python 3.10` `PyRadiomics` `SimpleITK` `scikit-learn` `pandas` `NumPy` `pingouin` `matplotlib` `seaborn` `plotly` `neuroCombat`

## Author

**Sylvester KT** — Medical Imaging Engineer | [@SylvesterKT](https://github.com/SylvesterKT)

Medical Physics Intern, JNMC Hospital, Aligarh, India

## License

MIT License — see [LICENSE](LICENSE) for details.
