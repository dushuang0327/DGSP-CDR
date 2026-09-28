# DGSP-CDR

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)

## DGSP-CDR: A Drug–Gene Synergistic Encoding and Selective Pseudo-Labeling Framework for Cancer Drug Response Prediction

> Official PyTorch implementation of **DGSP-CDR**, a cross-domain cancer drug response prediction framework that integrates drug–gene synergistic representation learning with selective iterative pseudo-labeling.

---

## Highlights

- **Structural-Enhanced Drug Encoding Module**  
  Enhances drug ECFP representations using a Single-Fingerprint Multi-Path Encoder and subspace attention.

- **Dual-Branch Variational Encoding Module**  
  Extracts shared domain-invariant and private domain-specific gene expression representations for cross-domain transfer.

- **Cross-Modal Attention Fusion Module**  
  Integrates drug structural representations and gene expression representations through attention-based interaction.

- **Selective Pseudo-Label Enhancement Module**  
  Generates provisional response labels for unlabeled TCGA patients using an ensemble of classifiers trained on labeled cell-line data, and progressively selects reliable pseudo-labeled samples through confidence filtering, ensemble voting, neighborhood consistency, and iterative retraining.

---

## Acknowledgement

This codebase is partially adapted from:

1. [CODE-AE](https://github.com/XieResearchGroup/CODE-AE)
2. [Weakly Supervised Subset Selection](https://github.com/hunterlang/weaksup-subset-selection)

We thank the original authors for their open-source contributions.

---

## Architecture

![architecture](./images/arch.png?raw=true)

---

## Overview

DGSP-CDR is designed for cross-domain cancer drug response prediction from cancer cell lines to patients.

The framework jointly models drug molecular structure and gene expression information. Drug structures are represented using ECFP fingerprints and enhanced through multi-path structural encoding and subspace attention. Gene expression profiles from cell lines and patients are encoded using a dual-branch variational representation module containing shared and private encoders. The resulting drug and gene representations are integrated through cross-modal attention.

Because labeled patient drug-response data are limited, DGSP-CDR further introduces a selective pseudo-labeling procedure. In this framework, pseudo-labels are provisional drug-response labels assigned to originally unlabeled TCGA patient samples using an ensemble of classifiers trained on labeled cell-line data. Candidate pseudo-labeled patients are progressively filtered according to prediction confidence, ensemble agreement, and neighborhood consistency in the learned representation space. Selected patient samples are subsequently incorporated into the augmented training set for iterative model refinement.

---

## Create the Environment

Create the Conda environment using:

```bash
conda env create -f environment.yml
conda activate dgsp-cdr

## Dataset

The experiments use the CODE-AE v2.0 dataset:

https://doi.org/10.5281/zenodo.4776448

The dataset directory can be specified using the `DGSP_DATA_DIR` environment variable.

Linux/macOS:

```bash
export DGSP_DATA_DIR=/path/to/data
```

Windows CMD:

```cmd
set DGSP_DATA_DIR=D:\path\to\data
```

Windows PowerShell:

```powershell
$env:DGSP_DATA_DIR="D:\path\to\data"
```

If `DGSP_DATA_DIR` is not specified, the code uses the `data/` directory under the project root.

---

## Reference Results

Reference five-fold results for all five drugs are provided in `logs/explain_2/code_adv_norm/`.

AUROC and AUPRC are reported as the mean ± sample standard deviation across the five folds:

```python
mean = np.mean(values)
std = np.std(values, ddof=1)
```
