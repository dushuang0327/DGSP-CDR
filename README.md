# DGSP-CDR

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)

## DGSP-CDR: A Drug–Gene Synergistic encoding and Selective Pseudo-labeling framework for Cancer Drug Response prediction

> Official PyTorch implementation (partially released) of **DGSP-CDR**, a cross-domain drug response prediction framework that integrates drug–gene synergistic encoding with dynamic pseudo-labeling.

---

## Highlights

- **Structural-Enhanced Drug Encoding Module**  
  Enhances drug ECFP fingerprints via Single-Fingerprint Multi-Path Encoding and subspace attention.

- **Dual-Branch Variational Encoding Module**  
  Extracts both domain-invariant and domain-specific gene features for robust cross-domain generalization.

- **Cross-Modal Attention Fusion Module**  
  Integrates structural and transcriptomic representations through attention-based interaction.

- **Selective Pseudo-Label Enhancement Module**  
  Employs a dynamic iterative pseudo-labeling strategy with neighborhood consistency filtering.

---

## Acknowledgement

This codebase is partially adapted from:

1. https://github.com/XieResearchGroup/CODE-AE  
2. https://github.com/hunterlang/weaksup-subset-selection  

We thank the original authors for their open-source contributions.

---

## Architecture

![architecture](./images/arch.png?raw=true)

---

## Overview

DGSP-CDR is designed to predict cancer drug responses using both in vitro and in vivo data. The model learns drug–gene interactions by fusing fingerprint-based drug embeddings and gene expression representations. To address the lack of labeled patient data, we introduce a **dynamic iterative pseudo-labeling strategy**, progressively incorporating confident samples to enhance learning.

---

## Installation

1. Install Anaconda:  
   [https://www.anaconda.com/download](https://www.anaconda.com/download)

2. Create environment and install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download benchmark dataset (CODE-AE v2.0) from:  
   [https://doi.org/10.5281/zenodo.4776448](https://doi.org/10.5281/zenodo.4776448)

4. Modify `config/data_config.py` to set your local dataset directory.

5. Run the main script:
   ```bash
   python main.py
