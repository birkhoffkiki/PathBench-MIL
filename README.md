<!-- prettier-ignore -->


<div align="center">
    <img src="https://birkhoffkiki.github.io/PathBench/images/pathbench.svg" width="50%" alt="PathBench" />
</div>
<hr>
<div align="center" style="line-height: 1;">
    <a href="https://pathbench.org" target="_blank"><img alt="Website" src="https://img.shields.io/badge/Live Benchmark-pathbench.org-blue"/></a>
    <a href="https://github.com/birkhoffkiki/PrePATH" target="_blank"><img alt="GitHub" src="https://img.shields.io/badge/PrePATH-Feature Extraction-lightgrey"/></a>
    <a href="LICENSE" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey"/></a>
    <a href="https://arxiv.org/abs/2505.20202" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-f5de53"/></a>
    <br>
</div>

# PathBench-MIL

> Downstream evaluation scripts and benchmark pipelines for PathBench — multiple-instance learning (MIL) based WSI classification and survival prediction.
 

---
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
    - [WSI Classification](#wsi-classification)
    - [Survival Prediction](#survival-prediction)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

PathBench-MIL contains the downstream evaluation code used by the PathBench benchmark. It provides training and evaluation pipelines for:

- WSI classification
- Survival prediction

The repository expects features to be extracted from whole-slide images (WSIs) using a separate extraction pipeline (for example, [PrePATH](https://github.com/birkhoffkiki/PrePATH)). See the Data Preparation section below.

## Features

- Modular training/evaluation scripts for classification and survival tasks
- Example splits and dataset templates
- Simple shell wrappers (`run.sh`) for reproducible experiments

## Repository Layout

```
README.md
classification/
    ├── main.py
    ├── run.sh
    ├── datasets/
    └── models/
    ├── splits/
    └── utils/
survival/
    ├── main_kfold.py
    ├── run.sh
    └── models/
    ├── splits/
    └── utils/
```

## Data Preparation

Recommended workflow:

1. Split each WSI into patches and extract features with a pretrained model (outside this repo, e.g., [PrePATH](https://github.com/birkhoffkiki/PrePATH)).
2. Store features as `.pt` or `.h5` files and organize them under a cohort directory.

Suggested layout:

```
DATA_ROOT/
└── CohortName/
        ├── pt_files/
        │   ├── resnet50/
        │   │   ├── slide_1.pt
        │   │   ├── slide_2.pt
        │   │   └── ...
        │   └── ...
        └── patches/
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
```

- Add dataset paths and labels to `classification/splits/datasets.xlsx` for classification experiments.
- For survival experiments, follow the `survival/splits/example.xlsx` format (time-to-event and event flag columns).

> Note: If you change storage format or file layout, please update the corresponding data loader in `datasets/`.

## Quick Start

We recommend using conda with Python 3.10 and installing dependencies from `requirements.txt` for reproducible environments. 

```bash
# create and activate a conda environment with Python 3.10
conda create -n pathbench python=3.10 -y
conda activate pathbench

# install from requirements.txt (preferred)
pip install --upgrade pip
pip install -r requirements.txt
```

### WSI Classification

1. Prepare feature files and update `classification/splits/datasets.xlsx`.
2. Edit `classification/run.sh` to set dataset root, hyperparameters and other options.
3. Run the training/benchmark:

```bash
cd classification
bash run.sh
```

### Survival Prediction

1. Prepare an Excel with survival info following `survival/splits/example.xlsx`.
2. Edit `survival/run.sh` (paths, hyperparameters).
3. Run:

```bash
cd survival
bash run.sh
```

## Citation

If you use this code or the benchmark in your research, please cite:

```bibtex
@article{ma2025pathbench,
    title={PathBench: A comprehensive comparison benchmark for pathology foundation models towards precision oncology},
    author={Ma, Jiabo and Xu, Yingxue and Zhou, Fengtao and Wang, Yihui and Jin, Cheng and Guo, Zhengrui and Wu, Jianfeng and Tang, On Ki and Zhou, Huajun and Wang, Xi and others},
    journal={arXiv preprint arXiv:2505.20202},
    year={2025}
}
```

## How to evaluate new models
To evaluate a new MIL model using this benchmark, please visit the [PrePATH](https://github.com/birkhoffkiki/PrePATH) repository.


## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

See the full license text in the `LICENSE` file at the repository root or online:

https://creativecommons.org/licenses/by-nc/4.0/

---

