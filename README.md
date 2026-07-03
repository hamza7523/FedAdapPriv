# FedAdaPriv-CPU

**Adaptive Differential Privacy with Frozen Backbone for Resource-Constrained Federated Medical Imaging**

Final Year Project — IBA Karachi
Authors: Hamza, Anusha Randhawa
Supervisor: Dr. Faisal Iradat

Presented as a poster at **AI4X – Accelerate 2026** (NUS & University of Toronto, Singapore).

---

## Overview

FedAdaPriv-CPU is a federated learning framework for chest X-ray classification across simulated, non-IID hospital clients, designed to run entirely on **CPU hardware**. It tackles a core problem in federated differential privacy: fixed, uniform privacy budgets penalize clients with harder or smaller minority-class distributions, causing severe underperformance on rare classes (e.g., Viral Pneumonia).

The framework introduces a **three-signal adaptive weighting scheme** that controls how much each client contributes to the global model at every round, combined with an **epsilon-coupled aggregation** strategy that ties a client's contribution weight to both its data volume and its allotted privacy budget.

### Headline result

At a global privacy budget of **ε = 3.0**:

| Metric | Fixed DP-FedAvg | FedAdaPriv-CPU |
|---|---|---|
| Viral Pneumonia F1 | 0.420 | **0.722** |
| Macro F1 | — | **0.810** |

## Core Idea: The Three Signals

Each client's per-round aggregation weight is computed as:

```
weight(k, r) = α_vol(k) × α_conv(k, r) × α_round(r)
```

| Signal | Formula | What it captures |
|---|---|---|
| **α_vol** | `log(n_k + 1) / log(n_max + 1)` | Data volume — clients with more data get proportionally more weight |
| **α_conv** | `1 − exp(−λ × deficiency)` | Convergence state — clients that are still learning (high deficiency, esp. on minority classes) get boosted weight |
| **α_round** | `r / R` | Round progression — a curriculum that ramps up client influence as training progresses |

An ablation study confirms **α_conv is the dominant signal**, accounting for roughly 87% of the Viral Pneumonia F1 gain, with α_vol and α_round contributing incrementally.

Privacy budgets are also allocated per client:

```
ε_client = ε_global × α_vol × α_conv × α_round
```

and client aggregation weights are coupled to this allocation (`w_client ∝ n_client × ε_client`) — a departure from prior adaptive DP-FL work, which typically adapts clipping/noise but keeps aggregation weights uniform.

## Repository Contents

```
FedAdapPriv/
├── notebook-final3          # Main experiment notebook (Jupyter/Kaggle, JSON format)
└── notebook-ablation study  # Ablation study & multi-seed validation notebook
```

> **Note:** Both files are Jupyter notebooks stored without the `.ipynb` extension. Rename them to `.ipynb` locally to open in Jupyter/VS Code, or upload directly to Kaggle.

### `notebook-final3` — Main Federated Training Pipeline

Implements the full FedAdaPriv-CPU system:

- **Backbone:** Frozen `TorchXRayVision` DenseNet121 (`densenet121-res224-all`) for feature extraction; only a lightweight classifier head is trained
- **Classifier head:** MLP (1280 → 512 → 128 → 4) with LayerNorm (Opacus-compatible), ~720K parameters
- **Privacy engine:** [Opacus](https://opacus.ai/) with RDP/PRV accountant, DP-Adam, and `BatchMemoryManager`
- **Clients:** 3 simulated hospitals (A, B, C) with non-IID data via constrained Dirichlet partitioning (α = 0.5)
- **Classes:** COVID, Lung Opacity, Normal, Viral Pneumonia

Pipeline stages (by cell):
1. Environment setup, config, and directory creation
2. Imports and reproducibility seeding
3. Feature/label loading and class-weight computation
4. MLP classifier definition
5. Deficiency vector computation (drives α_conv)
6. Three-signal adaptive weighting functions
7. Evaluation utilities, focal loss, weighted data loaders
8. Non-private warmstart federated training
9. Main adaptive DP federated training loop
10. Fixed DP-FedAvg baseline (for comparison)
11. Multi-seed statistical significance runs (Dirichlet seeds 0, 7, 42, 123, 999)
12. RDP/formal privacy accounting and threat model documentation
13. Result plots: per-class F1, macro F1 trajectories, confusion matrices, per-client ε spend, ε-sweep comparison

### `notebook-ablation study` — Ablation & Sensitivity Analysis

Isolates the individual contribution of each of the three signals (α_vol, α_conv, α_round) to final model performance, using the same client/feature setup as the main notebook. This is the authoritative source for the ablation results reported in the paper (isolating each signal's contribution to the Viral Pneumonia F1 gain).

## Key Configuration

| Parameter | Value |
|---|---|
| Rounds | 30 |
| Clients | 3 |
| Local epochs | 5 |
| Feature dimension | 1024 |
| Batch size (logical / physical) | 256 / 64 |
| Optimizer | Adam, lr = 2e-4 |
| Global ε | 3.0 |
| δ | 1e-5 |
| λ (α_conv sharpness) | 3.0 |

## Requirements

- Python 3.x, PyTorch
- [Opacus](https://opacus.ai/) (DP-SGD/DP-Adam, RDP accountant, `BatchMemoryManager`)
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) (pretrained DenseNet121 backbone)
- scikit-learn, matplotlib, seaborn, numpy

```bash
pip install opacus torchxrayvision torch scikit-learn matplotlib seaborn numpy
```

## Data

Trained and evaluated on the **[COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)**, partitioned into 3 simulated hospital clients using constrained Dirichlet sampling (α = 0.5) to create realistic non-IID class distributions. Feature extraction and client partitioning are performed in a separate notebook (Notebook 1, not included in this repo) and cached to disk before these notebooks run.

## Citation

If you use this work, please cite:

> Hamza, A. Randhawa, F. Iradat. *FedAdaPriv-CPU: Adaptive Differential Privacy with Frozen Backbone for Resource-Constrained Federated Medical Imaging.* Presented at AI4X – Accelerate 2026, Singapore.

## Acknowledgments

Institute of Business Administration (IBA), Karachi.
