# CASCADE: A Composable Analytical System of Chiplets for AI Devices at the Edge
## Democratizing Customization for ML at the Edge through Hetero-Chiplet SiP Architectures

**Authors:** Matthew Joseph Adiletta, Gu-Yeon Wei, and David Brooks  
**Affiliation:** Harvard University, Paulson School of Engineering and Applied Sciences  
**Publication:** [IEEE Journal on Emerging and Selected Topics in Circuits and Systems (JETCAS), 2025](https://ieeexplore.ieee.org/document/11096615)

---

## Overview

**CASCADE** is an early **macro-architecture Design Space Exploration (DSE)** framework for *System-in-Package (SiP)* architectures built from heterogeneous compute **chiplets**. It enables rapid evaluation of chiplet compositions for machine learning workloads by combining analytical roofline models with a dynamic kernel-mapping optimizer.

The framework models **composable chiplets**—including GPU, Convolution, Sparse, and Attention accelerators—each representing distinct computational motifs critical to modern ML applications such as vision, language, graph, and recommendation models.

CASCADE captures the **performance, bandwidth, and energy trade-offs** in multi-chiplet systems, demonstrating that hetero-chiplet SiPs can achieve *3–5× performance speedups* over homogeneous GPU-only baselines, depending on the workload.

---

## Features

- **Analytical chiplet performance models**  
  Compute and data-movement models based on kernel specific roofline bounds calibrated to real hardware.

- **Composable chiplet menu**  
  GPU, Convolution, Sparse, and Attention chiplets, each extending a shared analytical base model.

- **Dynamic kernel mapping engine**  
  Automatically optimizes workload partitioning across heterogeneous chiplets using constrained optimization.

- **Fast macro-architecture DSE**  
  Enables rapid evaluation of design points—orders of magnitude faster than cycle-accurate simulation.

- **Support for multi-tenant edge workloads**  
  Evaluate single-, dual-, and multi-tenant chiplet configurations for edge AI deployments.

---

## Installation

CASCADE uses **Python 3.10+** and common scientific libraries.

```bash
# 1. Create and activate the environment
conda create -n cascade python=3.10
conda activate cascade

# 2. Install dependencies
pip install matplotlib scipy pandas tqdm
```

---

## Running Experiments

All experiments are launched from the `dse/` directory.

```bash
cd dse
./run.sh
```

### `run.sh` Overview

```bash
WORKSPACE=../
TRACE_DIR=$WORKSPACE/traces
CHIPLET_LIBRARY=$WORKSPACE/dse/chiplet-library

# Example experiment configuration (change as needed)
EXPERIMENT_DIR=$WORKSPACE/dse/experiments/gpt-j-1024-weighted.json
# EXPERIMENT_DIR=$WORKSPACE/dse/experiments/resnet50-test.json
# EXPERIMENT_DIR=$WORKSPACE/dse/experiments/ogbn-products-test.json
# EXPERIMENT_DIR=$WORKSPACE/dse/experiments/sd-test.json

python test_system_eval.py \
    --chiplet-library=$CHIPLET_LIBRARY \
    --trace-dir=$TRACE_DIR \
    --experiment=$EXPERIMENT_DIR
```

Each experiment JSON defines:
- **Workload traces** (from ML models like GPT-J, ResNet50, Stable Diffusion, OGBN-Products)
- **Target design parameters** (number of chiplets, bandwidth, power, etc.)

---

## Configuring Chiplet Selections

Chiplet types are set in the Python configuration file (e.g., `test_system_eval.py`):

```python
# Hyperparameters for chiplet selection
# Specify a hub-and-spoke system with 6 GPU chiplets and 6 Sparse Chiplets
GPU     = 6
ATTEN   = 0
SPARSE  = 6
CONV    = 0
```

Adjust these to control the composition of your SiP system.

Each chiplet class in `chiplet-library/` extends a `base_chiplet` model implementing:
- Compute performance via roofline analysis
- Memory and interconnect bandwidth models
- Optimization routines for kernel sharding and workload mapping

---

## Repository Structure

```
CASCADE/
│
├── dse/
│   ├── run.sh                     # main entry point for experiments
│   ├── test_system_eval.py        # core evaluation script
│   ├── chiplet-library/           # analytical chiplet models
│   └── experiments/               # experiment configurations (JSON)
│
├── traces/                        # workload traces (LLM, StableDiffusion, GCN, ResNet, etc.)
└── README.md                      # this file
```

---

## Example: Running GPT-J Weighted Workload

```bash
cd dse
./run.sh
```

This will evaluate a **hetero-chiplet SiP** for the GPT-J model (prefill and decode phases), automatically optimizing workload partitioning across GPU, Attention, Sparse, and Convolution chiplets.

---

## Reference

If you use CASCADE or build upon this framework, please cite our IEEE JETCAS 2025 paper:

> M. J. Adiletta, G.-Y. Wei, and D. Brooks,  
> “Democratizing Customization for ML at the Edge through Hetero-Chiplet SiP Architectures,”  
> *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*, 2025.  
> doi: [10.1109/JETCAS.2025.3592677](https://doi.org/10.1109/JETCAS.2025.3592677)

### BibTeX
```bibtex
@ARTICLE{11096615,
  author={Adiletta, Matthew Joseph and Wei, Gu-Yeon and Brooks, David},
  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems}, 
  title={Democratizing Customization for ML at the Edge through Hetero-Chiplet SiP Architectures}, 
  year={2025},
  pages={1-1},
  doi={10.1109/JETCAS.2025.3592677},
  keywords={Hetero-Chiplet SiP; Bespoke Edge-Device; Design Space Exploration; Chiplet Ecosystem; Machine Learning; Analytical models}
}
```

---

## License and Usage

This repository is provided for **research and educational use** in alignment with IEEE publication policies.  
All rights to the underlying models, figures, and results are reserved © 2025 IEEE and the authors.

---
