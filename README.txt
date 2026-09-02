# Understanding Graph Geodesic Distance: Structure, Bounds, and Network Comparison

This repository contains code, data, numerical experiments, and figure-generation notebooks for computing and analyzing the **Graph Geodesic Distance (GGD)** between graphs, as proposed in the paper:

Understanding Graph Geodesic Distance: Structure, Bounds, and Network Comparison

This implementation is written in Python 3.

The central implementation is provided in `GGD.py`. Graph instances are stored as Python pickle files, calculation outputs are serialized to disk, and Jupyter notebooks reproduce the figures from the saved results.

## Repository structure

```text
.
├── data/
│   ├── graphs/
│   │   ├── README.txt
│   │   └── *.pkl
│   └── ggd_calculation_results/
│       ├── README.txt
│       └── *.pkl
├── docs/
│   └── ...                         # HTML API documentation generated with pdoc
├── figures/
│   └── ...                         # Generated figures
├── GRASP/
│   └── ...                         # Copy of the GRASP repository; see attribution below
├── Figure-1.ipynb                  # Notebook reproducing Figure 1
├── Figure-2.ipynb                  # Notebook reproducing Figure 2
├── GGD.py                          # Main GGD implementation
├── ggd_calculations.py             # Batch calculations and result serialization
├── tests.py                        # Tests for GGD.py
├── environment.yml                 # Conda environment specification
└── README.md
```

## Contents

| Path | Description |
|---|---|
| `GGD.py` | Main implementation of functions used to compute the Graph Geodesic Distance and related quantities. |
| `ggd_calculations.py` | Runs GGD calculations on the graph datasets in `data/graphs/` and stores serialized results in `data/ggd_calculation_results/`. |
| `tests.py` | Test suite for the functions in `GGD.py`. |
| `Figure-1.ipynb` | Loads calculation results and generates the figure(s) associated with Figure 1. |
| `Figure-2.ipynb` | Loads calculation results and generates the figure(s) associated with Figure 2. |
| `data/graphs/` | Input graph datasets, stored as `.pkl` files. See the descriptor file in this directory for dataset-specific details. |
| `data/ggd_calculation_results/` | Serialized outputs of GGD calculations, stored as `.pkl` files. See the descriptor file in this directory for the result format and file descriptions. |
| `figures/` | Figures generated from the calculation results. |
| `docs/` | HTML documentation for `GGD.py`, generated with [pdoc](https://pdoc.dev/). |
| `GRASP/` | Local copy of the [GRASP](https://github.com/AU-DIS/GRASP) repository, used in selected computations. |
| `environment.yml` | Conda environment required to reproduce the computational workflow. |
