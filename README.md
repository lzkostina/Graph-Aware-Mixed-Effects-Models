# Graph-Aware-Mixed-Effects-Models

This repository contains my attempt to implement an efficient and reproducible version of the EM algorithm for 
the **Graph-Aware Mixed-Effects model** from https://projecteuclid.org/journals/annals-of-applied-statistics/volume-17/issue-3/Graph-aware-modeling-of-brain-connectivity-networks/10.1214/22-AOAS1709.short
---

## Project Goals

The main goal of this project is to implement and optimize the EM algorithm for Graph-Aware Mixed-Effects model, sine 
original article does not provide Python implementation of described methods. 


## Repository Structure:

```text
project-root/
├── src/
│   ├── model/
│   │   ├── graph_aware_em.py       # Reference / naive EM implementation
│   │   ├── block_descent.py        # Optimized EM (block descent version)
│   │   ├── multicov_model.py       # Extension to include multiple covariates
│   │   └── variance_structure.py   # Diagonal or block diagonal structure of variance
│   ├── design/                  # Build Z, covariates, and cell assignments
│   │   ├── cells.py
│   │   ├── design_matrices.py
│   │   └── fixed_effects.py
│   ├── io/                      # Data loading 
│   │   ├── io_cobre.py
│   │   └── power_groups.py
│   └── utils/                   # Small helpers (logging, timing, etc.)
├── demo/
│   ├── demo.ipynb               # Main demo (runnable in < 30 minutes)
│   └── demo.py                  # Script version of the demo 
├── data/
│   ├── raw/                     # COBRE dataset used in article 
│   └── processed/              
├── pipeline/                    # simple scripts to run experiments reflected in Section 3.1 of the original article
│   ├── cobre_analysis.py
│   ├── sec_3_1_cell_only.py
│   └── section31.py
├── scripts
│   └── run_cobre.py
├── tests/
│   ├── test_io.py               # Basic correctness tests for design and other helpers
│   └── test_design_variance.py  # Sanity checks for variance constructions
├── Makefile
├── requirements.txt
├── README.md
└── report-KOSTINA.pdf           # Final write-up 
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/lzkostina/Graph-Aware-Mixed-Effects-Models
cd Graph-Aware-Mixed-Effects-Models
```
2. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

All main steps are automated via the Makefile.

### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
make test
```