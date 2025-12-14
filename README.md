# Graph-Aware-Mixed-Effects-Models

Efficient, reproducible Python implementation of the EM algorithm for the Graph-Aware Mixed-Effects (GA-ME) model 
for brain connectivity networks, based on Kim, Kessler & Levina (2023, Annals of Applied Statistics).
This project focuses on scalable estimation (Woodbury identity + block coordinate descent) and a reproducible demo 
pipeline suitable for running on a laptop.
*Reference paper:*
https://projecteuclid.org/journals/annals-of-applied-statistics/volume-17/issue-3/Graph-aware-modeling-of-brain-connectivity-networks/10.1214/22-AOAS1709.short
---

## Project Goals

* The main goal of this project is to implement and optimize the EM algorithm for Graph-Aware Mixed-Effects model, since 
original article does not provide Python implementation of described methods. 

* Make the workflow reproducible and easy to run (environment, tests, one-command demo).

* Provide a self-contained demonstration that runs in < 30 minutes.

## Repository Structure:

```text
project-root/
├── src/
│   ├── model/
│   │   ├── graph_aware_em.py       # Reference / naive EM implementation
│   │   ├── block_descent.py        # Optimized EM (block descent version)
│   │   ├── multicov_model.py       # Extension to include multiple covariates
│   │   └── variance_structure.py   # Diagonal or block diagonal structure of variance
│   ├── design/                     # Build Z, covariates, and cell assignments
│   │   ├── cells.py
│   │   ├── design_matrices.py
│   │   └── fixed_effects.py
    │   ├── io/                      # Data loading 
│   │   ├── io_cobre.py
│   │   └── power_groups.py
│   └── utils/                      # Helpers (logging, timing, etc.)
├── demo/
│   ├── demo.ipynb                  # Main demo (runnable in < 30 minutes)
│   └── demo.py                     # Script version of the demo 
├── data/
│   ├── raw/                        # COBRE dataset used in article 
│   └── processed/              
├── pipeline/                       # Scripts used for replication-style experiments
│   ├── cobre_analysis.py
│   ├── sec_3_1_cell_only.py
│   └── section31.py
├── scripts
│   └── run_cobre.py
├── tests/
│   ├── test_io.py                  # Basic correctness tests for design and other helpers
│   └── test_design_variance.py     # Sanity checks for variance constructions
├── Makefile
├── requirements.txt
├── README.md
└── report-KOSTINA.pdf              # Final write-up 
```

## Quickstart
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
Run demo (recommended)
```bash
make demo
```
This should:
* run a small end-to-end example
* fit the model
* produce a minimal set of outputs

### Data 

This project supports running on the COBRE-derived connectivity representation used in the paper.
Because raw neuroimaging-derived datasets are often large and restricted, this repository is designed so that:
the demo runs without requiring the full raw dataset (using a small bundled example or cached artifacts), and
full-data runs (if you have access) can be performed by placing data locally under data/raw/ and running the pipeline 
scripts. COBRE data downloaded using R is stored in `data\raw` for convenience. 


### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
make test
```