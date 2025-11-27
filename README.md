# Graph-Aware-Mixed-Effects-Models

This repository contains my attempt to implement an efficient and reproducible version of the EM algorithm for 
the **Graph-Aware Mixed-Effects model** from https://projecteuclid.org/journals/annals-of-applied-statistics/volume-17/issue-3/Graph-aware-modeling-of-brain-connectivity-networks/10.1214/22-AOAS1709.short
---

## Project Goals

The main goal of this project is to implement and optimize the EM algorithm for Graph-Aware Mixed-Effects model, sine 
original article does not provide Python implementation of described methods. 


## Repository Structure

Planned structure (!!!!!!! PLANNED, NEED TO BE MODIFIED !!!!!!!!!!!!!!!):

```text
project-root/
├── src/
│   ├── model/
│   │   ├── em_baseline.py       # Reference / naïve EM implementation
│   │   ├── em_optimized.py      # Optimized EM (sparse, vectorized)
│   │   └── gls_utils.py         # GLS / variance-related helpers
│   ├── design/                  # Build Z, covariates, and cell assignments
│   ├── io/                      # Data loading (e.g., COBRE or synthetic)
│   └── utils/                   # Small helpers (logging, timing, etc.)
├── demo/
│   ├── demo.ipynb               # Main demo (runnable in < 30 minutes)
│   └── demo.py                  # Script version of the demo (optional)
├── data/
│   ├── raw/                     # Empty or example-only (no large data committed)
│   └── processed/
├── results/
│   ├── figures/                 # Plots (e.g., log-likelihood vs iteration)
│   ├── tables/                  # Small summary tables / benchmarks
│   └── benchmarks/              # Timing / performance results
├── tests/
│   ├── test_em.py               # Basic correctness tests for EM
│   └── test_gls.py              # Sanity checks for GLS helpers
├── requirements.txt
├── README.md
└── report-KOSTINA.pdf           # Final write-up (1–3 pages)
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