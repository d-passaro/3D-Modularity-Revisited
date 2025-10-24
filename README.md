# 3D Modularity Revisited - Companion Code

This repository contains the computational companion to the paper **"3D Modularity Revisited"** by M. Cheng, I. Coman, P. Kucharski, D. Passaro, and G. Sgroi.

## Overview

This repository provides Jupyter notebooks and SageMath code that reproduce the computational examples from the paper. The notebooks demonstrate:

- Computation of Ẑ-invariants for 3-manifolds using plumbing graphs
- Weil representation calculations for mock theta functions
- False theta series and their modular properties
- Verification of 3D modularity conjectures through explicit examples

## Paper Information

- **arXiv**: [2403.14920](https://arxiv.org/abs/2403.14920)
- **PDF**: [arXiv PDF](https://arxiv.org/pdf/2403.14920)
- **Authors**: Miranda C. N. Cheng, Ioana Coman, Piotr Kucharski, Davide Passaro, Gabriele Sgroi

## Prerequisites

### 1. SageMath

[SageMath](https://www.sagemath.org/) is required to run the notebooks. We recommend installing via [Anaconda](https://www.anaconda.com/):

```bash
conda create -n sage sage python=3.12
conda activate sage
```

### 2. pyPlumbing

[pyPlumbing](https://github.com/d-passaro/pyPlumbing) is required for Ẑ-invariant computations:

```bash
pip install pyplumbing
```

**Note**: pyPlumbing requires Python 3.12 or newer.

### 3. Function Library

Load the main function library in your notebooks:

```python
load("notebooks/3d_modularity_revisited.sage")
```

## Key Functions

The `3d_modularity_revisited.sage` library provides:

### Weil Representation Functions
- `omega(m, n)` - Weil representation matrices
- `p_plus(m, n)`, `p_minus(m, n)` - Projection operators
- `weil_projector(m, K, irrep=True)` - Complete Weil projectors
- `weil_reps(m, K, irrep=True)` - Extract representation data

### Theta Series Functions
- `false_theta(m, r, max_n, q)` - False theta series
- `indefinite_theta(A, a, b, c1, c2, n_max)` - Indefinite theta functions
- `zhat_indefinite_theta(...)` - Ẑ-hat indefinite theta functions
- `ramanujan_theta(x, chi, n_max)` - Ramanujan theta functions

### Mock Theta Functions
- `F0(prec)`, `F1(prec)`, `F2(prec)` - Ramanujan's order 7 mock theta functions

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@misc{cheng20243dmodularityrevisited,
  title={3d Modularity Revisited},
  author={Miranda C. N. Cheng and Ioana Coman and Piotr Kucharski and Davide Passaro and Gabriele Sgroi},
  year={2024},
  eprint={2403.14920},
  archivePrefix={arXiv},
  primaryClass={hep-th},
  url={https://arxiv.org/abs/2403.14920},
}
```

## Contact

For questions about the paper, code, or this repository, contact any of the authors:

- [Miranda Cheng](mailto:c.n.cheng@uva.nl)
- [Ioana Coman](mailto:ioana.coman@ipmu.jp)
- [Piotr Kucharski](mailto:piotr.kucharski@mimuw.edu.pl)
- [Davide Passaro](mailto:dpassaro@caltech.edu)
- [Gabriele Sgroi](mailto:gabrielesgroi94@gmail.com)
