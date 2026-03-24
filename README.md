# SMT Optimization

A multi-fidelity constrained Bayesian optimization toolkit

## Key Features

The SMT Optimization package offers a collection of surrogate-based optimization frameworks. The following frameworks are available:


| Framework | Inequality Constraints | Equality Constraints | Multi-fidelity | As seen in                                                                               |
|-----------|------------------------|----------------------|----------------|------------------------------------------------------------------------------------------|
| SEGO      | Yes                    | Yes                  | No             | [https://doi.org/10.1080/03052150211751](https://doi.org/10.1080/03052150211751)         |
| MFSEGO    | Yes                    | Yes                  | Yes            | [https://doi.org/10.2514/6.2019-3236](https://doi.org/10.2514/6.2019-3236)               |



## Getting Started

### Prerequisites

`smt-optim` requires the following Python package to be installed:
1. Numpy
``pip install numpy``
2. SciPy
``pip install scipy``
3. SMT
``pip install smt``


### Installation

1. Clone the repo
```
git clone https://github.com/SMTOrg/smt-optim.git
```
2. Install `smt-optim` to your Python environment. In the root directory, type: 
```
pip install -e .
```

### Usage
See usage examples in the `examples/` directory.


## Please cite us when using SMT Optimization

If you are using SMT Optimization in your work, please cite the following paper.

[Oihan Cordelier, Youssef Diouane, Nathalie Bartoli and Eric Laurendeau. "Multi-Fidelity Constrained Bayesian Optimization with Application to Aircraft Wing Design," AIAA 2025-3474. AIAA AVIATION FORUM AND ASCEND 2025. July 2025.](https://doi.org/10.2514/6.2025-3474)