# Bayesian Optimization and Multi-fidelity Approaches

A multi-fidelity constrained Bayesian optimization toolkit

## Key Features

The BOMA package offers a collection of surrogate-based optimization frameworks. The following frameworks are available:


| Framework | Inequality Constraints | Equality Constraints | Multi-fidelity | As seen in                                                                               |
|-----------|------------------------|----------------------|----------------|------------------------------------------------------------------------------------------|
| SEGO      | Yes                    | Yes                  | No             | [https://doi.org/10.1080/03052150211751](https://doi.org/10.1080/03052150211751)         |
| MFSEGO    | Yes                    | Yes                  | Yes            | [https://doi.org/10.2514/6.2019-3236](https://doi.org/10.2514/6.2019-3236)               |
| VF-EI     | Yes                    | No                   | Yes            | [https://doi.org/10.1007/s00158-018-1971-x](https://doi.org/10.1007/s00158-018-1971-x)   |
| VF-PI     | Yes                    | No                   | Yes            | [https://doi.org/10.1007/s00158-020-02646-9](https://doi.org/10.1007/s00158-020-02646-9) |



## Getting Started

### Prerequisites

BOMA requires the following Python package to be installed:
1. Numpy
``pip install numpy``
2. SciPy
``pip install scipy``
3. SMT
``pip install smt``


### Installation

1. Clone the repo
```
git clone https://github.com/oihanc/boma.git
```
2. Install BOMA to your Python environment. In the root directory, type: 
```
pip install -e .
```

### Usage
See usage examples in the `boma/examples/` directory.


## Please cite us when using BOMA

If you are using BOMA in your work, please cite the following paper.

[Oihan Cordelier, Youssef Diouane, Nathalie Bartoli and Eric Laurendeau. "Multi-Fidelity Constrained Bayesian Optimization with Application to Aircraft Wing Design," AIAA 2025-3474. AIAA AVIATION FORUM AND ASCEND 2025. July 2025.](https://doi.org/10.2514/6.2025-3474)