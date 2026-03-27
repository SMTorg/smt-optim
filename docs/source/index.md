# smt-optim

## Introduction
`smt-optim` is an open-source python package for Bayesian Optimization. It is well suited for solving
expensive-to-evaluate blackbox problem with little exploitable properties such as derivatives. It can handle
constrained and multi-fidelity global optimization problems.

### Focus on multi-fidelity

`smt-optim` is designed for multi-fidelity optimization with hierarchical levels of fidelity to reduce the optimization
cost. The MFSEGO acquisition strategy judiciously select low fidelity and high fidelity levels when sampling the
blackbox functions.


```{toctree}
:maxdepth: 2
:caption: Contents:

Introduction <index>
get-started.md
concepts.md
examples.md
architecture.md
API reference <api/smt_optim>
```



