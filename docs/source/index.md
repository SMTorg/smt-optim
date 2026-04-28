# smt-optim: SMT Optimization

## Introduction

SMT Optim is an open-source Python package for Bayesian optimization developed for research applications. It is well suited to expensive-to-evaluate black-box problems that offer limited exploitable structure, such as derivative information. The package supports constrained and multi-fidelity global optimization for mixed-variable design spaces.

## Cite us

To cite SMT Optim:

```bibtex
@techreport{cordelier_etal_2026,
    author      = {Cordelier, Oihan and Diouane, Youssef and Bartoli, Nathalie and Laurendeau, Eric},
    title       = {Multi-fidelity approaches for general constrained Bayesian optimization with application to aircraft design},
    institution = {{GERAD}},
    year        = 2026,
    type        = {Cahier du GERAD},
    number      = {G-2026-17},
    address     = {Montr\'eal, QC, Canada},
    doi         = {10.48550/arXiv.2603.28987}
}
```
```bibtex
@inproceedings{cordelier_etal_2025,
    author      = {Cordelier, Oihan and Diouane, Youssef and Bartoli, Nathalie and Laurendeau, Eric},
    title       = {{Multi-Fidelity Constrained Bayesian Optimization with Application to Aircraft Wing Design}},
    booktitle   = {{AIAA AVIATION FORUM AND ASCEND 2025}},
    year        = {2025},
    address     = {Las Vegas, Nevada},
    month       = jul,
    publisher   = {American Institute of Aeronautics and Astronautics},
    doi         = {10.2514/6.2025-3474}
    }
```

## Focus on constrained Bayesian optimization

SMT Optim supports both equality and inequality blackbox constraints. For each constraint, it builds a surrogate model and uses it during acquisition function optimization. The acquisition function can be optimized either with respect to the surrogate mean prediction or by penalizing it with the probability of feasibility. The SMT Optim interface also allows users to define both lower and upper bounds for each black-box constraint.

## Focus on multi-fidelity

SMT Optim is designed for multi-fidelity optimization with hierarchical fidelity levels to reduce computational cost. The MFSEGO acquisition strategy judiciously selects low- and high-fidelity evaluations when sampling the blackbox functions. Currently, SMT Optim offers two state-of-the-art multi-fidelity frameworks: MFSEGO for nested design spaces and VF-PI for non-nested design spaces. Both frameworks can be further customized with specific acquisition functions and framework-specific parameters.

## Focus on mixed-variable

SMT Optim supports continuous, integer, and categorical variables. It relies on SMT's Design Space to define mixed-variable design spaces and on SMT's surrogate models to accurately represent the quantities of interest with respect to their input variables.

## Focus on a modular framework

SMT Optim is designed to be modular, allowing users to swap components such as surrogate models, acquisition strategies, and acquisition functions while maintaining a consistent overall structure that is well suited to research benchmarking. The package also offers a straightforward interface through the `minimize` method, enabling seamless implementation and automatically selecting an appropriate optimization framework based on the characteristics of the problem.


```{toctree}
:maxdepth: 1
:caption: Getting started:
getting_started/installation
getting_started/unconstrained_optim
getting_started/constrained_optim
getting_started/multifidelity_optim
getting_started/mixed_var_optim
```

```{toctree}
:maxdepth: 1
:caption: Reference: 
API <api/index>
```

```{toctree}
:maxdepth: 1
:caption: Advanced features:
advanced_features/examples/index.md
```

## Contributing institutions

&nbsp;

```{image} _static/banner.png
    :scale: 20%
    :alt: logo banner
    :align: center
```

