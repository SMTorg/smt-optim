---
title: 'SMT-Optim: A Python toolbox for constrained and multi-fidelity Bayesian optimization'
tags:
  - Python
  - Bayesian optimization
  - Blackbox optimization
  - Multi-fidelity
  - Mixed-variable
authors:
  - name: Oihan Cordelier
    orcid: 0009-0008-7916-7474
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Youssef Diouane
    orcid: 0000-0002-6609-7330
    affiliation: 1
  - name: Nathalie Bartoli
    orcid: 0000-0002-6451-2203
    affiliation: 2
  - name: Rémi Lafage
    orcid: 0000-0001-5479-2961
    affiliation: 2
  - name: Eric Laurendeau
    orcid: 0000-0001-5770-2597
    affiliation: 3
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, Canada
   index: 1
 - name: Fédération ENAC ISAE-SUPAERO ONERA, Université de Toulouse, France
   index: 2
 - name: Department of Mechanical Engineering, Polytechnique Montréal, Canada
   index: 3
  

date: 19 May 2026
bibliography: paper.bib

---

# Summary

SMT-Optim is a research-oriented Python package that brings together Bayesian optimization algorithms for solving, but not limited to, constrained, multi-fidelity, mixed-variable, expensive-to-evaluate blackbox problems of the type: 

$$
\begin{aligned}
\min_{\boldsymbol x \in \Omega \times S \times \mathbb{F}^l } \quad & f(\boldsymbol x) \\
\text{s.t.} \quad
& \boldsymbol g(\boldsymbol x) \le \boldsymbol 0 \\
& \boldsymbol h(\boldsymbol x) = \boldsymbol 0
\end{aligned}
$$

where $\Omega \subset \mathbb{R}^c$ represents the bounded continuous variables, $S \subset \mathbb{Z}^d$ the bounded integer variables, and $\mathbb{F}^l$ the discrete qualitative variables. $f: \mathbb{R}^c \times \mathbb{Z}^d \times \mathbb{F}^l \to \mathbb{R}$ is the objective function, $\boldsymbol g: \mathbb{R}^c \times \mathbb{Z}^d \times \mathbb{F}^l \to \mathbb{R}^m$ are the $m$ inequality constraints, and $\boldsymbol h: \mathbb{R}^c \times \mathbb{Z}^d \times \mathbb{F}^l \to \mathbb{R}^n$ are the $n$ equality constraints. In many applications, $f$, $\boldsymbol g$, and $\boldsymbol h$ offer little exploitable structure, such as derivative information. SMT-Optim is designed with a particular focus on constrained, multi-fidelity, and mixed-variable optimization. As such, it is particularly well suited to engineering optimization problems.
The SMT-Optim package also provides various benchmark optimization frameworks, as listed in Table \ref{tab:frameworks}.

: Optimization frameworks available in SMT-Optim \label{tab:frameworks}

| Framework | Surrogate | Equality constraints | Inequality constraints | Multi-fidelity | Mixed-variable |
| --------- | :-------: | :------------------: | :--------------------: | :------------: | :------------: |
| EGO       |  GPX/KRG  |        No            |         No             |       No       |       Yes      |
| SEGO      |  GPX/KRG  |       Yes            |        Yes             |       No       |       Yes      |
| MFSEGO    |    MFK    |       Yes            |        Yes             |       Yes      |       Yes      |
| VF-PI     |    MFCK   |        No            |        Yes             |       Yes      |        No      |

# Statement of need

One of SMT-Optim's goals is to provide an interface for solving constrained, multi-fidelity, and mixed-variable problems with Bayesian optimization. SMT-Optim has two target audiences:

1. Users who want to apply Bayesian optimization to expensive-to-evaluate blackbox problems; 
2. Researchers who want to experiment with Bayesian optimization and develop new frameworks.

## Application to multidisciplinary optimization

In engineering, Multidisciplinary Design Analysis (MDA) couples physical disciplines such as aerodynamics, structures, and propulsion, resulting in highly nonlinear, multimodal, and expensive functions. When embedded within Multidisciplinary Design Optimization (MDO), these analyses must be evaluated repeatedly, making the overall optimization process prohibitively costly, especially when high-fidelity simulations such as computational fluid dynamics (CFD) or finite element method (FEM) are involved. Global optimization methods such as Efficient Global Optimization (EGO) [@jones_1998] improve sample efficiency by constructing surrogate models and balancing exploration and exploitation of the design space. The EGO framework, along with its constrained (SEGO) [@sasena_2002] and multi-fidelity (MFSEGO) [@meliani_2019; @cordelier_etal_2026] counterparts, is implemented in SMT-Optim and can be launched with a single method call through the functional API. 

## Illustration of multi-fidelity Bayesian optimization

Figure \ref{fig:rosenbrock} illustrates the sampling behavior of SMT-Optim's MFSEGO implementation on the constrained multi-fidelity Rosenbrock test problem [@lam_2015] with low- (LF) and high-fidelity (HF) sampling costs of 0.1 and 1, respectively. The initial Design of Experiments (DOE) is comprised of 8 low- and 4 high-fidelity samples, totaling a initial sampling budget of 4.8 units. The solution is found after 15 iterations. The final DOE is comprised of 23 LF samples and 6 HF samples, resulting in a final sampling budget of 8.3 units. 

![Constrained Rosenbrock low-fidelity (LF) and high-fidelity (HF) functions and the sampled points at the end of the optimization process. The shaded areas denote the unfeasible regions due to the inequality constraint. \label{fig:rosenbrock}](figures/rosenbrock_multifidelity.pdf)

Figure \ref{fig:rosenbrock_best_obj} compares the required convergence budget between SMT-Optims's MFSEGO (multi-fidelity) and SEGO (mono-fidelity) implementations on the same test problem. SEGO is started with the same HF samples as MFSEGO and found a comparable solution with a sampling budget of 14 units. In this example, utilizing low-fidelity approximations for the objective and constraint reduced the budget required to find the solution by 41%.

![Comparison between the required budget for convergence for multi- and mono-fidelity Bayesian optimization (BO). \label{fig:rosenbrock_best_obj}](figures/rosenbrock_convergence.pdf){width="70%"}

# State of the field

Engineering design problems are commonly solved with genetic algorithms, such as those offered by the Pymoo Python library [@pymoo]. These optimization methods are sample-intensive and are therefore not well suited to expensive-to-evaluate blackbox functions. Several open-source Bayesian optimization libraries exist, including BoTorch [@balandat_2020] and EGObox [@lafage_2022]. BoTorch does support multi-fidelity optimization, but not for constrained and mixed-variable problems. Furthermore, BoTorch handles constraints by penalizing the acquisition function with the probability of feasibility [@schonlau_1997], whereas the SEGO implementation in SMT-Optim maximizes the acquisition function with respect to the mean prediction of the constraints. EGObox offers efficient implementation of the EGO and SEGO frameworks for continuous and mixed-variable design spaces. Written in Rust, it delivers high performance, although customization can be more challenging. It does not support multi-fidelity optimization.

SMT-Optim positions itself as an extension of the SMT package [@saves_2024]. It leverages SMT's feature-rich surrogate models, sampling methods, and design spaces to provide benchmarked Bayesian optimization frameworks to users. For example, SMT-Optim uses the multi-fidelity MFK [@le_gratiet_2013] and MFCK [@castano-aguirre_2026] models in its MFSEGO and VF-PI implementations, respectively.

# Software design

SMT-Optim is designed with an emphasis on modularity, reproducibility, and experimental rigor, enabling both practical applications and methodological development in Bayesian optimization.

The package follows a modular architecture in which the main components (surrogate models, acquisition strategies, and the optimization driver) are decoupled. This allows users to substitute custom components to experiment with new modeling tools and acquisition mechanisms while maintaining a standardized Bayesian optimization workflow, as well as consistent design of experiments and statistical logging procedures.

The package also offers a functional API, a component-based API, and an ask--tell API to accommodate different user requirements. The functional API enables users to launch a Bayesian optimization process with a single method call, making it well suited for quick experimentation, while the component-based API allows users to customize surrogate models, acquisition strategies, acquisition functions, and Bayesian optimization hyperparameters, making it better suited for rigorous optimization pipelines.

Furthermore, SMT-Optim provides base classes for its main components, enabling users to implement custom surrogate models or acquisition strategies. Users can choose among different constraint-handling techniques, exploration penalties, and fidelity-selection methods to construct acquisition strategies that fit their needs. The package also includes many test functions with diverse properties to support standardized benchmarking.

SMT-Optim trades execution speed for design flexibility. In some cases, however, it can leverage faster-executing models, such as EGObox's GPX, to reduce Bayesian optimization overhead. In general, Bayesian optimization is applied to time-consuming functions, making the selection of a high-quality infill point a higher priority than minimizing computational overhead.

# Research impact statement

SMT-Optim has been used in [@cordelier_etal_2025; @cordelier_etal_2026] demonstrating the MFSEGO framework on multidisciplinary wing design test cases. Furthermore, SMT-Optim aims at providing open-source implementation of the EGO, SEGO, and MFSEGO frameworks discussed in [@jones_1998; @sasena_2002; @meliani_2019]. The VF-PI framework follows the implementation described in [@ruan_2020]. The SMT-Optim documentation also provides reproducible examples for unconstrained, constrained, mono-fidelity, and multi-fidelity optimization over continuous and mixed-variable design spaces.

# AI usage disclosure

Generative AI was used to assist with code testing and debugging. All AI-generated code was reviewed before integration.

# Acknowledgements

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [funding reference number 110_2025_2026_Q3_251].

# References