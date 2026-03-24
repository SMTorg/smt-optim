.. smt-optim documentation master file, created by
   sphinx-quickstart on Tue Mar 17 17:51:37 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

smt-optim
=========

Introduction
============
`smt-optim` is an open-source python package for Bayesian Optimization. It is well suited for solving
expensive-to-evaluate blackbox problem with little exploitable properties such as derivatives. It can handle
constrained and multi-fidelity global optimization problems.

**Focus on multi-fidelity**

`smt-optim` is designed for multi-fidelity optimization with hierarchical levels of fidelity to reduce the optimization
cost. The MFSEGO acquisition strategy judiciously select low fidelity and high fidelity levels when sampling the
blackbox functions.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction <index>
   getting-started
   examples
   API Reference <api/smt_optim>


