# Concepts


## Bayesian Optimization
Bayesian optimization (BO) is a statistical optimization approach well-suited for black-box functions that are expensive to evaluate. "Black-box" implies problems with few exploitable properties, such as gradients. Rather than working directly with the function to be optimized, a surrogate model is constructed using a few samples of the function. Using the surrogate model, an acquisition function is employed to acquire new sample locations. The black-box function is sampled at these points. With this new data, the surrogate model is updated. These steps are repeated until a convergence criterion is met, usually a maximum number of iterations.


## Efficient Global Optimization (EGO)

