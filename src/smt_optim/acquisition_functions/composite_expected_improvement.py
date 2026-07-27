import numpy as np

def PositivePart(x):
    return max(x,0)

def init_bi_obj_composite_ei(state,kwargs):

    phi=kwargs["phi"]
    n_expectancy=kwargs.get("n_accuracy",1000)

    def composite_expected_improvement(mu: float, s2: float, f_min: float, n_expectancy=n_expectancy) -> float:
        """
        Expected Improvement composite acquisition function.

        Parameters
        ----------
        mu: np.array
            Mean prediction.
        s2: np.array
            Variance prediction.
        f_min: float
            Best minimum objective value in training data.
        phi: np.array -> float

        Returns
        -------
        float
            Expected Improvement value.
        """

        S=np.atleast_1d(0.0)
        for i in range(n_expectancy):
            sampleZ = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,1]]))
            S+=PositivePart(f_min-phi(mu+s2*sampleZ))
        ei=S/n_expectancy

        return ei[0]
    
    models=state.obj_models
    f_min=min([phi(y) for y in state.scaled_dataset.export_data([0,1],0)])

    def composite_ei(x_pred):
        s = np.array([
            np.sqrt(models[0].predict_variances(x_pred)).item(),
            np.sqrt(models[1].predict_variances(x_pred)).item()
        ])

        y = np.array([
            models[0].predict_values(x_pred).item(),
            models[1].predict_values(x_pred).item(),
        ])
        return composite_expected_improvement(y,s,f_min)
    return composite_ei