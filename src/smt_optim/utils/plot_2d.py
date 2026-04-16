import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

def get_plot2d_data(func: Callable, bounds: np.ndarray, num_points: int = 101) -> tuple:

    X = np.linspace(bounds[0, 0], bounds[0, 1], num_points)
    Y = np.linspace(bounds[1, 0], bounds[1, 1], num_points)
    XX, YY = np.meshgrid(X, Y)

    data = np.vstack((XX.ravel(), YY.ravel())).T
    z = np.empty(data.shape[0])

    for i in range(data.shape[0]):
        z[i] = func(data[i, :])

    Z = z.reshape(XX.shape)

    return XX, YY, Z