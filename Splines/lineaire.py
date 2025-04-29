import numpy as np

def linear_spline(x_points, y_points, x):
    n = len(x_points) - 1

    i = np.searchsorted(x_points, x) - 1

    # Gestion des cas limites
    if i < 0:
        return y_points[0]
    if i >= n:
        return y_points[n]

    t = (x - x_points[i]) / (x_points[i + 1] - x_points[i])
    return y_points[i] + t * (y_points[i + 1] - y_points[i])


def linear_spline_vector(x_points, y_points, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = linear_spline(x_points, y_points, x)
    return results