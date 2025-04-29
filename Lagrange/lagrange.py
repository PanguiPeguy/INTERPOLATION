import numpy as np

def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points) - 1

    for i in range(n + 1):
        if x == x_points[i]:
            return y_points[i]

    d = np.ones(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                d[i] *= (x_points[i] - x_points[j])

    num = np.prod(x - x_points)
    p = 0.0
    for i in range(n + 1):
        q = num / (x - x_points[i])
        p += y_points[i] * q / d[i]

    return p

def lagrange_interpolation_vector(x_points, y_points, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = lagrange_interpolation(x_points, y_points, x)
    return results