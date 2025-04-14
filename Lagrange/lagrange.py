import numpy as np

# Interpolation de Lagrange
def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points) - 1

    # Vérifier si x correspond à un des points d'interpolation
    for i in range(n + 1):
        if x == x_points[i]:
            return y_points[i]

    # Calcul des dénominateurs d_i pour chaque terme
    d = np.ones(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                d[i] *= (x_points[i] - x_points[j])

    # Calcul du numérateur commun num = Π(x - x_i)
    num = np.prod(x - x_points)

    # Calcul du polynôme de Lagrange
    p = 0.0
    for i in range(n + 1):
        q = num / (x - x_points[i])
        p += y_points[i] * q / d[i]

    return p

# Version vectorisée de l'interpolation de Lagrange pour plusieurs points
def lagrange_interpolation_vector(x_points, y_points, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = lagrange_interpolation(x_points, y_points, x)
    return results