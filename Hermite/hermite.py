import numpy as np

# Interpolation d'Hermite
def hermite_interpolation(x_points, f_values, f_derivatives, x):
    n = len(x_points) - 1

    # Vérifier si x correspond à un des points d'interpolation
    for i in range(n + 1):
        if x == x_points[i]:
            return f_values[i]

    p = 0.0
    for i in range(n + 1):
        l_i = 1.0
        c_i = 0.0

        for j in range(n + 1):
            if i != j:
                l_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
                c_i += 1.0 / (x_points[i] - x_points[j])

        p += ((1.0 - 2.0 * (x - x_points[i]) * c_i) * f_values[i] +
              (x - x_points[i]) * f_derivatives[i]) * l_i * l_i

    return p

# Version vectorisée de l'interpolation d'Hermite pour plusieurs points
def hermite_interpolation_vector(x_points, f_values, f_derivatives, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = hermite_interpolation(x_points, f_values, f_derivatives, x)
    return results