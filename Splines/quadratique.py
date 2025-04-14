import numpy as np
from Splines.systèmeTridiagonal import solve_tridiagonal

# Coefficients pour les splines quadratiques
def quadratic_spline_coefficients(x_points, y_points):
    n = len(x_points) - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x_points[i + 1] - x_points[i]

    # Construire le système tridiagonal pour les coefficients b
    diag_a = np.zeros(n)
    diag_b = np.zeros(n + 1)
    diag_c = np.zeros(n)
    d = np.zeros(n + 1)

    # Condition de bord: b[0] = 0 (spline quadratique naturelle)
    diag_b[0] = 1.0
    d[0] = 0.0

    for i in range(1, n):
        diag_a[i - 1] = h[i - 1]
        diag_b[i] = 2 * (h[i - 1] + h[i])
        diag_c[i] = h[i]
        d[i] = 3 * ((y_points[i + 1] - y_points[i]) / h[i] -
                    (y_points[i] - y_points[i - 1]) / h[i - 1])

    diag_a[n - 1] = h[n - 1]
    diag_b[n] = 1.0
    d[n] = 0.0  # Condition de bord: b[n] = 0

    # Résoudre le système pour les coefficients b
    b = solve_tridiagonal(diag_a, diag_b, diag_c, d)

    # Calculer les coefficients a et c
    a = np.zeros(n)
    c = np.zeros(n)
    for i in range(n):
        a[i] = y_points[i]
        c[i] = ((y_points[i + 1] - y_points[i]) - b[i] * h[i]) / (h[i] * h[i])

    return a, b, c


# Interpolation par spline quadratique
def quadratic_spline(x_points, y_points, x):
    n = len(x_points) - 1

    # Calcul des coefficients
    a, b, c = quadratic_spline_coefficients(x_points, y_points)

    # Recherche de l'intervalle contenant x
    i = np.searchsorted(x_points, x) - 1

    # Gestion des cas limites
    if i < 0:
        return y_points[0]
    if i >= n:
        return y_points[n]

    # Calculer la valeur du spline : S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2
    dx = x - x_points[i]
    return a[i] + b[i] * dx + c[i] * dx * dx


# Version vectorisée de l'interpolation par spline quadratique
def quadratic_spline_vector(x_points, y_points, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = quadratic_spline(x_points, y_points, x)
    return results