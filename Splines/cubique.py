import numpy as np
from Splines.systèmeTridiagonal import solve_tridiagonal

# Coefficients pour les splines cubiques
def cubic_spline_coefficients(x_points, y_points):
    n = len(x_points) - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x_points[i + 1] - x_points[i]

    # Construire le système tridiagonal pour les coefficients c
    alpha = np.zeros(n)
    diag_a = np.zeros(n)
    diag_b = np.zeros(n + 1)
    diag_c = np.zeros(n)
    rhs = np.zeros(n + 1)

    # Condition de bord: c[0] = 0 (spline cubique naturelle)
    diag_b[0] = 1.0
    rhs[0] = 0.0

    for i in range(1, n):
        # Calculer les coefficients alpha (second membre)
        alpha[i] = 3.0 * ((y_points[i + 1] - y_points[i]) / h[i] -
                          (y_points[i] - y_points[i - 1]) / h[i - 1])

        # Remplir les diagonales
        diag_a[i - 1] = h[i - 1]
        diag_b[i] = 2.0 * (h[i - 1] + h[i])
        diag_c[i] = h[i]
        rhs[i] = alpha[i]

    # Condition de bord: c[n] = 0 (spline cubique naturelle)
    diag_a[n - 1] = h[n - 1]
    diag_b[n] = 1.0
    rhs[n] = 0.0

    # Résoudre le système pour les coefficients c
    c = solve_tridiagonal(diag_a, diag_b, diag_c, rhs)

    # Calculer les coefficients a, b et d
    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        a[i] = y_points[i]
        b[i] = (y_points[i + 1] - y_points[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a, b, c, d


# Interpolation par spline cubique
def cubic_spline(x_points, y_points, x):
    n = len(x_points) - 1

    # Calcul des coefficients
    a, b, c, d = cubic_spline_coefficients(x_points, y_points)

    # Recherche de l'intervalle contenant x
    i = np.searchsorted(x_points, x) - 1

    # Gestion des cas limites
    if i < 0:
        return y_points[0]
    if i >= n:
        return y_points[n]

    # Calculer la valeur du spline : S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    dx = x - x_points[i]
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx


# Version vectorisée de l'interpolation par spline cubique
def cubic_spline_vector(x_points, y_points, x_eval):
    results = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        results[i] = cubic_spline(x_points, y_points, x)
    return results