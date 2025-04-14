import numpy as np

# Génération des points de Tchebychev
def tchebychev_points(a, b, n):
    i = np.arange(n+1)
    t = np.cos(np.pi * (2.0 * i + 1.0) / (2.0 * (n + 1)))
    return 0.5 * (a + b) + 0.5 * (b - a) * t