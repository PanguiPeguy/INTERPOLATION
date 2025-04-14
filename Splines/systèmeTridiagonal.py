import numpy as np

# Résolution de système tridiagonal
def solve_tridiagonal(a, b, c, d):
    n = len(b) - 1
    c_prime = np.zeros(n + 1)
    d_prime = np.zeros(n + 1)

    # Phase de décomposition
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / m
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / m

    d_prime[n] = (d[n] - a[n - 1] * d_prime[n - 1]) / (b[n] - a[n - 1] * c_prime[n - 1])

    # Phase de substitution
    x = np.zeros(n + 1)
    x[n] = d_prime[n]
    for i in range(n - 1, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x