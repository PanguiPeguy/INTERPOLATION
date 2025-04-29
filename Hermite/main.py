import numpy as np
import matplotlib.pyplot as plt
from Lagrange.lagrange import lagrange_interpolation_vector
from Lagrange.tchebytchev import tchebychev_points
from hermite import hermite_interpolation_vector
from deriveNumerique import numerical_derivative
from fonction import function_to_interpolate


def main():
    # Paramètres d'interpolation
    a = -25  # Borne inférieure
    b = 25  # Borne supérieure
    n = 200  # Nombre de points d'interpolation - 1

    # Points d'évaluation pour les tracés
    x_eval = np.linspace(a, b, 1000)
    f_exact = function_to_interpolate(x_eval)

    # Génération des points d'interpolation (Tchebychev pour meilleure stabilité)
    x_points = tchebychev_points(a, b, n)
    y_points = function_to_interpolate(x_points)

    # Calcul des dérivées aux points d'interpolation
    y_derivatives = np.array([numerical_derivative(function_to_interpolate, xi) for xi in x_points])

    # Calcul des interpolations
    lagrange = lagrange_interpolation_vector(x_points, y_points, x_eval)
    hermite = hermite_interpolation_vector(x_points, y_points, y_derivatives, x_eval)

    # Création de la figure avec deux sous-graphiques
    plt.figure(figsize=(12, 10))

    # Premier sous-graphique: Comparaison des interpolations
    plt.subplot(2, 1, 1)
    plt.plot(x_eval, f_exact, 'b-', linewidth=2, label='Fonction exacte')
    plt.plot(x_eval, lagrange, 'r-', linewidth=1.5, label='Lagrange')
    plt.plot(x_eval, hermite, 'g-', linewidth=1.5, label='Hermite')
    plt.plot(x_points, y_points, 'ko', markersize=5, label='Points d\'interpolation')
    plt.title('Comparaison des interpolations polynomiales')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Deuxième sous-graphique: Erreurs d'interpolation
    plt.subplot(2, 1, 2)
    plt.plot(x_eval, np.abs((f_exact - lagrange)/f_exact), 'r-', linewidth=1.5, label='Erreur Lagrange')
    plt.plot(x_eval, np.abs((f_exact - hermite)/f_exact), 'g-', linewidth=1.5, label='Erreur Hermite')
    plt.title('Erreur d\'interpolation')
    plt.xlabel('x')
    plt.ylabel('Erreur relative')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()