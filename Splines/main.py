import numpy as np
import matplotlib.pyplot as plt

from Lagrange.pointEquidistant import equidistant_points
from fonction import function_to_interpolate
from lineaire import linear_spline_vector
from quadratique import quadratic_spline_vector
from cubique import cubic_spline_vector


def main():
    # Paramètres d'interpolation
    a = -25  # Borne inférieure
    b = 25  # Borne supérieure
    n = 200  # Nombre de points d'interpolation - 1

    # Points d'évaluation pour les tracés
    x_eval = np.linspace(a, b, 1000)
    f_exact = function_to_interpolate(x_eval)

    # Génération des points d'interpolation (équidistants)
    x_points = equidistant_points(a, b, n)
    y_points = function_to_interpolate(x_points)

    # Calcul des différentes interpolations par splines
    linear = linear_spline_vector(x_points, y_points, x_eval)
    quadratic = quadratic_spline_vector(x_points, y_points, x_eval)
    cubic = cubic_spline_vector(x_points, y_points, x_eval)

    # Création de la figure avec deux sous-graphiques
    plt.figure(figsize=(12, 10))

    # Premier sous-graphique: Comparaison des splines
    plt.subplot(2, 1, 1)
    plt.plot(x_eval, f_exact, 'k-', linewidth=2, label='Fonction exacte')
    plt.plot(x_eval, linear, 'b-', linewidth=1.5, label='Spline linéaire')
    plt.plot(x_eval, quadratic, 'r-', linewidth=1.5, label='Spline quadratique')
    plt.plot(x_eval, cubic, 'g-', linewidth=1.5, label='Spline cubique')
    plt.plot(x_points, y_points, 'ko', markersize=5, label='Points d\'interpolation')
    plt.title('Comparaison des différentes splines d\'interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Deuxième sous-graphique: Erreurs d'interpolation
    plt.subplot(2, 1, 2)
    plt.plot(x_eval, np.abs(f_exact - linear), 'b-', linewidth=1.5, label='Erreur linéaire')
    plt.plot(x_eval, np.abs(f_exact - quadratic), 'r-', linewidth=1.5, label='Erreur quadratique')
    plt.plot(x_eval, np.abs(f_exact - cubic), 'g-', linewidth=1.5, label='Erreur cubique')
    plt.title('Erreur d\'interpolation')
    plt.xlabel('x')
    plt.ylabel('Erreur absolue')
    plt.yscale('log')  # Échelle logarithmique pour mieux voir les différences
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()