import numpy as np
import matplotlib.pyplot as plt
from pointEquidistant import equidistant_points
from tchebytchev import tchebychev_points
from lagrange import lagrange_interpolation_vector
from fonction import function_to_interpolate


def main():
    # Paramètres d'interpolation
    a = -25  # Borne inférieure
    b = 25  # Borne supérieure
    n = 10  # Nombre de points d'interpolation - 1

    # Points d'évaluation pour les tracés
    x_eval = np.linspace(a, b, 1000)
    f_exact = function_to_interpolate(x_eval)

    # Génération des points d'interpolation
    x_equi = equidistant_points(a, b, n)
    y_equi = function_to_interpolate(x_equi)

    x_tcheb = tchebychev_points(a, b, n)
    y_tcheb = function_to_interpolate(x_tcheb)

    # Calcul des interpolations
    interp_equi = lagrange_interpolation_vector(x_equi, y_equi, x_eval)
    interp_tcheb = lagrange_interpolation_vector(x_tcheb, y_tcheb, x_eval)

    # Création de la figure avec deux sous-graphiques
    plt.figure(figsize=(12, 8))

    # Premier sous-graphique: Comparaison des interpolations
    plt.subplot(2, 1, 1)
    plt.plot(x_eval, f_exact, 'b-', linewidth=2, label='Fonction exacte')
    plt.plot(x_eval, interp_equi, 'r-', linewidth=1.5, label='Points équidistants')
    plt.plot(x_eval, interp_tcheb, 'g-', linewidth=1.5, label='Points de Tchebychev')
    plt.plot(x_equi, y_equi, 'ko', markersize=5, label='Points d\'interpolation')
    plt.title('Comparaison des interpolations de Lagrange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Deuxième sous-graphique: Erreurs d'interpolation
    plt.subplot(2, 1, 2)
    plt.plot(x_eval, np.abs((f_exact - interp_equi)/f_exact), 'r-', linewidth=1.5, label='Erreur (équidistants)')
    plt.plot(x_eval, np.abs((f_exact - interp_tcheb)/f_exact), 'g-', linewidth=1.5, label='Erreur (Tchebychev)')
    plt.title('Erreur d\'interpolation')
    plt.xlabel('x')
    plt.ylabel('Erreur relative')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # Pour éviter les chevauchements
    plt.show()


if __name__ == "__main__":
    main()