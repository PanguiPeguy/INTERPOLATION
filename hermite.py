import matplotlib
matplotlib.use("TkAgg")  # Utilise le backend TkAgg pour éviter les conflits avec GTK4

import numpy as np                      # Bibliothèque pour les calculs numériques
import matplotlib.pyplot as plt         # Pour les tracés graphiques
from scipy.interpolate import KroghInterpolator  # Pour l'interpolation de Hermite
import sympy as sp                      # Pour le calcul symbolique (ex : dérivée automatique)

# Fonction qui transforme une chaîne mathématique (ex: "sin(x)") en fonction Python exécutable
def creer_fonction(expr_str):
    def f(x):
        return eval(expr_str, {
            "x": x,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": np.pi
        })
    return f

# Introduction
print("=== Interpolation de Hermite ===")
print("Fonctions autorisées : sin(x), cos(x), exp(x), log(x), sqrt(x), etc.\n")

# Saisie de la fonction f(x)
expr_fx = input("Entrez f(x) (exemple : sin(x)) : ")
f = creer_fonction(expr_fx)

# Saisie ou calcul automatique de f'(x)
expr_dfx = input("Entrez f'(x) (laisser vide pour calcul automatique) : ")
if not expr_dfx:
    x_sym = sp.Symbol('x')                        # Déclare x comme variable symbolique
    expr_sym = sp.sympify(expr_fx)                # Transforme f(x) en expression symbolique
    dfx_sym = sp.diff(expr_sym, x_sym)            # Dérive symboliquement
    expr_dfx = str(dfx_sym)                       # Convertit en chaîne pour recréer la fonction
    print(f"Dérivée calculée automatiquement : f'(x) = {expr_dfx}")
df = creer_fonction(expr_dfx)

# Saisie des points d'interpolation
n = int(input("Nombre de points : "))
x_points = [float(input(f"x[{i}] = ")) for i in range(n)]  # Saisie des abscisses
x_points = np.array(x_points)              # Conversion en tableau numpy
y_points = f(x_points)                     # Calcul des ordonnées
dy_points = df(x_points)                   # Calcul des dérivées

# Préparation des données pour Hermite (chaque point est dupliqué)
x_hermite = []
y_hermite = []
for i in range(n):
    x_hermite += [x_points[i], x_points[i]]        # Le même x pour f(x) et f'(x)
    y_hermite += [y_points[i], dy_points[i]]       # f(x) suivi de f'(x)

# Interpolateur de Hermite
interpolateur = KroghInterpolator(x_hermite, y_hermite)

# Génération de points pour le tracé
x_plot = np.linspace(min(x_points), max(x_points), 500)  # Abscisses pour le tracé
f_plot = f(x_plot)                         # Valeurs de f(x)
hermite_plot = interpolateur(x_plot)      # Valeurs de H(x)
erreur_plot = np.abs(f_plot - hermite_plot)  # Erreur absolue entre f(x) et H(x)

# Création du graphique
plt.figure(figsize=(10, 6))  # Taille de la figure

# Courbe de f(x)
plt.plot(x_plot, f_plot, label="f(x)", color="blue")

# Courbe de l'interpolation de Hermite
plt.plot(x_plot, hermite_plot, label="Interpolation Hermite", color="orange", linestyle="--")

# Courbe de l'erreur absolue
plt.plot(x_plot, erreur_plot, label="Erreur |f(x) - H(x)|", color="green", linestyle=":")

# Points d'interpolation
plt.scatter(x_points, y_points, color="red", label="Points d'interpolation")

# Mise en forme du graphique
plt.title("f(x), Interpolation de Hermite et Erreur")
plt.xlabel("x")
plt.ylabel("Valeurs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()  # Affichage du graphique