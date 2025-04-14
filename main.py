import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from Lagrange.lagrange import lagrange_interpolation_vector
from Lagrange.pointEquidistant import equidistant_points
from Hermite.deriveNumerique import numerical_derivative
from Hermite.hermite import hermite_interpolation, hermite_interpolation_vector
from Lagrange.tchebytchev import tchebychev_points
from Splines.cubique import cubic_spline_vector
from Splines.lineaire import linear_spline_vector
from Splines.quadratique import quadratic_spline_vector
from fonction import function_to_interpolate


class InterpolationApp:
    def __init__(self):
        # Configuration initiale
        self.current_tab = 'lagrange'
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Programme d'Interpolation Numérique")

        # Paramètres par défaut
        self.lagrange_params = {'a': -25.0, 'b': 25.0, 'n': 10}
        self.hermite_params = {'a': -10.0, 'b': 10.0, 'n': 10, 'x_eval': 0.25}
        self.spline_params = {'a': -25.0, 'b': 25.0, 'n': 10}

        # Zone pour les graphiques
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(223)
        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.axis('off')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25, hspace=0.4)

        # Créer l'interface utilisateur
        self.create_ui_components()

        # Première exécution par défaut
        self.show_lagrange()

    def create_ui_components(self):
        # Ajouter les boutons de navigation
        self.axlag = plt.axes([0.1, 0.95, 0.25, 0.04])
        self.blag = Button(self.axlag, 'Interpolation de Lagrange')
        self.blag.on_clicked(self.show_lagrange)

        self.axher = plt.axes([0.4, 0.95, 0.25, 0.04])
        self.bher = Button(self.axher, 'Interpolation d\'Hermite')
        self.bher.on_clicked(self.show_hermite)

        self.axspl = plt.axes([0.7, 0.95, 0.25, 0.04])
        self.bspl = Button(self.axspl, 'Interpolation par Splines')
        self.bspl.on_clicked(self.show_spline)

        # Créer les contrôles Lagrange
        self.axlag_a = plt.axes([0.15, 0.15, 0.3, 0.03])
        self.slagrange_a = Slider(self.axlag_a, 'Borne inf', -10.0, 0.0, valinit=self.lagrange_params['a'])

        self.axlag_b = plt.axes([0.15, 0.1, 0.3, 0.03])
        self.slagrange_b = Slider(self.axlag_b, 'Borne sup', 0.0, 10.0, valinit=self.lagrange_params['b'])

        self.axlag_n = plt.axes([0.15, 0.05, 0.3, 0.03])
        self.slagrange_n = Slider(self.axlag_n, 'Degré', 2, 20, valinit=self.lagrange_params['n'], valstep=1)

        self.axlag_run = plt.axes([0.5, 0.05, 0.15, 0.05])
        self.blagrange_run = Button(self.axlag_run, 'Exécuter')
        self.blagrange_run.on_clicked(self.update_lagrange)

        # Créer les contrôles Hermite
        self.axher_a = plt.axes([0.15, 0.15, 0.3, 0.03])
        self.shermite_a = Slider(self.axher_a, 'Borne inf', -10.0, 0.0, valinit=self.hermite_params['a'])

        self.axher_b = plt.axes([0.15, 0.1, 0.3, 0.03])
        self.shermite_b = Slider(self.axher_b, 'Borne sup', 0.0, 10.0, valinit=self.hermite_params['b'])

        self.axher_n = plt.axes([0.15, 0.05, 0.3, 0.03])
        self.shermite_n = Slider(self.axher_n, 'Nb points', 2, 20, valinit=self.hermite_params['n'], valstep=1)

        self.axher_x = plt.axes([0.55, 0.15, 0.3, 0.03])
        self.shermite_x = Slider(self.axher_x, 'Point éval', -5.0, 5.0, valinit=self.hermite_params['x_eval'])

        self.axher_run = plt.axes([0.55, 0.05, 0.15, 0.05])
        self.bhermite_run = Button(self.axher_run, 'Exécuter')
        self.bhermite_run.on_clicked(self.update_hermite)

        # Créer les contrôles Spline
        self.axspl_a = plt.axes([0.15, 0.15, 0.3, 0.03])
        self.sspline_a = Slider(self.axspl_a, 'Borne inf', -10.0, 0.0, valinit=self.spline_params['a'])

        self.axspl_b = plt.axes([0.15, 0.1, 0.3, 0.03])
        self.sspline_b = Slider(self.axspl_b, 'Borne sup', 0.0, 10.0, valinit=self.spline_params['b'])

        self.axspl_n = plt.axes([0.15, 0.05, 0.3, 0.03])
        self.sspline_n = Slider(self.axspl_n, 'Nb intervals', 2, 20, valinit=self.spline_params['n'], valstep=1)

        self.axspl_run = plt.axes([0.5, 0.05, 0.15, 0.05])
        self.bspline_run = Button(self.axspl_run, 'Exécuter')
        self.bspline_run.on_clicked(self.update_spline)

        # Rendre initialement invisibles tous les contrôles spécifiques
        self.toggle_lagrange_controls(False)
        self.toggle_hermite_controls(False)
        self.toggle_spline_controls(False)

    def toggle_lagrange_controls(self, visible):
        # Activer/désactiver la visibilité des axes plutôt que des widgets
        for ax in [self.axlag_a, self.axlag_b, self.axlag_n, self.axlag_run]:
            ax.set_visible(visible)
        # Redessiner si nécessaire
        if hasattr(self, 'fig'):
            self.fig.canvas.draw_idle()

    def toggle_hermite_controls(self, visible):
        for ax in [self.axher_a, self.axher_b, self.axher_n, self.axher_x, self.axher_run]:
            ax.set_visible(visible)
        if hasattr(self, 'fig'):
            self.fig.canvas.draw_idle()

    def toggle_spline_controls(self, visible):
        for ax in [self.axspl_a, self.axspl_b, self.axspl_n, self.axspl_run]:
            ax.set_visible(visible)
        if hasattr(self, 'fig'):
            self.fig.canvas.draw_idle()

    def show_lagrange(self, event=None):
        self.current_tab = 'lagrange'
        self.toggle_hermite_controls(False)
        self.toggle_spline_controls(False)
        self.toggle_lagrange_controls(True)
        self.clear_plots()
        self.run_lagrange()

    def show_hermite(self, event=None):
        self.current_tab = 'hermite'
        self.toggle_lagrange_controls(False)
        self.toggle_spline_controls(False)
        self.toggle_hermite_controls(True)
        self.clear_plots()
        self.run_hermite()

    def show_spline(self, event=None):
        self.current_tab = 'spline'
        self.toggle_lagrange_controls(False)
        self.toggle_hermite_controls(False)
        self.toggle_spline_controls(True)
        self.clear_plots()
        self.run_spline()

    def clear_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax_text.clear()
        self.ax_text.axis('off')
        self.fig.canvas.draw_idle()

    def update_lagrange(self, event=None):
        # Mettre à jour les paramètres
        self.lagrange_params['a'] = self.slagrange_a.val
        self.lagrange_params['b'] = self.slagrange_b.val
        self.lagrange_params['n'] = int(self.slagrange_n.val)
        self.run_lagrange()

    def update_hermite(self, event=None):
        # Mettre à jour les paramètres
        self.hermite_params['a'] = self.shermite_a.val
        self.hermite_params['b'] = self.shermite_b.val
        self.hermite_params['n'] = int(self.shermite_n.val)
        self.hermite_params['x_eval'] = self.shermite_x.val
        self.run_hermite()

    def update_spline(self, event=None):
        # Mettre à jour les paramètres
        self.spline_params['a'] = self.sspline_a.val
        self.spline_params['b'] = self.sspline_b.val
        self.spline_params['n'] = int(self.sspline_n.val)
        self.run_spline()

    def run_lagrange(self):
        # Effacer les graphiques précédents
        self.ax1.clear()
        self.ax2.clear()
        self.ax_text.clear()
        self.ax_text.axis('off')

        # Récupérer les paramètres
        a = self.lagrange_params['a']
        b = self.lagrange_params['b']
        n = self.lagrange_params['n']

        # Générer les points équidistants
        x_equi = equidistant_points(a, b, n)
        y_equi = np.array([function_to_interpolate(x) for x in x_equi])

        # Générer les points de Tchebychev
        x_cheb = tchebychev_points(a, b, n)
        y_cheb = np.array([function_to_interpolate(x) for x in x_cheb])

        # Points d'évaluation
        x_eval = np.linspace(a, b, 500)
        y_exact = np.array([function_to_interpolate(x) for x in x_eval])

        # Interpolation de Lagrange
        y_equi_interp = lagrange_interpolation_vector(x_equi, y_equi, x_eval)
        y_cheb_interp = lagrange_interpolation_vector(x_cheb, y_cheb, x_eval)

        # Calcul des erreurs
        err_equi = np.abs(y_exact - y_equi_interp)
        err_cheb = np.abs(y_exact - y_cheb_interp)

        # Graphique de comparaison des interpolations
        self.ax1.plot(x_eval, y_exact, 'b-', label='Fonction exacte')
        self.ax1.plot(x_eval, y_equi_interp, 'r-', label='Interp. points équidistants')
        self.ax1.plot(x_eval, y_cheb_interp, 'g-', label='Interp. points Tchebychev')
        self.ax1.plot(x_equi, y_equi, 'ro', markersize=4, label='Points équidistants')
        self.ax1.plot(x_cheb, y_cheb, 'go', markersize=4, label='Points Tchebychev')
        self.ax1.set_title(f"Interpolation de Lagrange (degré {n})")
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.grid(True)
        self.ax1.legend()

        # Graphique des erreurs
        self.ax2.semilogy(x_eval, err_equi, 'r-', label='Erreur points équidistants')
        self.ax2.semilogy(x_eval, err_cheb, 'g-', label='Erreur points Tchebychev')
        self.ax2.set_title(f"Erreurs d'interpolation (échelle logarithmique)")
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('Erreur absolue')
        self.ax2.grid(True)
        self.ax2.legend()

        # Afficher les erreurs maximales
        max_err_equi = np.max(err_equi)
        max_err_cheb = np.max(err_cheb)

        result_text = f"Interpolation de Lagrange\n\n"
        result_text += f"Paramètres:\n"
        result_text += f"- Borne inférieure: {a}\n"
        result_text += f"- Borne supérieure: {b}\n"
        result_text += f"- Degré du polynôme: {n}\n\n"
        result_text += f"Résultats:\n"
        result_text += f"- Erreur maximale avec points équidistants: {max_err_equi:.6e}\n"
        result_text += f"- Erreur maximale avec points de Tchebychev: {max_err_cheb:.6e}"

        self.ax_text.text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center')

        self.fig.canvas.draw_idle()

    def run_hermite(self):
        # Effacer les graphiques précédents
        self.ax1.clear()
        self.ax2.clear()
        self.ax_text.clear()
        self.ax_text.axis('off')

        # Récupérer les paramètres
        a = self.hermite_params['a']
        b = self.hermite_params['b']
        n = self.hermite_params['n'] - 1  # Convertir en indice maximal
        x_eval_point = self.hermite_params['x_eval']

        # Générer les points d'interpolation
        x_points = equidistant_points(a, b, n)
        f_values = np.array([function_to_interpolate(x) for x in x_points])
        f_derivatives = np.array([numerical_derivative(function_to_interpolate, x) for x in x_points])

        # Calcul de l'interpolation au point d'évaluation
        result = hermite_interpolation(x_points, f_values, f_derivatives, x_eval_point)
        exact = function_to_interpolate(x_eval_point)
        error = abs(exact - result)

        # Points d'évaluation pour le graphique
        x_eval = np.linspace(a, b, 500)
        y_exact = np.array([function_to_interpolate(x) for x in x_eval])

        # Calcul de l'interpolation d'Hermite sur tous les points
        y_hermite = hermite_interpolation_vector(x_points, f_values, f_derivatives, x_eval)

        # Erreur d'interpolation
        err_hermite = np.abs(y_exact - y_hermite)

        # Graphique de comparaison
        self.ax1.plot(x_eval, y_exact, 'b-', label='Fonction exacte')
        self.ax1.plot(x_eval, y_hermite, 'r-', label='Interpolation d\'Hermite')
        self.ax1.plot(x_points, f_values, 'ro', markersize=4, label='Points d\'interpolation')
        self.ax1.plot(x_eval_point, result, 'go', markersize=6, label='Point d\'évaluation')
        self.ax1.set_title(f"Interpolation d'Hermite (n={n})")
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.grid(True)
        self.ax1.legend()

        # Graphique des erreurs
        self.ax2.semilogy(x_eval, err_hermite, 'r-', label='Erreur d\'interpolation d\'Hermite')
        self.ax2.set_title("Erreur d'interpolation (échelle logarithmique)")
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('Erreur absolue')
        self.ax2.grid(True)
        self.ax2.legend()

        # Afficher les résultats
        result_text = f"Interpolation d'Hermite\n\n"
        result_text += f"Paramètres:\n"
        result_text += f"- Borne inférieure: {a}\n"
        result_text += f"- Borne supérieure: {b}\n"
        result_text += f"- Nombre de points: {n + 1}\n"
        result_text += f"- Point d'évaluation: {x_eval_point}\n\n"
        result_text += f"Résultats:\n"
        result_text += f"- Valeur d'interpolation: {result:.6f}\n"
        result_text += f"- Valeur exacte: {exact:.6f}\n"
        result_text += f"- Erreur absolue: {error:.6e}\n"
        result_text += f"- Erreur maximale: {np.max(err_hermite):.6e}"

        self.ax_text.text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center')

        self.fig.canvas.draw_idle()

    def run_spline(self):
        # Effacer les graphiques précédents
        self.ax1.clear()
        self.ax2.clear()
        self.ax_text.clear()
        self.ax_text.axis('off')

        # Récupérer les paramètres
        a = self.spline_params['a']
        b = self.spline_params['b']
        n = self.spline_params['n']

        # Générer les points d'interpolation
        x_points = equidistant_points(a, b, n)
        y_points = np.array([function_to_interpolate(x) for x in x_points])

        # Points d'évaluation pour le graphique
        x_eval = np.linspace(a, b, 500)
        y_exact = np.array([function_to_interpolate(x) for x in x_eval])

        # Calcul des différentes interpolations par splines
        y_linear = linear_spline_vector(x_points, y_points, x_eval)
        y_quadratic = quadratic_spline_vector(x_points, y_points, x_eval)
        y_cubic = cubic_spline_vector(x_points, y_points, x_eval)

        # Erreurs d'interpolation
        err_linear = np.abs(y_exact - y_linear)
        err_quadratic = np.abs(y_exact - y_quadratic)
        err_cubic = np.abs(y_exact - y_cubic)

        # Erreurs maximales
        max_err_linear = np.max(err_linear)
        max_err_quadratic = np.max(err_quadratic)
        max_err_cubic = np.max(err_cubic)

        # Graphique de comparaison des splines
        self.ax1.plot(x_eval, y_exact, 'b-', label='Fonction exacte')
        self.ax1.plot(x_eval, y_linear, 'r-', label='Spline linéaire')
        self.ax1.plot(x_eval, y_quadratic, 'g-', label='Spline quadratique')
        self.ax1.plot(x_eval, y_cubic, 'm-', label='Spline cubique')
        self.ax1.plot(x_points, y_points, 'ko', markersize=4, label='Points d\'interpolation')
        self.ax1.set_title(f"Interpolation par splines ({n} intervalles)")
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.grid(True)
        self.ax1.legend()

        # Graphique des erreurs
        self.ax2.semilogy(x_eval, err_linear, 'r-', label='Erreur spline linéaire')
        self.ax2.semilogy(x_eval, err_quadratic, 'g-', label='Erreur spline quadratique')
        self.ax2.semilogy(x_eval, err_cubic, 'm-', label='Erreur spline cubique')
        self.ax2.set_title("Erreurs d'interpolation (échelle logarithmique)")
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('Erreur absolue')
        self.ax2.grid(True)
        self.ax2.legend()

        # Afficher les résultats
        result_text = f"Interpolation par Splines\n\n"
        result_text += f"Paramètres:\n"
        result_text += f"- Borne inférieure: {a}\n"
        result_text += f"- Borne supérieure: {b}\n"
        result_text += f"- Nombre d'intervalles: {n}\n\n"
        result_text += f"Résultats:\n"
        result_text += f"- Erreur maximale avec spline linéaire: {max_err_linear:.6e}\n"
        result_text += f"- Erreur maximale avec spline quadratique: {max_err_quadratic:.6e}\n"
        result_text += f"- Erreur maximale avec spline cubique: {max_err_cubic:.6e}"

        self.ax_text.text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center')

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = InterpolationApp()
    plt.show()
    plt.savefig('interpolation_result.png')