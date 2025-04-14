import matplotlib.pyplot as plt

# Fonction pour tracer le graphe de comparaison entre la fonction et son interpolation
def plot_comparison(x_range, f_exact, interp_values, title, labels=None, points_x=None, points_y=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer la fonction exacte
    ax.plot(x_range, f_exact, 'b-', linewidth=2, label='Fonction exacte')

    # Tracer les interpolations
    if isinstance(interp_values, list):
        colors = ['r', 'g', 'c', 'm', 'y']
        for i, interp in enumerate(interp_values):
            label = labels[i] if labels and i < len(labels) else f'Interpolation {i + 1}'
            ax.plot(x_range, interp, colors[i % len(colors)] + '-', linewidth=1.5, label=label)
    else:
        ax.plot(x_range, interp_values, 'r-', linewidth=1.5, label='Interpolation')

    # Tracer les points d'interpolation
    if points_x is not None and points_y is not None:
        ax.plot(points_x, points_y, 'ko', markersize=5, label='Points d\'interpolation')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()

    return fig