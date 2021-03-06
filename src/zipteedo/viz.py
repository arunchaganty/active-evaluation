"""
Common, useful visualization routines.
"""
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def draw_matrix(X, x_labels=None, y_labels=None, with_values=True, vmin=None, vmax=None):
    """
    Plots a matrix using imshow
    """
    plt.imshow(abs(X), cmap="viridis", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    # Add the text
    if with_values:
        y_size, x_size = X.shape
        x_positions = np.linspace(start=0, stop=x_size, num=x_size, endpoint=False)
        y_positions = np.linspace(start=0, stop=y_size, num=y_size, endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = "{:.2f}".format(X[y_index, x_index])
                plt.text(x, y, label, color='white', ha='center', va='center', fontsize=11)

    if x_labels:
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    if y_labels:
        plt.yticks(np.arange(len(y_labels)), y_labels)


def violinplot(ax, xy, distinct_values, color=None):
    x, y = xy.T[0], xy.T[1]

    vs, vs_, widths = [], [], []
    for v in distinct_values:
        if (x == v).any():
            y_ = y[x == v]
            vs.append(y_)
            vs_.append(v)
            widths.append(len(y_)/len(y))
    parts = ax.violinplot(vs, vs_, widths=widths, showmeans=True)

    if color:
        for name, cols in parts.items():
            if name == "bodies":
                for pc in cols:
                    pc.set_facecolor(color)
            else:
                cols.set_edgecolor(color)
