import matplotlib.pyplot as plt

import numpy as np

# Custom DPI for the different plots, 300 is very high quality but makes PDF files not able to be submitted to the course assignment system, for that case use 100
customDPI = 300

# Folder where to save the plots

plot_folder = './plots/'


def plot_confusion_matrix(y: np.ndarray, p: np.ndarray, tags: list[str], filename: str, title: str = "") -> None:
    """Plots the confusion matrix for a given prediction using the provided tags as tick labels.

    Args:
        y (np.ndarray): Expected values (class indices).
        p (np.ndarray): Predicted values (class indices).
        tags (list[str]): List of tag names corresponding to the classes.
        filename (str): File to store the plot.
    """
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Determine number of classes from tags and set tick positions and labels
    num_tags = len(tags)
    tick_positions = np.arange(num_tags)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticklabels(tags)

    # Create confusion matrix initialized to 0
    cm = np.zeros((num_tags, num_tags))
    for i in range(len(y)):
        # Adjust indices if your y and p values start at 1 instead of 0
        cm[y[i] - 1][p[i] - 1] += 1

    cax = ax.matshow(cm, cmap='Reds')
    fig.colorbar(cax)

    # Optionally add minor ticks for grid lines
    ax.set_xticks(np.arange(-.5, num_tags, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_tags, 1), minor=True)
    plt.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)

    # Write text annotations in each cell of the confusion matrix
    for (i, j), z in np.ndenumerate(cm):
        text_color = 'white' if z > cm.max() / 2 else 'black'
        ax.text(j, i, f'{z:.1f}', ha='center',
                va='center', fontsize=8, color=text_color)

    # Save the figure (ensure that plot_folder and customDPI are defined)
    plt.savefig(filename, dpi=customDPI)
