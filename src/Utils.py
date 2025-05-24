import matplotlib.pyplot as plt
import numpy as np

#: DPI setting for plot output (300 for high quality, 100 for course submissions)
customDPI = 300

#: Directory path for saving output plots
plot_folder = './plots/'


def plot_confusion_matrix(y: np.ndarray, p: np.ndarray, tags: list[str], 
                         filename: str, title: str = "") -> None:
    """Generate and save a confusion matrix visualization.

    Creates a heatmap-style confusion matrix with class labels, grid lines,
    and value annotations in each cell. Saves the plot to specified file.

    :param y: Ground truth class indices (0-based or 1-based)
    :type y: np.ndarray
    :param p: Predicted class indices (same indexing as y)
    :type p: np.ndarray
    :param tags: Class names corresponding to indices
    :type tags: list[str]
    :param filename: Output file path for saving the plot
    :type filename: str
    :param title: Optional title for the plot, defaults to ""
    :type title: str, optional

    Note:
        - Handles both 0-based and 1-based class indexing automatically
        - Uses red color gradient (Reds colormap) for the heatmap
        - Includes grid lines between cells for better readability
        - Text annotations are white/black for optimal contrast
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

    # Add minor ticks for grid lines
    ax.set_xticks(np.arange(-.5, num_tags, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_tags, 1), minor=True)
    plt.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)

    # Add text annotations in each cell
    for (i, j), z in np.ndenumerate(cm):
        text_color = 'white' if z > cm.max() / 2 else 'black'
        ax.text(j, i, f'{z:.1f}', ha='center',
                va='center', fontsize=8, color=text_color)

    # Save figure with specified DPI
    plt.savefig(filename, dpi=customDPI, bbox_inches='tight')
    plt.close()