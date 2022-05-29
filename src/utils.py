import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

import settings


def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, cmap='YlGn', xticklabels=settings.CLASSES, yticklabels=settings.CLASSES)
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.show()


def visualize_ds(dataset):
    x_batch, y_batch = next(dataset)
    fig = plt.figure(1, (4., 4.))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.3, )
    for i, axes in enumerate(grid):
        axes.set_title(settings.CLASSES[np.argmax(y_batch[i])], fontdict=None, loc='center', color='k')
        axes.imshow(x_batch[i] / 255)

    plt.show()
