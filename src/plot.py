import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .constants import ASSETS_DIR


class CountPlotter:
    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.data_colors = {}
        self.data_count = {}
        self.class_size = 900
        self.figure_size = (12, 6)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def set_data_class(self, name: str = None, color: str = None, count: int = 0):
        self.data_colors[name] = color
        self.data_count[name] = count

    def plot(self):
        plt.figure(figsize=self.figure_size)
        class_names, class_colors = list(zip(*self.data_colors.items()))
        ax = sns.countplot(x=self._get_data(), palette=class_colors, order=class_names)

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        bbox = {
            "boxstyle": "round,pad=0.5",
            "edgecolor": "skyblue",
            "facecolor": "white",
        }

        plt.annotate(
            self._get_plot_legend(),
            (20, 20),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=4,
            bbox=bbox,
        )

    def show_plot(self):
        plt.show()

    def save_plot(self, path):
        file_name = ASSETS_DIR / path
        plt.savefig(file_name, self.class_size)

        return str(file_name)

    def _get_data(self):
        return np.concatenate(
            [[name] * count for name, count in self.data_count.items()]
        )

    def _get_plot_legend(self):
        legend_text = "\n".join(
            f"{name}: {rate:.2f}" for name, rate in self._get_data_class_rates().items()
        )

    def _get_data_class_rates(self):
        total_count = self._get_total_count()
        return {
            name: count / total_count * 100 for name, count in self.data_count.items()
        }

    def _get_total_count(self):
        return sum(self.data_count.values())



class ConfusionMatrixPlotter:
    def __init__(self):
        self.class_names = ["Normal", "Neumonía"]
        self.class_size = 900

    def plot(self, confusion_matrix_data):
        plt.figure(figsize=(12,4))
        sns.set(font_scale=1.2)

        sns.heatmap(
            confusion_matrix_data,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=False,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )

        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Etiqueta Verdadera")

    def show_plot(self):
        plt.show()

    def save_plot(self, path):
        file_name = ASSETS_DIR / path
        plt.savefig(str(file_name), dpi=self.class_size)
