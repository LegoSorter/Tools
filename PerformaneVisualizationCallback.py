import numpy as np
import matplotlib.pyplot as plt
import pandas

from pathlib import Path
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from tensorflow.keras.callbacks import Callback
from CustomImageDataGenerator import DataGenerator


class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, data: DataGenerator, output_dir: Path):
        super().__init__()
        self.model = model
        self.data = data
        self.image_dir = output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs={}):
        y_predictions = []
        y_true = []
        for chunk in self.data:
            y_predictions.extend(self.model.predict(chunk[0]))
            y_true.extend([self.data.one_hot_to_label(one_hot) for one_hot in chunk[1]])

        # y_predictions = self.model.predict(self.validation_data)
        # y_true = self.validation_data.get_all_classes(one_hot=False)
        y_true = np.array(y_true)
        y_predictions = np.array(y_predictions)
        y_pred = np.array([self.data.one_hot_to_label(y_pred) for y_pred in y_predictions])

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pandas.DataFrame(report).transpose()
        report_df.to_csv(self.image_dir / f'classification_report_epoch_{epoch}.csv')

        results_df = pandas.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        results_df.to_csv(str(self.image_dir / f"classification_results_epoch_{epoch}.csv"))

        # plot and save confusion matrix
        fig_size = len(self.data.labels) // 2
        fig, ax = plt.subplots(figsize=(fig_size + 4, fig_size + 3))

        plot_confusion_matrix(y_true, y_pred, ax=ax, x_tick_rotation=1)
        fig.savefig(str(self.image_dir / f'confusion_matrix_epoch_{epoch}'))
        plt.close(fig)

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(fig_size + 4, fig_size + 3))
        plot_roc(y_true, y_predictions, ax=ax)
        fig.savefig(str(self.image_dir / f'roc_curve_epoch_{epoch}'))
        plt.close(fig)
