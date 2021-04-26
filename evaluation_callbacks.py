import wandb
import numpy as np
import matplotlib.pyplot as plt
import pandas

from pathlib import Path
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from tensorflow.keras.callbacks import Callback
from generators import DataGenerator


class PRMetrics(Callback):
    """ Custom callback to compute metrics at the end of each training epoch"""

    def __init__(self, generator: DataGenerator = None, num_log_batches=1):
        super().__init__()
        self.generator = generator
        self.num_batches = num_log_batches
        # store full names of classes
        self.flat_class_names = generator.extract_labels()

    def on_epoch_end(self, epoch, logs={}):
        # collect validation data and ground truth labels from generator
        val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
        val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

        # use the trained model to generate predictions for the given number
        # of validation data batches (num_batches)
        val_predictions = self.model.predict(val_data)
        ground_truth_class_ids = val_labels.argmax(axis=1)
        # take the argmax for each set of prediction scores
        # to return the class id of the highest confidence prediction
        top_pred_ids = val_predictions.argmax(axis=1)

        # Log confusion matrix
        # the key "conf_mat" is the id of the plot--do not change
        # this if you want subsequent runs to show up on the same plot
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           preds=top_pred_ids,
                                                           y_true=ground_truth_class_ids,
                                                           class_names=self.flat_class_names)})


class PerformanceVisualizationCallback(Callback):
    def __init__(self, data: DataGenerator, output_dir: Path, evaluate_every_x_epoch=1):
        super().__init__()
        self.data = data
        self.image_dir = output_dir
        self.evaluate_every_x_epoch = evaluate_every_x_epoch
        output_dir.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.evaluate_every_x_epoch != 0:
            return

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
        fig_size = len(self.data.labels) // 20
        fig, ax = plt.subplots(figsize=(fig_size + 4, fig_size + 3))

        plot_confusion_matrix(y_true, y_pred, ax=ax, x_tick_rotation=90)
        fig.savefig(str(self.image_dir / f'confusion_matrix_epoch_{epoch}'))
        plt.close(fig)

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(fig_size + 4, fig_size + 3))
        plot_roc(y_true, y_predictions, ax=ax)
        fig.savefig(str(self.image_dir / f'roc_curve_epoch_{epoch}'))
        plt.close(fig)
