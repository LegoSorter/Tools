import argparse as argparse
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import imgaug as ia
import pandas
import cv2 as cv

from pathlib import Path
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import metrics
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras with an imgaug sequence"""

    def __init__(self,
                 dataframe,
                 x_col,
                 y_col,
                 aug_sequence,
                 reduction=0.0,
                 batch_size=32,
                 size=224,
                 shuffle=True,
                 balance_dataset=True):
        if reduction != 0.0 and balance_dataset is False:
            raise Exception("Cannot set reduction without balancing dataset.")

        self.df = dataframe if not balance_dataset else self.balance_dataset(dataframe, y_col, x_col, reduction)
        self.x_col = x_col
        self.y_col = y_col
        self.df_index = self.df.index.tolist()
        self.labels = self.extract_labels()
        self.indexes = np.arange(len(self.df_index))
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug_sequence = aug_sequence
        self.future_data_provider = None
        self.current_index = 0
        self.prefetch = False

        self.on_epoch_end()

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.prefetch:
            data_batch = self.future_data_provider.result()
            self.current_index = (self.current_index + 1) % self.__len__()
            self.__prefetch_data(self.current_index)

            return data_batch

        return self.__get_data(index)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.df_index) // self.batch_size

    def reduce(self, dataframe, reduction):
        if reduction > 0.0:
            reduction_count = int(len(dataframe) * reduction)
            drop_indices = np.random.choice(dataframe.index, reduction_count, replace=False)
            return dataframe.drop(drop_indices).reset_index()
        return dataframe

    def balance_dataset(self, dataframe, label_column, path_column, reduction):
        count_per_label = \
        dataframe.value_counts(subset=[label_column]).reset_index(name='count').set_index(label_column)[
            'count'].to_dict()
        upper_limit = int((1.0 - reduction) * max(count_per_label.values()))
        grouped_by_label = dataframe.groupby(label_column)[path_column].apply(list).reset_index(name='paths')

        for index, row in grouped_by_label.iterrows():
            label_name = row[label_column]
            current_count = count_per_label[label_name]
            missing = upper_limit - current_count

            if missing < 0:
                indices = dataframe.index[dataframe[label_column] == label_name]
                drop_indices = np.random.choice(indices, abs(missing), replace=False)
                dataframe = dataframe.drop(drop_indices).reset_index(drop=True)
            elif missing > 0:
                repeated = random.choices(row['paths'], k=missing)
                extension = pandas.DataFrame([[label_name, path] for path in repeated],
                                             columns=[label_column, path_column])
                dataframe = pandas.concat([dataframe, extension])

        return dataframe

    def extract_labels(self):
        labels = self.df[self.y_col].unique().tolist()
        labels.sort()
        return labels

    def get_all_classes(self, one_hot=False):
        y_true_list = []
        for label in self.df['label']:
            y_true = self.__to_one_hot(label) if one_hot else label
            y_true_list.append(y_true)

        return np.array(y_true_list)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.df_index))
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if self.prefetch:
            self.__prefetch_data(0)

    def __prefetch_data(self, index):
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.future_data_provider = executor.submit(self.__get_data, index)

    def __to_one_hot(self, label):
        encoding = np.zeros((len(self.labels)))
        encoding[self.labels.index(label)] = 1.0
        return encoding

    def one_hot_to_label(self, one_hot):
        index = np.argmax(one_hot)
        return self.labels[index]

    def __get_data(self, index):
        # X.shape : (batch_size, *dim)
        index = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch = [self.df_index[k] for k in index]

        labels = []
        for k in batch:
            label = self.df.iloc[k][self.y_col]
            label_encoding = self.__to_one_hot(label)
            labels.append(label_encoding)

        labels = np.array(labels)
        images = self.resize_with_pad([cv.imread(self.df.iloc[k][self.x_col]) for k in batch], image_size=self.size)

        if self.aug_sequence is not None:
            images = self.aug_sequence.augment_images(images)

        return np.stack(images), labels

    @staticmethod
    def resize_with_pad(images, image_size=224):
        return iaa.Sequential([
            iaa.Resize({"longer-side": image_size, "shorter-side": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(width=image_size, height=image_size)])(images=images)


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


def get_augmenting_sequence():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    return iaa.Sequential([
        iaa.Fliplr(0.3),
        iaa.Flipud(0.3),
        iaa.Affine(
            scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-20, 20),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        ),
        iaa.SomeOf((0, 5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).
                       sometimes(
                           iaa.Superpixels(
                               p_replace=(0, 1.0),
                               n_segments=(20, 200)
                           )
                       ),

                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(2, 7)),
                           iaa.MedianBlur(k=(3, 11)),
                       ]),

                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect).
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                       # Same as sharpen, but for an embossing effect.
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=0.2
                           ),
                       ]),

                       # Invert each image's channel with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.05, per_channel=True),  # invert color channels

                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-10, 10), per_channel=0.5),

                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.5, 1.2), per_channel=0.5),

                       # Improve or worsen the contrast of images.
                       iaa.LinearContrast((0.5, 1.5), per_channel=0.5),

                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       iaa.Grayscale(alpha=(0.0, 1.0)),

                       # In some images move pixels locally around (with random
                       # strengths).
                       sometimes(
                           iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                       ),

                       # In some images distort local areas with varying strength.
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],
                   # do all of the above augmentations in random order
                   random_order=True)
    ])


def build_model(num_classes, img_size):
    img_augmentation = Sequential(
        [
            preprocessing.RandomRotation(factor=0.5),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(),
            preprocessing.RandomContrast(factor=0.3),
        ],
        name="img_augmentation",
    )

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                  metrics=["accuracy", metrics.top_k_categorical_accuracy, metrics.categorical_accuracy])

    return model


def unfreeze_model(model, layers_to_unfreeze=20, learning_rate=1e-4):
    # We unfreeze the top x layers while leaving BatchNorm layers frozen
    for layer in model.layers[-layers_to_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


def save_history_to_file(history, filename: Path):
    hist_df = pandas.DataFrame(history.history)
    with open(filename, mode='w') as f:
        hist_df.to_csv(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains a network using data listed in provided csv files in format [\'label\', \'image_path\']')
    parser.add_argument('-t' '--train_data', required=True, help='A CSV file containing training data',
                        type=str, dest='train')
    parser.add_argument('-v', '--validation_data', required=True, help='A CSV file containing validation/test data',
                        type=str, dest='validation')
    parser.add_argument('-s', '--size', required=True, help='A size of images.')
    parser.add_argument('-b', '--batch_size', help='Batch size used in training', dest='batch')
    parser.add_argument('-o', '--history_output', help="Output directory for training history", dest='history')

    args = parser.parse_args()

    train_df = pandas.read_csv(args.train)
    test_df = pandas.read_csv(args.validation)
    train_base_generator = DataGenerator(train_df, 'image_path', 'label', size=int(args.size),
                                         aug_sequence=get_augmenting_sequence(),
                                         batch_size=int(args.batch))
    test_base_generator = DataGenerator(test_df, 'image_path', 'label', reduction=0.98, aug_sequence=None,
                                        size=int(args.size),
                                        batch_size=int(args.batch))

    classes_train = train_df['label'].unique()
    classes_train.sort()
    classes_test = test_df['label'].unique()
    classes_test.sort()

    if not (classes_test == classes_test).all():
        raise Exception("Labels in train and test dataset are different!")

    NUM_CLASSES = len(classes_train)
    model = build_model(NUM_CLASSES, int(args.size))
    performance_callback_1 = PerformanceVisualizationCallback(
        model=model,
        data=test_base_generator,
        output_dir=Path(args.history) / 'performance_visualizations_1')
    history_1 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=100, epochs=10,
                          callbacks=[performance_callback_1])
    save_history_to_file(history_1, Path(args.history) / 'history_1.csv')

    performance_callback_2 = PerformanceVisualizationCallback(
        model=model,
        data=test_base_generator,
        output_dir=Path(args.history) / 'performance_visualizations_2')
    unfreeze_model(model, 30, 1e-3)
    history_2 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=500, epochs=20,
                          callbacks=[performance_callback_2])
    save_history_to_file(history_2, Path(args.history) / 'history_2.csv')

    performance_callback_3 = PerformanceVisualizationCallback(
        model=model,
        data=test_base_generator,
        output_dir=Path(args.history) / 'performance_visualizations_3')
    unfreeze_model(model, 100, 1e-4)
    history_3 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=500, epochs=80,
                          callbacks=[performance_callback_3])
    save_history_to_file(history_3, Path(args.history) / 'history_3.csv')
