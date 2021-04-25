import argparse as argparse
import numpy
import tensorflow as tf
import imgaug.augmenters as iaa
import imgaug as ia
import pandas
import wandb

from wandb.keras import WandbCallback
from pathlib import Path
from tensorflow.keras import metrics
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from CustomImageDataGenerator import DataGenerator

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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


def build_model(num_classes, config):
    img_augmentation = Sequential(
        [
            preprocessing.RandomRotation(factor=0.5),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(),
            preprocessing.RandomContrast(factor=0.3),
        ],
        name="img_augmentation",
    )

    inputs = layers.Input(shape=(config.img_size, config.img_size, 3))
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=config.loss_function,
                  metrics=["accuracy", metrics.top_k_categorical_accuracy])

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

    ####### PREPARE ######

    train_df = pandas.read_csv(args.train)
    test_df = pandas.read_csv(args.validation)
    classes_train = train_df.label.unique()
    classes_train.sort()
    classes_test = test_df.label.unique()
    classes_test.sort()

    if classes_test.shape != classes_train.shape or not (numpy.array(classes_test) == numpy.array(classes_train)).all():
        print("Labels in train and test dataset are different")
        intersection = classes_test[numpy.in1d(classes_test, classes_train)]
        print(f"There is {len(intersection)} common labels, the rest is ignored.")
        train_df = train_df[train_df.label.isin(intersection)]
        test_df = test_df[test_df.label.isin(intersection)]

    NUM_CLASSES = len(train_df.label.unique())

    train_base_generator = DataGenerator(train_df, 'image_path', 'label', image_size=int(args.size),
                                         aug_sequence=get_augmenting_sequence(),
                                         batch_size=int(args.batch))
    test_base_generator = DataGenerator(test_df, 'image_path', 'label', reduction=0.98, aug_sequence=None,
                                        image_size=int(args.size),
                                        batch_size=int(args.batch))

    wandb.login()

    run = wandb.init(project='lego',
                     config={
                         "learning_rate": 1e-2,
                         "epochs": 25,
                         "batch_size": 32,
                         "loss_function": "categorical_crossentropy",
                         "architecture": "CNN",
                         "img_size": int(args.size),
                     })
    config = wandb.config
    model = build_model(NUM_CLASSES, config)

    ####### TRAIN #######

    history_1 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=100, epochs=10,
                          callbacks=[WandbCallback()])
    save_history_to_file(history_1, Path(args.history) / 'history_1.csv')

    unfreeze_model(model, 30, 1e-3)
    history_2 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=500, epochs=20,
                          callbacks=[WandbCallback()])
    save_history_to_file(history_2, Path(args.history) / 'history_2.csv')

    unfreeze_model(model, 80, 1e-4)
    history_3 = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=500, epochs=80,
                          callbacks=[WandbCallback()])
    save_history_to_file(history_3, Path(args.history) / 'history_3.csv')
