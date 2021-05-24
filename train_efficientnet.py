import argparse as argparse
import numpy
import tensorflow as tf
import imgaug.augmenters as iaa
import imgaug as ia
import pandas
import wandb

from pathlib import Path
from tensorflow.keras import metrics
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from wandb.integration.keras import WandbCallback

from evaluation_callbacks import PerformanceVisualizationCallback
from generators import DataGenerator

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_augmenting_sequence():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    return iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.9, 1.0), "y": (0.9, 1.0)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-15, 15),
            shear=(-10, 10),
            order=[0, 1],
            cval=(0, 255),
            mode=["constant"]
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
                           iaa.MotionBlur()
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


def unfreeze_model(model, layers_to_unfreeze=20, learning_rate=1e-4, whole_model=False):
    # We unfreeze the top x layers while leaving BatchNorm layers frozen
    _layers = model.layers[-layers_to_unfreeze:] if not whole_model else model.layers
    for layer in _layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", metrics.top_k_categorical_accuracy])


def save_history_to_file(history, filename: Path):
    hist_df = pandas.DataFrame(history.history)
    with open(filename, mode='w') as f:
        hist_df.to_csv(f)


def train(model, epochs=20, name='1', history_path=Path("./history")):
    checkpoint_filepath = history_path / (f"./weights/{name}/" + "checkpoint_{epoch:02d}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max')

    history = model.fit(train_base_generator, validation_data=test_base_generator, steps_per_epoch=1000, epochs=epochs,
                        callbacks=[WandbCallback(), model_checkpoint_callback,
                                   PerformanceVisualizationCallback(data=test_base_generator, evaluate_every_x_epoch=5,
                                                                    output_dir=history_path / f"stage_name")])
    save_history_to_file(history, history_path / f"history_{name}.csv")

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

    history_path = Path(args.history)
    history_path.mkdir(exist_ok=True, parents=True)

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

    train_base_generator = DataGenerator(train_df, 'image_path', 'label',
                                         reduction=0.6,
                                         image_size=int(args.size),
                                         aug_sequence=get_augmenting_sequence(),
                                         batch_size=int(args.batch))

    from PIL import Image
    import random
    Path("train_renders").mkdir(exist_ok=True)
    counter = 0
    for data in train_base_generator:
        for image, label in zip(data[0], data[1]):
            Image.fromarray(image).save("train_renders/" + f"{train_base_generator.one_hot_to_label(label)}_" + str(random.randint(0, 100)) + ".jpg")
        counter += 1
        if counter > 10:
            break

    test_base_generator = DataGenerator(test_df, 'image_path', 'label',
                                        reduction=0.95,
                                        aug_sequence=None,
                                        image_size=int(args.size),
                                        batch_size=int(args.batch))

    counter = 0
    Path("test_photos").mkdir(exist_ok=True)
    for data in test_base_generator:
        for image, label in zip(data[0], data[1]):
            Image.fromarray(image).save("test_photos/" + f"{test_base_generator.one_hot_to_label(label)}_" + str(random.randint(0, 100)) + ".jpg")
        counter += 1
        if counter > 10:
            break

    wandb.login()

    run = wandb.init(project='lego',
                     name=args.history,
                     config={
                         "learning_rate": 1e-2,
                         "epochs": 25,
                         "batch_size": args.batch,
                         "loss_function": "categorical_crossentropy",
                         "architecture": "CNN",
                         "img_size": int(args.size),
                     })
    config = wandb.config
    model = build_model(NUM_CLASSES, config)

    ###### TRAIN #######

    train(model, 10, "10_epochs_0.01_lr_1_layer", args.history)
    unfreeze_model(model, 30, 5e-3)
    train(model, 20, "20_epochs_0.005_lr_30_layers", args.history)
    unfreeze_model(model, 100, 1e-3)
    train(model, 40, "40_epochs_0.001lr_100_layers", Path(args.history))
    unfreeze_model(model, 200, 5e-4)
    train(model, 40, "40_epochs_0.0005lr_200_layers", Path(args.history))
    unfreeze_model(model, layers_to_unfreeze=0, learning_rate=5e-5, whole_model=True)
    train(model, 80, "80_epochs_0.00005lr_all_layers", Path(args.history))
    unfreeze_model(model, layers_to_unfreeze=0, learning_rate=1e-5, whole_model=True)
    train(model, 40, "40_epochs_0.00001lr_all_layers", Path(args.history))
    unfreeze_model(model, layers_to_unfreeze=0, learning_rate=1e-6, whole_model=True)
    train(model, 40, "40_epochs_0.000001lr_all_layers", Path(args.history))
