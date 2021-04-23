import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import random

import numpy
import imgaug.augmenters as iaa
import imgaug as ia

from pathlib import Path
from PIL import Image


def get_processing_sequence(image_size):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    return iaa.Sequential([
        # iaa.Resize({"longer-side": image_size, "shorter-side": "keep-aspect-ratio"}),
        # iaa.PadToFixedSize(width=image_size, height=image_size),
        iaa.Fliplr(0.3),
        iaa.Flipud(0.3),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-20, 20),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        iaa.ChangeColorTemperature(kelvin=(1000, 11000)),
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


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path, seq):
    start_time = time.time()
    counter = 0

    dirs = list(input_path.iterdir())

    if len(dirs) > 200:
        print(f"Skipping {input_path}")
        return

    for file in dirs:
        if file.is_file() and is_image(file.name):
            image = seq(image=numpy.array(Image.open(file)))
            im = Image.fromarray(image)
            im.save(output_path / ("6_" + file.name))

    seconds_elapsed = time.time() - start_time
    logging.info(
        f"Processing path {input_path} took {seconds_elapsed} seconds, "
        f"{1000 * (seconds_elapsed / counter) if counter != 0 else 0} ms per image."
    )


def process_recursive(input_path: Path, output_path: Path, executor, seq):
    output_path.mkdir(exist_ok=True)
    dirs_to_process = []

    for file in input_path.iterdir():
        if file.is_dir():
            dirs_to_process.append(file)

    futures = []
    for directory in dirs_to_process:
        sub_out_path = (output_path / directory.name)
        futures += process_recursive(directory, sub_out_path, executor, seq)

    futures.append(executor.submit(process_images_in_path, input_path, output_path, seq))
    return futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment data')
    parser.add_argument('-i' '--input_path', required=True, help='A path to a directory containing images to process.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='An output path.', type=str, dest='output')
    parser.add_argument('-s', '--size', required=True, help='A size of a resized image.')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process images in the input_path and its subdirectories.')
    args = parser.parse_args()

    if args.recursive:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = process_recursive(Path(args.input), Path(args.output), executor,
                                        get_processing_sequence(int(args.size)))
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output), get_processing_sequence(int(args.size)))

## TESTS
# import matplotlib.pyplot as plt
#
# processing = get_processing_sequence(244)
#
# for file in Path("/backup/TEST_AUG").iterdir():
#     start_time = time.time()
#     image_before = Image.open(file)
#     image_after = processing(image=numpy.asarray(image_before))
#     seconds_elapsed = time.time() - start_time
#     print(
#         f"Processing path took {seconds_elapsed} seconds, "
#     )
#     f, ax = plt.subplots(1, 2)
#     ax[0].imshow(image_before)
#     ax[1].imshow(image_after)
#
#     f.savefig(f"output/aug_{file.name}")
