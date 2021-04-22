import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy
import imgaug.augmenters as iaa

from pathlib import Path
from PIL import Image


def get_processing_sequence(image_size):
    return iaa.Sequential([
        iaa.Resize({"longer-side": image_size, "shorter-side": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(width=image_size, height=image_size)
    ])


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path, seq):
    start_time = time.time()
    counter = 0

    for file in input_path.iterdir():
        if file.is_file() and is_image(file.name):
            image = seq(image=numpy.array(Image.open(file)))
            im = Image.fromarray(image)
            im.save(output_path / file.name)

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
        futures.append(*process_recursive(directory, sub_out_path, executor, seq))

    futures.append(executor.submit(process_images_in_path, input_path, output_path, seq))
    return futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images keeping aspect ratio')
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
                                        get_processing_sequence(args.size))
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output), get_processing_sequence(args.size))
