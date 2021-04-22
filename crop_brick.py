import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
import logging

from pathlib import Path


def find_contours(image, threshold, color_option=cv.COLOR_BGR2GRAY):
    image_color = cv.cvtColor(image, color_option)
    image_color = cv.blur(image_color, (3, 3))
    canny_output = cv.Canny(image_color, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours


def get_image_from_path(path: Path):
    return cv.imread(str(path.absolute()))


def rectangle_area(contour):
    rectangle = cv.boundingRect(contour)
    return rectangle[2] * rectangle[3]


def draw_contours(image, contours, option=2):
    bounding_boxes = [cv.boundingRect(x) for x in contours]

    for rect in bounding_boxes:
        color = (256, 256, 256)
        cv.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), color,
                     option)

    return image


def get_bounding_box(image, threshold=25, color_option=cv.COLOR_BGR2HLS):
    contours_original_image = find_contours(image, threshold, color_option)
    drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawing = draw_contours(drawing, contours_original_image, cv.FILLED)
    drawing = cv.copyMakeBorder(drawing, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, [0, 0, 0])
    contours_mask = find_contours(drawing, threshold)
    if len(contours_mask) == 0:
        raise Exception("Empty render!")

    return cv.boundingRect(max(contours_mask, key=rectangle_area))


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path, min_size):
    start_time = time.time()
    counter = 0

    for file in input_path.iterdir():
        if file.is_file() and is_image(file.name):
            try:
                cropped_image = find_and_crop_brick(file, threshold=20, color_option=None)
                if cropped_image.shape[0] * cropped_image.shape[1] >= min_size:
                    save_image(cropped_image, output_path / file.name)
                    counter += 1
                else:
                    raise Exception("Detected brick too small")
            except Exception:
                logging.error(f"Got an empty render - {str(file)}. Trying to decrease the threshold!")
                try:
                    cropped_image = find_and_crop_brick(file, threshold=5, color_option=cv.COLOR_BGR2HLS)
                    if cropped_image.shape[0] * cropped_image.shape[1] >= min_size:
                        save_image(cropped_image, output_path / file.name)
                        counter += 1
                    else:
                        raise Exception("Detected brick too small")
                except Exception:
                    logging.error(f"Got an empty render after decreasing the threshold - {str(file)}. Skipping...")

    seconds_elapsed = time.time() - start_time
    logging.info(
        f"Processing path {input_path} took {seconds_elapsed} seconds, "
        f"{1000 * (seconds_elapsed / counter) if counter != 0 else 0} ms per image."
    )


def process_recursive(input_path: Path, output_path: Path, executor, min_size):
    output_path.mkdir(exist_ok=True)
    dirs_to_process = []

    for file in input_path.iterdir():
        if file.is_dir():
            dirs_to_process.append(file)

    futures = []
    for directory in dirs_to_process:
        sub_out_path = (output_path / directory.name)
        futures.append(*process_recursive(directory, sub_out_path, executor, min_size))

    futures.append(executor.submit(process_images_in_path, input_path, output_path, min_size))
    return futures


def save_image(cropped_image, output_path: Path):
    cv.imwrite(str(output_path), cropped_image)


def find_and_crop_brick(file, threshold=20, color_option=None):
    image = get_image_from_path(file)
    x, y, w, h = get_bounding_box(image, threshold=threshold, color_option=color_option)
    cropped_image = image[y: y + h, x: x + w]
    return cropped_image


# file = Path("/backup/RENDER_4/10928/10928_Black_2_1619057697.jpeg")
# find_and_crop_brick(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a foreground from the specified input.')
    parser.add_argument('-i' '--input_path', required=True, help='A path to a directory containing images to process.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='An output path.', type=str, dest='output')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process images in the input_path and its subdirectories.')
    parser.add_argument('-m', '--min_size', required=True, help='Minimum area ofcropped image', dest='minsize')
    args = parser.parse_args()

    logging.basicConfig(filename='crop_brick_render_3.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if args.recursive:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = process_recursive(Path(args.input), Path(args.output), executor, args.minsize)
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output), args.minsize)
