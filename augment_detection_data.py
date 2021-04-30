import argparse
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
from imgaug import BoundingBox, BoundingBoxesOnImage


def to_label_file(filename, path, image_width, image_height, bbs_xyxy_array):
    objects = ""
    for coord in bbs_xyxy_array:
        objects += get_object(*coord)

    return f"""<annotation>
            <folder>images</folder>
            <filename>{filename}</filename>
            <path>{path}</path>
            <source>
                    <database>LegoSorterPGR</database>
            </source>
            <size>
                    <width>{image_width}</width>
                    <height>{image_height}</height>
                    <depth>3</depth>
            </size>
            <segmented>0</segmented>
            {objects}
    </annotation>"""


def get_object(x1, y1, x2, y2):
    return f"""<object>
                    <name>lego</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                            <xmin>{int(x1)}</xmin>
                            <ymin>{int(y1)}</ymin>
                            <xmax>{int(x2)}</xmax>
                            <ymax>{int(y2)}</ymax>
                    </bndbox>
            </object>"""


def augment_image(image, bbs, image_size=640):
    bbs = [BoundingBox(*bb) for bb in bbs]
    bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
    resize = iaa.Sequential([iaa.Resize({"longer-side": image_size, "shorter-side": "keep-aspect-ratio"}),
                             iaa.PadToFixedSize(width=image_size, height=image_size)])

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Crop(percent=(0, 0.05)), # random crops
        # Gaussian blur with random sigma between 0 and 0.2.
        iaa.GaussianBlur(sigma=(0, 0.2)),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.8, 1.2)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 30% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.3),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (1, 1.1), "y": (1, 1.1)},
            #             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=[90, -90, 180, 88, -92, 178, 182],
            shear={"x": (-5, 5), "y": (-5, 5)},
            cval=(0, 255)
        ),
    ], random_order=True)  # apply augmenters in order

    image, bbs = resize(image=image, bounding_boxes=bbs)
    image, bbs = seq(image=image, bounding_boxes=bbs)
    bbs = bbs.remove_out_of_image().clip_out_of_image()
    return image, bbs


def read_data_from_xml_file(path: Path):
    bbs = []
    tree = ET.parse(path)
    image_size = (int(tree.find('size').find('width').text), int(tree.find('size').find('height').text))
    image_name = tree.find('filename').text

    for obj in tree.iter('object'):
        bb = obj.find('bndbox')
        coordinates = (int(bb.find('xmin').text), int(bb.find('ymin').text), int(bb.find('xmax').text), int(bb.find('ymax').text))
        bbs.append(coordinates)

    return image_name, image_size, bbs


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path):
    start_time = time.time()
    counter = 0

    for file in input_path.iterdir():
        if file.is_file() and is_image(file.name):
            xml_file_name = file.name.split(".")[0] + ".xml"
            xml_path = file.parent / xml_file_name
            image_data = read_data_from_xml_file(xml_path)
            image, bbs = augment_image(image=np.array(Image.open(file)), bbs=image_data[2])
            dest_path_img = output_path / file.name
            dest_path_xml = output_path / xml_file_name
            xml_file = to_label_file(file.name, str(dest_path_img), image_width=image_data[1][0],
                                     image_height=image_data[1][1], bbs_xyxy_array=bbs.to_xyxy_array())
            Image.fromarray(image).save(dest_path_img)
            with open(str(dest_path_xml), "w") as label_xml:
                label_xml.write(xml_file)

    seconds_elapsed = time.time() - start_time
    print(
        f"Processing path {input_path} took {seconds_elapsed} seconds, "
        f"{1000 * (seconds_elapsed / counter) if counter != 0 else 0} ms per image."
    )


def process_recursive(input_path: Path, output_path: Path, executor):
    output_path.mkdir(exist_ok=True)
    dirs_to_process = []

    for file in input_path.iterdir():
        if file.is_dir():
            dirs_to_process.append(file)

    futures = []
    for directory in dirs_to_process:
        sub_out_path = (output_path / directory.name)
        futures += process_recursive(directory, sub_out_path, executor)

    futures.append(executor.submit(process_images_in_path, input_path, output_path))
    return futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment detection data, for each image file there has '
                                                 'to be a corresponding xml file in VOC format')
    parser.add_argument('-i' '--input_path', required=True, help='A path to a directory containing images to process.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='An output path.', type=str, dest='output')
    parser.add_argument('-s', '--size', required=True, help='A size of a resized image.')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process images in the input_path and its subdirectories.')
    args = parser.parse_args()

    if args.recursive:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = process_recursive(Path(args.input), Path(args.output), executor)
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output))
