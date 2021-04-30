from pathlib import Path

import shutil
import xml.etree.ElementTree as ET

import argparse


def list_directories(path):
    directories = []
    for path in path.glob("*"):
        if path.is_dir():
            directories.append(path)
            continue

    return directories


def convert(xmin, ymin, xmax, ymax):
    size = (int(xmax) - int(xmin), int(ymax) - int(ymin))
    center = ((int(xmin) + int(xmax)) / 2, (int(ymin) + int(ymax)) / 2)

    return center, size


def read_bbs_from_file(path: Path):
    bbs = []
    tree = ET.parse(path)
    image_size = (int(tree.find('size').find('width').text), int(tree.find('size').find('height').text))

    for obj in tree.iter('object'):
        bb = obj.find('bndbox')
        coords = (bb.find('xmin').text, bb.find('ymin').text, bb.find('xmax').text, bb.find('ymax').text)
        bbs.append(coords)

    converted_bbs = []
    for bb in bbs:
        converted = convert(*bb)
        scaled = (converted[0][0] / image_size[0], converted[0][1] / image_size[1], converted[1][0] / image_size[0],
                  converted[1][1] / image_size[1])
        converted_bbs.append(scaled)

    return converted_bbs


def get_image_name(path: Path):
    tree = ET.parse(path)
    return tree.find('filename').text


def create_annotation_file(path: Path, bbs):
    if not path.parent.exists():
        path.parent.mkdir()

    if not (path.parent / "classes.txt").exists():
        with open(path.parent / "classes.txt", "w") as classes:
            classes.write("lego")

    with open(path, "w+") as annotation_file:
        for bb in bbs:
            annotation_file.write(f'0 {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n')


def get_output_path(input_path: Path, output_dir: Path):
    name = input_path.name.split(".")[0] + ".txt"
    return output_dir / input_path.parent.name / name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert xml files with bounding boxes to Yolo format and save them to the output directory '
                    'together with corresponding images.')
    parser.add_argument('-i' '--input_path', required=True, help='Input directory',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='Output directory',
                        type=str, dest='output')
    args = parser.parse_args()

    output_dir = Path(args.output)
    input_dir = Path(args.input)

    for directory in list_directories(input_dir):
        for xml_path in directory.glob("*.xml"):
            image_name = get_image_name(xml_path)
            image_path = xml_path.parent / image_name
            if not image_path.exists():
                print(f"Image {image_path} doesn't exist but is mentioned in {xml_path}!")
                continue

            bbs = read_bbs_from_file(xml_path)
            output_path = get_output_path(xml_path, output_dir)
            create_annotation_file(output_path, bbs)
            shutil.copy(image_path, output_path.parent / image_name)
