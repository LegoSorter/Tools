import argparse
import shutil

from pathlib import Path

SUBCLASSES_EXTENSIONS = ["a", "b", "c"]


def get_names_in_groups(file_name):
    groups = []

    with open(file_name) as file:
        for line in file:
            groups.append([name.strip() for name in line.split(' ') if len(name.strip()) > 0])

    return groups


def copy_directory(input: Path, output: Path, files_only):
    output.mkdir(exist_ok=True, parents=True)
    if files_only:
        for i in input.iterdir():
            if i.is_file():
                shutil.copy(i, output / i.name)
    else:
        shutil.copytree(str(input), str(output), dirs_exist_ok=True)


def extract_dataset(names, output_path: Path, input_path: Path, files_only, include_subclasses, skip_if_exists):
    common_name = names[0]

    for name in names:
        src = input_path / name
        copy_if_exists(common_name, files_only, output_path, src, skip_if_exists)

        if include_subclasses:
            for ext in SUBCLASSES_EXTENSIONS:
                src = input_path / (name + ext)
                copy_if_exists(common_name, files_only, output_path, src, skip_if_exists)


def copy_if_exists(common_name, files_only, output_path, src, skip_if_exists):
    if not src.exists():
        print(f"Couldn't find a directory {src}")
        return

    destination = output_path / common_name

    if skip_if_exists and destination.exists():
        print(f"Skipping copying from {src} to {destination} because the directory already exists.")
        return

    destination.mkdir(parents=True, exist_ok=True)
    copy_directory(src, destination, files_only)


def extract_dataset_recursively(names, output_path: Path, input_path: Path, files_only, include_subclasses,
                                skip_if_exists):
    for directory in input_path.iterdir():
        if not directory.is_dir():
            continue

        if directory.name in names or \
                (include_subclasses and directory.name[-1] in SUBCLASSES_EXTENSIONS and directory.name[:-1] in names):
            common_name = names[0]
            destination = output_path / common_name

            if skip_if_exists and destination.exists():
                print(f"Skipping copying from {directory} to {destination} because the directory already exists.")
                continue

            copy_directory(directory, destination, files_only)

        extract_dataset_recursively(names, output_path, directory, files_only, include_subclasses, skip_if_exists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetches dataset for bricks listed in file from the specified directory. '
                    'Numbers in the same line indicate that the output for them has to be merged. '
                    'For example, line \'30413 15207 35255 43337\' means that all images from directories '
                    'named as specified, will be copied to a directory \'30413\' in the output directory')
    parser.add_argument('-i', '--input_path', required=True, help='A path to an input directory containing bricks.',
                        dest='input')
    parser.add_argument('-b' '--bricks', required=True, help='A path to a file containing list of bricks.',
                        type=str, dest='bricks')
    parser.add_argument('-o', '--output_path', required=True, help='A path to an output directory.', type=str,
                        dest='output')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Whether to process the input directory recursively, false by default')
    parser.add_argument('-fo' '--files_only', action='store_true', dest='files_only',
                        help='Whether to copy the whole found directory or only files inside it. False by default.')
    parser.add_argument('-sub', '--sub_classes', action='store_true', dest='sub',
                        help='Whether to look for subclasses of a name, for example, '
                             'when requesting a dataset for a class 3040, it will grab 3040b also.')
    parser.add_argument('-s', '--skip_existing', action='store_true', dest='skip',
                        help='Skip if a destination directory exists.')
    args = parser.parse_args()

    groups = get_names_in_groups(args.bricks)
    for parts_group in groups:
        if args.recursive:
            extract_dataset_recursively(parts_group, Path(args.output), Path(args.input), args.files_only, args.sub,
                                        args.skip)
        else:
            extract_dataset(parts_group, Path(args.output), Path(args.input), args.files_only, args.sub, args.skip)
