from pathlib import Path

import argparse


def list_dir_names(path: Path, level=0):
    for directory in path.iterdir():
        if directory.is_dir():
            print(f"[{level}] {directory.name}")
            list_dir_names(directory, level + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List directory names recursively')
    parser.add_argument('-i' '--input_path', required=True, help='An input path for starting point.',
                        type=str, dest='input')
    args = parser.parse_args()

    list_dir_names(Path(args.input))
