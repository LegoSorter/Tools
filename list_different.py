from pathlib import Path

import argparse


def print_different(base_names, compare_names):
    for name in base_names:
        if name not in compare_names:
            print(f"\t{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List differences between two directories')
    parser.add_argument('first', help='First directory to compare')
    parser.add_argument('second', help='Second directory to compare')
    args = parser.parse_args()

    first_dir = Path(args.first)
    second_dir = Path(args.second)

    first_names = [p.name for p in first_dir.iterdir()]
    second_names = [p.name for p in second_dir.iterdir()]

    print(f"Names that are in {first_dir} but not in {second_dir}:")

    print_different(first_names, second_names)

    print(f"Names that are in {second_dir} but not in {first_dir}:")

    print_different(second_names, first_names)
