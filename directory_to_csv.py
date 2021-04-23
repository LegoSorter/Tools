import argparse
import csv

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepares a csv containing a directory name as a label (y) and images paths as a train data (x).')
    parser.add_argument('-i' '--input_path', required=True,
                        help='A root path containing dataset',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_file', required=True, help='An output file name.', type=str, dest='output')
    args = parser.parse_args()

    dataset_path = Path(args.input)

    yx_data = []

    for label_path in dataset_path.iterdir():
        if not label_path.is_dir():
            print(f"Skipping {label_path} as it's not a directory")
            continue
        for image_path in label_path.iterdir():
            if not image_path.is_file():
                print(f"Skipping {image_path} as it's not a file")
                continue
            yx_data.append((label_path.name, str(image_path)))

    with open(args.output, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label', 'image_path'])
        for yx_row in yx_data:
            writer.writerow(yx_row)


