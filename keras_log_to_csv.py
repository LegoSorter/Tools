import argparse
import csv
import re

# variables = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
from pathlib import Path


def extract_from_file(filename, variables):
    print(variables)
    results = []
    with open(filename, 'r') as file:
        for line in file:
            if len(line) == 0 or line.startswith('Epoch'):
                continue

            line_results = []
            for variable in variables:
                search = re.search(f' {variable}: (.*?)(\s|$)', line)
                if search:
                    line_results.append(search.group(1))
                else:
                    line_results.append('None')
            results.append(line_results)
    return results


def save_to_csv(results, output_file: Path):
    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts results from keras logs.')
    parser.add_argument('-i' '--input_file', required=True, help='A path to a log file containing keras results',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_file', required=True, help='A path to an output file.', type=str, dest='output')
    parser.add_argument('-v', '--variable', required=True, nargs='+',
                        help='Specify variable to extract, can be multiple',
                        dest='variables')
    args = parser.parse_args()

    results = extract_from_file(args.input, args.variables)
    save_to_csv([args.variables, *results], Path(args.output))
