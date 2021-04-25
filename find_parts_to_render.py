import argparse as argparse
import json
import requests

from pathlib import Path


def get_unique_names_from_file(input_file: Path):
    all_names = set()

    with open(input_file) as file:
        for line in file:
            all_names.update([name.strip() for name in line.split(' ') if len(name.strip()) > 0])

    return all_names


def get_missing_names_in_parts_file(names, parts_file_json):
    with open(parts_file_json) as json_parts_file:
        parts_list = json.load(json_parts_file)['parts']

        all_parts = set()
        for part in parts_list:
            all_parts.add(part['base_file_name'])

        return names.difference(all_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepares a json list of lego bricks ready to render.')
    parser.add_argument('-i' '--input_file', required=True,
                        help='A file containing a list of bricks, with repeated names and common names',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_file', required=True, help='An output path.', type=str, dest='output')
    args = parser.parse_args()

    names = get_unique_names_from_file(Path(args.input))
    missing_parts = get_missing_names_in_parts_file(names, 'parts.json')

    found_names = names.difference(missing_parts)

    print(f"Couldn't find following parts in the json parts list: "
          f"\n\n{missing_parts},\n\n"
          f"searching for general names (without letters at the end of a name)")

    missing_parts_general = set()
    for missing_part in missing_parts:
        if missing_part[-1] == 'a' or missing_part[-1] == 'b' or missing_part[-1] == 'c':
            missing_part = missing_part[:-1]
            missing_parts_general.add(missing_part)

    missing_after_second_try = get_missing_names_in_parts_file(missing_parts_general, 'parts.json')
    found_names = found_names.union(missing_parts_general.difference(missing_after_second_try))

    print(f"Couldn't find following parts in the json parts list: "
          f"\n\n{missing_after_second_try},\n\n"
          f"Searching in unofficial parts")

    with open('all_parts.json') as file:
        found_parts_json = {
            'parts': [part for part in json.load(file)['parts'] if
                      part['base_file_name'] in missing_after_second_try or part[
                          'file_name'] in missing_after_second_try]}
        with open('unofficial_' + args.output, 'w') as output_file:
            json.dump(found_parts_json, output_file)

    with open('parts.json') as file:
        found_parts_json = {
            'parts': [part for part in json.load(file)['parts'] if part['base_file_name'] in found_names]}
        with open(args.output, 'w') as output_file:
            json.dump(found_parts_json, output_file)
