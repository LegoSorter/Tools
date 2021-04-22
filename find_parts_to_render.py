import argparse as argparse
import json
import requests

from pathlib import Path
from group_common_names import get_alternate_names


def get_unique_names_from_file(input_file: Path):
    all_names = set()

    with open(input_file) as file:
        for line in file:
            all_names.add(line.strip())

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
    parser.add_argument('-sc', '--search_common_names', help='Search for alternative names if missing.',
                        dest='search_common')
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
    found_names.union(missing_parts_general.difference(missing_after_second_try))

    print(f"Couldn't find following parts in the json parts list: "
          f"\n\n{missing_after_second_try},\n\n")

    if args.search_common:
        print(f"searching for alternative names.")
        alternate_names_all = set()
        for missing_part in missing_after_second_try:
            alternate_names = get_alternate_names(missing_part).split(' ')
            print(f"Found alternate names {alternate_names} for {missing_part}")
            for new_name in alternate_names:
                alternate_names_all.add(new_name)

        unknown = get_missing_names_in_parts_file(alternate_names_all, 'parts.json')
        found_among_alternate_names = alternate_names_all.difference(unknown)

        found_names = found_names.union(found_among_alternate_names)

        print(f"Found {len(found_names)} from {len(names)}")

    with open('parts.json') as file:
        found_parts_json = {
            'parts': [part for part in json.load(file)['parts'] if part['base_file_name'] in found_names]}
        with open(args.output, 'w') as output_file:
            json.dump(found_parts_json, output_file)
