import argparse
import json
import requests
import re

BRICK_LINK_LIBRARY_ADDRESS_BASE = "https://www.bricklink.com/v2/catalog/catalogitem.page?P="


def get_names_from_json_parts_file(file_name):
    parts = json.load(open(file_name))
    return set(part["base_file_name"] for part in parts["parts"])


def get_names_from_text_file(file_name):
    parts = [line.strip() for line in open(file_name)]
    return parts


def get_alternate_names(part_number):
    address = BRICK_LINK_LIBRARY_ADDRESS_BASE + str(part_number)
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:87.0) Gecko/20100101 Firefox/87.0',
    }
    response = requests.get(address, headers=headers)
    return find_alternate_numbers(response.text)


def find_alternate_numbers(text: str):
    result = re.search("Alternate Item No: <span.*>(.+?)</span>", text)
    if result:
        return result.group(1).replace(',', '')
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetches all common names for bricks listed in an input file.')
    parser.add_argument('-i' '--input_file', required=True, help='A path to a file containing list of bricks.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_file', required=True, help='A path to an output file.', type=str, dest='output')
    args = parser.parse_args()

    if args.input.split('.')[-1] == 'json':
        names = get_names_from_json_parts_file(args.input)
    else:
        names = get_names_from_text_file(args.input)

    with open(args.output, 'a') as output_file:
        for name in names:
            alternate_names = get_alternate_names(name)
            if len(alternate_names) > 0:
                print(f"Found alternate names: {alternate_names} for {name}.")
            else:
                print(f"No alternate names for {name} found.")
            output_file.write(f"{name} {alternate_names}\n")
            output_file.flush()
