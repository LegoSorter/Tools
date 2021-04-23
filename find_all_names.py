from pathlib import Path
from typing import List

import requests
import argparse
import json


class RebrickableClient:
    base_link = 'https://rebrickable.com/api/v3'

    def __init__(self, access_key):
        self.access_key = access_key

    def get_lego_set_parts(self, lego_set_number: str):
        return requests.get(f"{self.base_link}/lego/sets/{lego_set_number}/parts?page_size=10000",
                            headers={'Authorization': f"key {self.access_key}"}).json()

    def get_lego_part_info(self, lego_part_number: str):
        return requests.get(f"{self.base_link}/lego/parts/{lego_part_number}/?inc_part_details=1",
                            headers={'Authorization': f"key {self.access_key}"}).json()

    def get_lego_parts_info(self, lego_parts_numbers: List[str]):
        parts_nums = ','.join(lego_parts_numbers)
        return requests.get(f"{self.base_link}/lego/parts/?part_nums={parts_nums}",
                            headers={'Authorization': f"key {self.access_key}"}).json()

    def get_all_lego_part_ids(self, lego_part_number: str):
        part_info = self.get_lego_part_info(lego_part_number)
        return set(self.__get_ids_from_external(part_info['external_ids']) + [lego_part_number])

    def get_all_lego_parts_ids(self, lego_parts_numbers: List[str]):
        parts_json = self.get_lego_parts_info(lego_parts_numbers)
        return self.__extract_ids_from_results(parts_json)

    def get_lego_part_ids_from_set(self, lego_set_number: str):
        parts_json = self.get_lego_set_parts(lego_set_number)
        return self.__extract_ids_from_results(parts_json)

    def __extract_ids_from_results(self, parts_json):
        all_parts = dict()
        for part in parts_json['results']:
            part_info = part['part'] if 'part' in part else part
            part_num = part_info['part_num']
            external_ids = part_info['external_ids']
            ids = self.__get_ids_from_external(external_ids)

            if part_num in all_parts:
                continue

            all_parts[part_num] = set(ids + [part_num])
        return all_parts

    @staticmethod
    def __get_ids_from_external(external_ids):
        brick_owl_nums = external_ids['BrickOwl'] if 'BrickOwl' in external_ids else []
        ldraw_nums = external_ids['LDraw'] if 'LDraw' in external_ids else []
        lego_numbers = external_ids['LEGO'] if 'LEGO' in external_ids else []
        return brick_owl_nums + ldraw_nums + lego_numbers


def read_input_file(file_path: Path):
    with open(file_path, 'r') as file:
        all_names = set()
        for line in file:
            all_names.update([name.strip() for name in line.split(' ') if len(name) > 0])
        return all_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts results from keras logs.')
    parser.add_argument('-i' '--input_file', required=True,
                        help='A file containing list of parts or sets to find alternative names for.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_file', required=True, help='A path to an output file.', type=str, dest='output')
    parser.add_argument('-k', '--key', required=True, help='A Rebrickable authorization key')
    parser.add_argument('-s', '--set', action='store_true', help='An input file contains sets.')
    args = parser.parse_args()
    client = RebrickableClient(args.key)

    all_names = list(read_input_file(args.input))

    if args.set:
        all_ids = dict()
        for set_name in all_names:
            all_ids = {**all_ids, **client.get_lego_part_ids_from_set(set_name)}
    else:
        all_ids = client.get_all_lego_parts_ids(all_names)

    with open(args.output, 'w') as file:
        for values in all_ids.values():
            file.write(' '.join(values) + '\n')
