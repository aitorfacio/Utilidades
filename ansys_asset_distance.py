import json
from math import sqrt
from argparse import ArgumentParser


def calculate_distance(filepath):
    with open(filepath, 'r') as file:
        asset_data = json.load(file)
        for ship, data in asset_data.items():
            data['distance'] = []
            for index, pos in enumerate(data['position_y']):
                distance = -1
                if pos == 0:
                    x = data['position_x'][index]
                    z = data['position_z'][index]
                    distance =  sqrt( x ** 2 + z ** 2)
                data['distance'].append(distance)
        return asset_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', help='Input JSON data.', required=True)
    parser.add_argument('-o', '--output', help='File where the calculated data will be saved.')

    args = parser.parse_args()
    calculated_data = calculate_distance(args.data)
    if args.output:
        with open(args.output, 'w') as output_file:
            json.dump(calculated_data, output_file)
    else:
        print(calculated_data)

