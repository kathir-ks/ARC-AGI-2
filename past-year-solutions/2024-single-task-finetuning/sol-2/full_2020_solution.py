import sys
import os
import argparse

from arc24.logging import logging


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    script_folder = os.path.realpath(os.path.dirname(__file__))

    logging.info('Running icecuber solution')
    cmd = f'python {script_folder}/icecuber_solution.py --dataset_filepath {args.dataset_filepath} --output_filepath {args.icecuber_output_filepath}'
    os.system(cmd)

    logging.info('Running dsl solution')
    cmd = f'python {script_folder}/program_search_dsl.py --dataset_filepath {args.dataset_filepath} --output_filepath {args.dsl_output_filepath}'
    os.system(cmd)


def parse_args(args):
    epilog = """
    """
    description = "Runs the full 2020 solution, first icecuber and then dsl"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--dataset_filepath', required=True, help='Path to json file with the dataset that we want to solve')
    parser.add_argument('--icecuber_output_filepath', required=True, help='Path to json file that will be created with the solution')
    parser.add_argument('--dsl_output_filepath', required=True, help='Path to json file that will be created with the solution')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
