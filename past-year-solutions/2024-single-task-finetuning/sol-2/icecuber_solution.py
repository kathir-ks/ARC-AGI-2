"""
This script has been copied and modified from the following source:
https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions

This script only runs icecuber solution
"""
import sys
import argparse
import os
import json
import time
from subprocess import Popen, PIPE, STDOUT
from tqdm.auto import tqdm

import warnings # suppress warnings
warnings.filterwarnings('ignore')


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    run_icecuber_solver(
        dataset_filepath=args.dataset_filepath, solution_filepath=args.output_filepath)


def parse_args(args):
    epilog = """
    """
    description = """
Tries to solve the ARC task using icecuber solution.
https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--dataset_filepath', required=True, help='Path to json file with the dataset that we want to solve')
    parser.add_argument('--output_filepath', required=True, help='Path to json file that will be created with the solution')
    args = parser.parse_args(args)
    print(args)
    return args


"""Icecube"""
def adapt_arc24_files_to_arc20_format(json_file_path, output_dir):
    # Load the JSON content
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create the 'test' directory
    os.makedirs(output_dir, exist_ok=True)

    # Split the JSON content into individual files
    for task_id, task_data in tqdm(data.items(), total=len(data), desc='Splitting tasks into multiple files'):
        output_file_path = os.path.join(output_dir, f'{task_id}.json')
        with open(output_file_path, 'w') as output_file:
            json.dump(task_data, output_file, indent=4)


def mySystem(cmd):
    print(cmd)
    t0 = time.time()
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    for line in iter(process.stdout.readline, b''):
        print(line.decode("utf-8"), end='')
    assert(process.wait() == 0)
    print(f'{time.time() - t0:.1f} seconds took to run {cmd}')


def translate_submission_from_old_csv_format_to_new_json_format(input_filepath, output_filepath):
    # Read the original submission file
    with open(input_filepath, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # Skip the header line
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # Split predictions based on ' '

        # Take only the first two predictions
        if len(predictions) > 2:
            predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # Check if pred is not an empty string
                pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_1 = processed_predictions[0] if len(processed_predictions) > 0 else []
        attempt_2 = processed_predictions[1] if len(processed_predictions) > 1 else []

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        attempt_dict = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)
        else:
            submission_dict[task_id].append(attempt_dict)

    # Write to the new json file
    with open(output_filepath, 'w') as file:
        json.dump(submission_dict, file, indent=4)


def check_icecube_dataset_has_correct_version():
    if open("../input/arc-solution-source-files-by-icecuber/version.txt").read().strip() == "671838222":
        print("Dataset has correct version")
    else:
        print("Dataset version not matching!")
        assert(0)


def run_icecuber_solver(dataset_filepath, solution_filepath):
    print("\n\n\t\tRunning Icecube solver")
    adapt_arc24_files_to_arc20_format(
        json_file_path=dataset_filepath,
        output_dir='/kaggle/working/abstraction-and-reasoning-challenge/test')
    check_icecube_dataset_has_correct_version()
    mySystem("cp -r ../input/arc-solution-source-files-by-icecuber ./absres-c-files")
    mySystem("cd absres-c-files; make -j")
    mySystem("cd absres-c-files; python3 safe_run.py")
    mySystem("cp absres-c-files/submission_part.csv old_icecube_submission.csv")
    mySystem("tar -czf store.tar.gz absres-c-files/store")
    mySystem("rm -r absres-c-files")
    translate_submission_from_old_csv_format_to_new_json_format(
        '/kaggle/working/old_icecube_submission.csv', solution_filepath)
    print("\t\tIcecube solver completed\n\n")


if __name__ == '__main__':
    main()
