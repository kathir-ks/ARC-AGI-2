import sys
import argparse
import json
from collections import defaultdict
import numpy as np

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.input_filepath.endswith('_task_results.json'):
        print('Using the second voting implementation to solve ties with logprob')
        with open(args.input_filepath) as f:
            task_results = json.load(f)
        solutions = select_most_voted_solutions_solving_ties_with_logprob(task_results, args.n)
    else:
        print('Using the first voting implementation, ties will be solved randomly')
        with open(args.input_filepath) as f:
            solutions = json.load(f)
        solutions = select_most_voted_solutions(solutions, args.n)
    args.output_filepath = args.output_filepath or args.input_filepath.replace('.json', '_voting.json')
    with open(args.output_filepath, 'w') as f:
        json.dump(solutions, f)


# First implementation that simply does voting
def select_most_voted_solutions(solutions, n):
    most_voted_solutions = dict()
    for task_id, task_solutions in solutions.items():
        most_voted_solutions[task_id] = list()
        for sample_solutions in task_solutions:
            valid_solutions = [solution for solution in sample_solutions.values() if solution]
            if valid_solutions:
                unique_solutions, counts = get_unique_matrices_and_counts_sorted(valid_solutions)
                most_voted_sample_solutions = {f'attempt_{i+1}': solution for i, solution in enumerate(unique_solutions[:n])}
            else:
                most_voted_sample_solutions = {}
            if len(most_voted_sample_solutions) < n:
                for i in range(len(most_voted_sample_solutions), n):
                    most_voted_sample_solutions[f'attempt_{i+1}'] = []
            most_voted_solutions[task_id].append(most_voted_sample_solutions)
    return most_voted_solutions


def get_unique_matrices_and_counts_sorted(matrices):
    # Dictionary to store counts of unique matrices
    matrix_count = defaultdict(int)

    # Convert each matrix to a tuple of tuples and count occurrences
    for matrix in matrices:
        matrix_tuple = tuple(tuple(row) for row in matrix)
        matrix_count[matrix_tuple] += 1

    # Sort the matrices by their counts in descending order
    sorted_matrices = sorted(matrix_count.items(), key=lambda item: item[1], reverse=True)

    # Convert the sorted tuples back to lists for easier interpretation
    unique_matrices = [list([list(row) for row in matrix]) for matrix, _ in sorted_matrices]
    counts = [count for _, count in sorted_matrices]

    return unique_matrices, counts


# Second implementation that solves ties using the solution with highest logprob
def select_most_voted_solutions_solving_ties_with_logprob(task_outputs, n, tie_breaking_metric='mean_cumulative_logprob'):
    grouped_predictions = group_predictions(task_outputs)
    most_voted_solutions = dict()
    for task_id, task_solutions in grouped_predictions.items():
        most_voted_solutions[task_id] = [{f'attempt_{i+1}': [] for i in range(n)} for _ in range(max(task_solutions.keys())+1)]
        for test_idx, sample_solutions in task_solutions.items():
            sample_solutions = task_solutions[test_idx]
            for solution in sample_solutions.values():
                solution['ranking'] = (len(solution[tie_breaking_metric]), np.mean(solution[tie_breaking_metric]))
            sorted_sample_solutions = sorted(sample_solutions.values(), key=lambda x: x['ranking'], reverse=True)

            most_voted_sample_solutions = {f'attempt_{i+1}': output['grid'] for i, output in enumerate(sorted_sample_solutions[:n])}
            if len(most_voted_sample_solutions) < n:
                for i in range(len(most_voted_sample_solutions), n):
                    most_voted_sample_solutions[f'attempt_{i+1}'] = []
            most_voted_solutions[task_id][test_idx] = most_voted_sample_solutions
    add_empty_solutions(task_outputs, most_voted_solutions, n)
    return most_voted_solutions


def group_predictions(task_outputs):
    """
    Group predictions by task_id, test_idx and grid
    Allows to know how many times a grid was predicted and the cumulative logprob of those predictions
    """
    grouped_predictions = dict()
    for output in task_outputs:
        if not output['grid']:
            # discard empty grids
            continue

        task_id = output['task_id']
        test_idx = output['idx']
        grid_key = str(output['grid'])
        if task_id not in grouped_predictions:
            grouped_predictions[task_id] = dict()
        if test_idx not in grouped_predictions[task_id]:
            grouped_predictions[task_id][test_idx] = dict()
        if grid_key not in grouped_predictions[task_id][test_idx]:
            grouped_predictions[task_id][test_idx][grid_key] = dict(
                grid=output['grid'], cumulative_logprob=[], mean_cumulative_logprob=[])

        grouped_predictions[task_id][test_idx][grid_key]['cumulative_logprob'].append(output['cumulative_logprob'])
        grouped_predictions[task_id][test_idx][grid_key]['mean_cumulative_logprob'].append(output['cumulative_logprob'] / output['n_tokens'])
    return grouped_predictions


def add_empty_solutions(task_outputs, solutions, n):
    empty_response = {f'attempt_{i+1}': [] for i in range(n)}
    for output in task_outputs:
        task_id = output['task_id']
        test_idx = output['idx']
        if task_id not in solutions:
            solutions[task_id] = [empty_response]
        if test_idx >= len(solutions[task_id]):
            solutions[task_id].append(empty_response)
    return solutions


def parse_args(args):
    epilog = """
    """
    description = """
Creates a new solution by voting and choosing the most common solutions for each task.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--input_filepath', required=True, help='Path to input json file with the solutions')
    parser.add_argument('--output_filepath', required=False, help='Path to output json file with the solutions')
    parser.add_argument('--n', type=int, default=2, help='Number of final solutions to keep')
    args = parser.parse_args(args)
    print(description)
    print(args)
    return args


if __name__ == '__main__':
    main()
