"""
Evaluation

This is not a script, but rather a collection of functions that can be imported on a notebook
to evaluate and visualize the performance of the model.
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict

from arc24.data import load_arc_data_with_solutions
from voting import select_most_voted_solutions

# Evaluation
def analyze_number_of_predictions_per_task(data, texts):
    # TODO: update the function
    number_of_predictions = dict()
    for task_id, task in data.items():
        number_of_predictions[task_id] = len(texts[task_id]['responses'])/len(task['test'])
    plt.title('Distribution of the number of predictions per task')
    plt.hist(number_of_predictions.values(), bins=np.arange(1.5, 9))
    plt.xlabel('number of predictions')
    plt.ylabel('count')
    return number_of_predictions


def evaluate(ground_truth, solutions, verbose=True):
    """
    Computes the following metrics:

    - Accuracy
    - Correct pixels
    - Correct size
    """
    metrics = dict()
    for task_id in solutions:
        task_metrics = []
        for test_idx, test_sample in enumerate(ground_truth[task_id]['test']):
            correct_grid = test_sample['output']
            predicted_grids = list(solutions[task_id][test_idx].values())
            task_metrics.append(evaluate_predicted_grids(correct_grid, predicted_grids))
            #print_metrics(task_metrics[-1], f'{task_id}_{test_idx}')
        metrics[task_id] = average_metrics(task_metrics)
    global_metrics = average_metrics(list(metrics.values()))
    if verbose:
        print_sorted_task_metrics(metrics)
        print('\n'*2 + '# Aggregated metrics:')
        print_metrics(global_metrics)
        #plot_metrics_distribution(metrics)
        print_metrics(global_metrics)
    return global_metrics, metrics


def print_sorted_task_metrics(metrics):
    task_ids = get_sorted_task_ids(metrics)
    for task_id in task_ids:
        print_metrics(metrics[task_id], f'Task {task_id} ')


def get_sorted_task_ids(metrics, ascending=False):
    task_ids = list(metrics.keys())
    task_ids = sorted(task_ids, key=lambda x: (metrics[x]['accuracy'], metrics[x]['correct_pixels'], metrics[x]['correct_size']), reverse=not ascending)
    return task_ids


def plot_metrics_distribution(metrics):
    for key in metrics[0]:
        values = [x[key] for x in metrics]
        plt.title(f'Distribution of {key}')
        plt.hist(values, bins=np.linspace(0, 1, 10))
        plt.xlabel(key)
        plt.ylabel('count')
        plt.show()


def average_metrics(metrics):
    averaged_metrics = dict()
    if metrics:
        for key in metrics[0]:
            averaged_metrics[key] = np.mean([x[key] for x in metrics])
    return averaged_metrics


def save_metrics(metrics):
    metrics['global_metrics'] = average_metrics(list(metrics.values()))
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)


def print_metrics(metrics, prefix=''):
    text = f'{prefix}'
    for key, value in metrics.items():
        if key == 'n':
            text += f'{key}: {value}\t'
        else:
            text += f'{key}: {value:.1%}\t'
    print(text)


def evaluate_predicted_grids(correct_grid, predicted_grids):
    correct_grid = np.array(correct_grid, dtype=int)
    valid_predicted_grids = [np.array(grid, dtype=int) for grid in predicted_grids if grid]
    metrics = dict(accuracy=0,
                   correct_pixels=0,
                   max_correct_pixels=0,
                   correct_size=0,
                   any_correct_size=0,
                   pass_n=0,
                   unanswered=(len(predicted_grids) - len(valid_predicted_grids))/len(predicted_grids))
    for predicted_grid in valid_predicted_grids:
        if correct_grid.shape == predicted_grid.shape:
            if np.all(predicted_grid == correct_grid):
                metrics['pass_n'] = 1
                metrics['accuracy'] += 1./len(predicted_grids)
            metrics['correct_pixels'] += np.mean(predicted_grid == correct_grid)/len(predicted_grids)
            metrics['correct_size'] += 1./len(predicted_grids)
            metrics['max_correct_pixels'] = max(metrics['max_correct_pixels'], np.mean(predicted_grid == correct_grid))
            metrics['any_correct_size'] = 1
    return metrics


def study_effect_of_the_number_of_solutions(solutions, data, n_tries=40,
                                            ignore_metrics = ['unanswered', 'accuracy', 'correct_pixels', 'max_correct_pixels', 'accuracy', 'correct_size', 'any_correct_size'],
                                            title='',
                                            min_predictions=8):
    max_predictions = max([len(x[0]) for x in solutions.values()])
    print(f'Maximum number of predictions: {max_predictions}')
    n_predictions_range = 2**np.arange(int(np.log2(min_predictions)), int(np.log2(max_predictions) + 1))
    mean_metrics, all_metrics = [], []
    for n_predictions in n_predictions_range:
        metrics = []
        for _ in range(n_tries):
            solutions_subset = subsample_solutions(solutions, n_predictions)
            subset_metrics = evaluate(data, solutions_subset, verbose=False)[0]
            for i in range(1, 3):
                subset_metrics[f'vote_{i}'] = evaluate(data, select_most_voted_solutions(solutions_subset, i), verbose=False)[0]['pass_n']
            subset_metrics = {key: value for key, value in subset_metrics.items() if key not in ignore_metrics}
            metrics.append(subset_metrics)

        mean_metrics.append(average_metrics(metrics))
        all_metrics.extend(metrics)
        print_metrics(mean_metrics[-1], f'Number of predictions: {n_predictions} ')

    metric_keys = [key for key in mean_metrics[0].keys() if key not in ['unanswered', 'accuracy']]
    plt.suptitle(f'Effect of the number of predictions on the metrics. {title}')
    for plot_idx, key in enumerate(metric_keys):
        plt.subplot(1, len(metric_keys), plot_idx + 1)
        plt.plot(n_predictions_range, [metrics[key] for metrics in mean_metrics])
        plt.scatter(np.repeat(n_predictions_range, n_tries) + np.random.random(len(all_metrics))*0.1 - 0.05,
                    [metrics[key] for metrics in all_metrics],
                    alpha=0.5, c='grey')
        plt.xlabel('Number of predictions')
        plt.ylabel(key)
        plt.title(key)
        plt.xscale('log')
        plt.xticks(n_predictions_range, [str(x) for x in n_predictions_range])
        plt.minorticks_off()
        plt.grid(which='both')
    plt.tight_layout()
    plt.show()


def subsample_solutions(solutions, n_predictions):
    solutions_subset = dict()
    for task_id, task_solutions in solutions.items():
        solutions_subset[task_id] = []
        for sample_solutions in task_solutions:
            random_keys = np.random.choice(list(sample_solutions.keys()), n_predictions, replace=True)
            solutions_subset[task_id].append({idx: sample_solutions[key] for idx, key in enumerate(random_keys)})
    return solutions_subset


def study_attempt_accuracy(solutions, data):
    max_predictions = max([len(x[0]) for x in solutions.values()])
    for attempt_idx in range(1, max_predictions + 1):
        solutions_attempt = {task_id: [{f'attempt_{attempt_idx}': sample_solutions[f'attempt_{attempt_idx}']} for sample_solutions in task_solutions] for task_id, task_solutions in solutions.items()}
        metrics = evaluate(data, solutions_attempt, verbose=False)[0]
        print_metrics(metrics, f'Attempt {attempt_idx} ')



# Visualization
def plot_task(task):
    samples = task['train'] + task['test']
    for plot_idx, sample in enumerate(samples):
        plt.subplot(2, len(samples), plot_idx + 1)
        plot_grid(sample['input'])
        if plot_idx < len(task['train']):
            plt.xlabel(f'Train {plot_idx}')
        else:
            plt.xlabel(f'Test {plot_idx - len(task["train"])}')
        if 'output' in sample:
            plt.subplot(2, len(samples), plot_idx + 1 + len(samples))
            plot_grid(sample['output'])
    plt.tight_layout()


def plot_grids(grids):
    for plot_idx, grid in enumerate(grids):
        plt.subplot(1, len(grids), plot_idx + 1)
        plot_grid(grid)


def plot_grid(grid, write_numbers=False):
    grid = np.array(grid)
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True,which='both',color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1]), [])
    plt.yticks(np.arange(-0.5, grid.shape[0]), [])
    plt.xlim(-0.5, grid.shape[1]-0.5)
    if write_numbers:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                plt.text(j, i, str(grid[i, j]), ha='center', va='center')


def visualize_tasks_and_predictions(solutions, ground_truth, only_correct=False,
                                    ascending=False, max_predictions=4, figsize=(25, 4)):
    _, task_metrics = evaluate(ground_truth, solutions, verbose=False)
    for task_id in get_sorted_task_ids(task_metrics, ascending=ascending):
        if only_correct and task_metrics[task_id]['pass_n'] < 1:
            continue
        plt.figure(figsize=figsize); plot_task(ground_truth[task_id]); plt.suptitle(f'{task_id}'); plt.show()
        for test_idx, test_sample in enumerate(ground_truth[task_id]['test']):
            predicted_grids = list(solutions[task_id][test_idx].values())
            predicted_grids = [grid for grid in predicted_grids if grid]
            print_metrics(task_metrics[task_id], f'{task_id}_{test_idx}')
            plt.figure(figsize=(figsize[0], figsize[1]/2))
            plot_predictions(test_sample['output'], predicted_grids, max_grids=max_predictions)
            plt.suptitle(f'{task_id}_{test_idx}')
            plt.show()


def get_unique_matrices_and_counts_sorted(matrices):
    # TODO: move this function to a common place
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


def plot_predictions(correct_grid, predicted_grids, max_grids=5):
    unique_predicted_grids, counts = get_unique_matrices_and_counts_sorted(predicted_grids)
    if max_grids:
        if len(unique_predicted_grids) > max_grids:
            print(f'Too many unique grids: {len(unique_predicted_grids)}, leaving just {max_grids}')
        correct_grid_idx = [idx for idx, grid in enumerate(unique_predicted_grids) if grid == correct_grid]
        if correct_grid_idx and correct_grid_idx[0] >= max_grids:
            correct_grid_idx = correct_grid_idx[0]
            unique_predicted_grids = unique_predicted_grids[:max_grids - 1] + [correct_grid]
            counts = counts[:max_grids - 1] + [counts[correct_grid_idx]]
        else:
            unique_predicted_grids = unique_predicted_grids[:max_grids]
            counts = counts[:max_grids]
    plt.subplot(1, len(unique_predicted_grids) + 1, 1)
    plot_grid(correct_grid)
    plt.xlabel('Ground truth')
    for plot_idx, (grid, count) in enumerate(zip(unique_predicted_grids, counts)):
        plt.subplot(1, len(unique_predicted_grids) + 1, plot_idx + 2)
        plot_grid(grid)
        if grid == correct_grid:
            plt.xlabel(f'Correct\nCount: {count}')
        else:
            plt.xlabel(f'Count: {count}')

# The script
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    with open(args.inference_path, 'r') as f:
        solutions = json.load(f)
    if args.max_attempts is not None:
        valid_keys = set(f'attempt_{i}' for i in range(1, args.max_attempts + 1))
        for task_id, task_solutions in solutions.items():
            for sample_idx, sample_solutions in enumerate(task_solutions):
                sample_solutions = {key: sample_solutions[key] for key in sample_solutions if key in valid_keys}
                solutions[task_id][sample_idx] = sample_solutions
    ground_truth = load_arc_data_with_solutions(args.dataset_path)
    metrics = evaluate(ground_truth, solutions, verbose=False)[0]
    print_metrics(metrics)


def parse_args(args):
    epilog = """
    """
    description = """
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('inference_path', help='Path to json file with the inference results')
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference",
                        default='/mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json')
    parser.add_argument('--max_attempts', default=None, type=int)
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()