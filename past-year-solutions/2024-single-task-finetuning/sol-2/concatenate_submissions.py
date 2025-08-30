import sys
import json
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    submissions = []
    for filepath in args.input_paths:
        with open(filepath, 'r') as f:
            submissions.append(json.load(f))
    concatenated_submission = concatenate_submissions(submissions)
    with open(args.output_path, 'w') as f:
        json.dump(concatenated_submission, f)


def concatenate_submissions(submissions):
    concatenated_submission = create_empty_submission(submissions)
    for submission in submissions:
        for task_id, task_predictions in submission.items():
            for sample_idx, sample_predictions in enumerate(task_predictions):
                sample_predictions = list(sample_predictions.values())
                for attempt_idx, prediction in enumerate(sample_predictions, len(concatenated_submission[task_id][sample_idx]) + 1):
                    concatenated_submission[task_id][sample_idx][f'attempt_{attempt_idx}'] = prediction
    return concatenated_submission


def create_empty_submission(submissions):
    empty_submission = dict()
    task_ids = []
    for submission in submissions:
        task_ids.extend(list(submission.keys()))
    task_ids = set(task_ids)
    for task_id in task_ids:
        max_samples = 0
        for submission in submissions:
            if task_id in submission:
                max_samples = max(max_samples, len(submission[task_id]))
        empty_submission[task_id] = [dict() for _ in range(max_samples)]
    return empty_submission


def parse_args(args):
    epilog = """
    """
    description = """
Creates a new submission by concatenating the attempts of all the input submissions
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--output-path', help='Path to the new submission that will be created')
    parser.add_argument('--input-paths', help='Path to the input submissions', nargs='+')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
