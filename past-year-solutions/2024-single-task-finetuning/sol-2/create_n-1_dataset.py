import sys
import argparse
import json


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    with open(args.input_dataset, 'r') as f:
        data = json.load(f)
    print(f'{args.input_dataset} has {len(data)} tasks')
    new_data = create_n_minus_1_dataset(data)
    print(f'New n-1 dataset has {len(new_data)} tasks')
    with open(args.output_dataset, 'w') as f:
        json.dump(new_data, f)


def parse_args(args):
    epilog = """
    """
    description = """
    Given an ARC dataset creates a new one using n-1 train samples
    This is used for test time fine-tuning
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('input_dataset', help='Path to json file with input dataset')
    parser.add_argument('output_dataset', help='Path to json file with output dataset')
    args = parser.parse_args(args)
    print(args)
    return args


def create_n_minus_1_dataset(data):
    new_data = dict()
    for task_id, task in data.items():
        if len(task['train']) < 2:
            continue
        new_data[task_id] = dict(
            train=task['train'][1:],
            test=task['train'][:1],
        )
        # This previous implementation was doing train-test swap, that can be done via data augmentation
        # for i, test_sample in enumerate(task['train']):
        #     new_data[f'{task_id}_{i}'] = dict(
        #         train=[sample for j, sample in enumerate(task['train']) if i != j],
        #         test=[test_sample],
        #     )
    return new_data


if __name__ == '__main__':
    main()
