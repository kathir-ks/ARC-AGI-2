import sys
import argparse
import json


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    with open(args.sub_1, 'r') as f:
        sub_1 = json.load(f)
    with open(args.sub_2, 'r') as f:
        sub_2 = json.load(f)

    combined_sub = combine_submissions(sub_1, sub_2, args.give_preference_to_second_submission_on_second_attempt)
    with open(args.output, 'w') as f:
        json.dump(combined_sub, f)


def combine_submissions(sub_1, sub_2, give_preference_to_second_submission_on_second_attempt=True):
    # trick to give preference to sub_1, but at the same time we have all the keys from sub_2
    combined_sub = sub_2.copy()
    combined_sub.update(sub_1)
    for task_id, values in combined_sub.items():
        if task_id not in sub_2:
            continue
        for i, _ in enumerate(values):
            sub_1_attempt_1_is_empty = not combined_sub[task_id][i]['attempt_1']
            sub_1_attempt_2_is_empty = not combined_sub[task_id][i]['attempt_2']
            if sub_1_attempt_1_is_empty: # take second submission
                combined_sub[task_id][i] = sub_2[task_id][i]
            elif sub_1_attempt_2_is_empty or give_preference_to_second_submission_on_second_attempt:
                sub_2_attempt_1_is_not_empty = bool(sub_2[task_id][i]['attempt_1'])
                sub_2_attempt_2_is_not_empty = bool(sub_2[task_id][i]['attempt_2'])
                sub_2_attempt_1_is_different = sub_2[task_id][i]['attempt_1'] != combined_sub[task_id][i]['attempt_1']
                sub_2_attempt_2_is_different = sub_2[task_id][i]['attempt_2'] != combined_sub[task_id][i]['attempt_1']
                if sub_2_attempt_1_is_not_empty and sub_2_attempt_1_is_different:
                    combined_sub[task_id][i]['attempt_2'] = sub_2[task_id][i]['attempt_1']
                elif sub_2_attempt_2_is_not_empty and sub_2_attempt_2_is_different:
                    combined_sub[task_id][i]['attempt_2'] = sub_2[task_id][i]['attempt_2']
    return combined_sub


def parse_args(args):
    epilog = """
    """
    description = """
Creates a single submission by combining two submissions.

There are two modes of operation:

- The default mode will give total preference to the first submission, and will only use the second submission if the first submission is empty.
- If the flag --give_preference_to_second_submission_on_second_attempt is set, the second submission will be given preference on the second attempt.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--sub_1', help='Path to json file with first submission', required=True)
    parser.add_argument('--sub_2', help='Path to json file with second submission', required=True)
    parser.add_argument('--output', help='Path to save the combined submission', required=True)
    parser.add_argument('--give_preference_to_second_submission_on_second_attempt', action='store_true', help='If True, the second submission will be given preference on the second attempt')
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
