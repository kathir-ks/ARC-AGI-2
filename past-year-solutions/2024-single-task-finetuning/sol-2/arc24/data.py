import os
import json
import logging
import random
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_arc_data_with_solutions(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    solutions_filepath = filepath.replace('challenges.json', 'solutions.json')
    if filepath != solutions_filepath and os.path.exists(solutions_filepath):
        with open(solutions_filepath, 'r') as f:
            solutions = json.load(f)
        for sample_id, task in data.items():
            for idx, sample in enumerate(task['test']):
                sample['output'] = solutions[sample_id][idx]
    verify_that_all_samples_have_output(data)
    return data


def verify_that_all_samples_have_output(data):
    for task in data.values():
        if isinstance(task, dict):
            verify_that_task_has_outputs(task)
        elif isinstance(task, list):
            for subtask in task:
                verify_that_task_has_outputs(subtask)


def verify_that_task_has_outputs(task):
    for partition, samples in task.items():
        if partition not in ['train', 'test']:
            continue
        for sample in samples:
            if 'output' not in sample:
                raise ValueError('Not all samples have output')


class BarcDataset():
    """
    Class to load the BARC datasets from a file and sample tasks from it.

    https://huggingface.co/collections/barc0/synthetic-arc-dataset-6725aa6031376d3bacc34f76
    """
    def __init__(self, filepath, n_tasks=1e5, max_samples_per_task=10, min_samples_per_task=6):
        logger.info(f'Loading BARC dataset from: {filepath}')
        with open(filepath, 'r') as f:
            self.tasks = [json.loads(line)['examples'][:max_samples_per_task] for line in tqdm(f, total=n_tasks, desc='Loading BARC dataset')]
        self.tasks = [task for task in self.tasks if len(task) >= min_samples_per_task]
        logger.info(f'Loaded {len(self.tasks)} tasks')

    def sample(self):
        n_samples = random.randint(3, 5) # not exactly the ARC distribution, but close enough
        task = random.choice(self.tasks)
        examples = random.sample(task, n_samples)
        formatted_task = dict(
            train=[dict(input=examples[i][0], output=examples[i][1]) for i in range(n_samples - 1)],
            test=[dict(input=examples[-1][0], output=examples[-1][1])],
        )
        return formatted_task
