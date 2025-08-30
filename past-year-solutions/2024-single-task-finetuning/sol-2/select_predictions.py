import sys
import argparse
import json
import random
from tqdm.auto import tqdm
import numpy as np

from inference import (
    clear_vllm_gpu_memory,
    generate_outputs_with_batches
)
from arc24.data_augmentation import (
    get_random_geometric_augmentation_params,
    get_random_color_map,
    apply_data_augmentation,
    set_random_seed
)
from arc24.prompting import create_prompts_from_task
from arc24.logging import log_execution_time, logging
from verify_predictions import (
    load_data,
    create_inference_artifacts,
)

logger = logging.getLogger(__name__)


@log_execution_time
def main():
    cfg = parse_args()
    dataset, unique_predictions = load_data(cfg.dataset_path, cfg.predictions_path)
    # unique_predictions = {key: unique_predictions[key] for key in list(unique_predictions.keys())[:10]}
    matches_results = create_matches_results(unique_predictions)
    tokenizer, grid_encoder, llm, sampling_params = create_inference_artifacts(cfg)
    set_random_seed(cfg.random_seed)
    total_number_of_prompts = 0
    n_rounds = get_n_rounds(unique_predictions) + 1
    for round_idx in range(n_rounds):
        prompts = create_prompts(
            matches_results, unique_predictions, dataset, grid_encoder, tokenizer,
            prompt_version=cfg.prompt_version, max_matches_per_round=cfg.max_matches_per_round)
        total_number_of_prompts += len(prompts)
        logger.info(f'Round {round_idx+1}/{n_rounds}: {len(prompts)} prompts')
        if not prompts:
            break
        outputs = generate_outputs_with_batches(llm, prompts, sampling_params, batch_size=cfg.batch_size)
        matches_results = update_matches_results(outputs, prompts, matches_results)
    logger.info(f'Total number of prompts: {total_number_of_prompts}')

    selected_predictions = select_predictions(unique_predictions, matches_results, cfg.n_top)

    with open(cfg.output_path, 'w') as f:
        json.dump(selected_predictions, f, indent=4)
    rich_output = create_rich_output(unique_predictions, matches_results)
    with open(cfg.output_path.replace('.json', '_rich_output.json'), 'w') as f:
        json.dump(rich_output, f, indent=4)

    del llm.llm_engine.model_executor
    del llm
    clear_vllm_gpu_memory()


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]
    epilog = """
    """
    description = """
Creates a ranking of the predictions using a model that verifies if a prediction is correct or not.
This works because verifying that a prediction is correct is an easier task than making the prediction itself.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--model-path', type=str, help="Path to the verifier model")
    parser.add_argument('--max-model-len', default=12000,
                        type=int, help="Maximum number of tokens in the model")
    parser.add_argument('--grid-encoder', default='GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))',
                        type=str, help="Name of the grid encoder")
    parser.add_argument('--prompt-version', default='select-output-from-examples-v0',
                        type=str, help="Prompt version")
    parser.add_argument('--dataset-path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--predictions-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--output-path', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--temperature', default=0, type=float, help="temperature for sampling, 0.0 for greedy search")
    parser.add_argument('--batch-size', default=512, type=int, help="batch size for inference")
    parser.add_argument('--max-output-tokens', default=5, type=int, help="Maximum number of tokens to generate")
    parser.add_argument('--random-seed', default=None, type=int, help="Random seed for data augmentation")
    parser.add_argument('--swap-space', default=0, type=int, help="CPU swap space size (GiB) per GPU")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    parser.add_argument('--n-top', default=2, type=int, help="Number of top predictions to select")
    parser.add_argument('--max-matches-per-round', default=32, type=int, help="Maximum number of matches per round (that will be used only on a 1v1 comparison)")
    print(args)
    return parser.parse_args()


def create_matches_results(unique_predictions):
    """
    Creates a dictionary to store the matches results, each sample has:

    - matches_results: a matrix of zeros with the shape (n_predictions, n_predictions). In the
      rows we have the victories of the predictions in the columns.
    - rounds: a list with the predictions that participated on each round
    """
    matches_results = dict()
    for task_id, task_predictions in unique_predictions.items():
        matches_results[task_id] = []
        for _, sample_predictions in enumerate(task_predictions):
            matches_results[task_id].append(dict(
                matches_results=np.zeros((len(sample_predictions), len(sample_predictions))),
                rounds=[],
            ))
    return matches_results


def get_n_rounds(unique_predictions):
    max_predictions = 0
    for task_predictions in unique_predictions.values():
        for sample_predictions in task_predictions:
            max_predictions = max(max_predictions, len(sample_predictions))
    return int(np.ceil(np.log2(max_predictions)))


def create_prompts(matches_results, predictions, dataset, grid_encoder, tokenizer,
                   prompt_version, max_matches_per_round):
    """ Creates prompt doing an all vs all comparison of the predictions """
    prompts = []
    for task_id, task_predictions in predictions.items():
        for sample_idx, sample_predictions in enumerate(task_predictions):
            sample_matches_results = matches_results[task_id][sample_idx]
            indices = select_indices_for_new_round(sample_matches_results)
            # logger.info(f'{task_id}_{sample_idx}: {indices}')
            if indices is None:
                continue
            sample_matches_results['rounds'].append(indices.copy())
            np.random.shuffle(indices)
            n_matches = get_n_matches(len(indices), max_n_matches=max_matches_per_round)
            for shift in range(n_matches//2):
                for idx1, idx2 in zip(indices, np.roll(indices, shift % (len(indices) - 1) + 1)):
                    task = dataset[task_id].copy()
                    task['test'] = [dict(input=task['test'][sample_idx]['input'],
                                         output_1=sample_predictions[idx1],
                                         output_2=sample_predictions[idx2])]
                    data_augmentation_kwargs = get_random_geometric_augmentation_params()
                    data_augmentation_kwargs['color_map'] = get_random_color_map(change_background_probability=0.1)
                    augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)
                    if random.random() < 0.5:
                        augmented_task['test_output_choices'] = [
                            augmented_task['test'][0]['output_1'],
                            augmented_task['test'][0]['output_2'],
                        ]
                        prediction_indices=[idx1, idx2]
                    else:
                        augmented_task['test_output_choices'] = [
                            augmented_task['test'][0]['output_2'],
                            augmented_task['test'][0]['output_1'],
                        ]
                        prediction_indices=[idx2, idx1]
                    prompt = create_prompts_from_task(
                        augmented_task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                        is_train_prompt=False, prompt_version=prompt_version)[0]
                    prompts.append(dict(task_id=task_id,
                                        data_augmentation_kwargs=data_augmentation_kwargs,
                                        prompt=prompt,
                                        sample_idx=sample_idx,
                                        prediction_indices=prediction_indices))
    return prompts


def get_n_matches(n_predictions, max_n_matches=32, min_n_matches=8):
    """
    For max_n_matches=64, min_n_matches=8:
    n_predictions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    n_matches = [65, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    """
    n_matches = int(np.ceil(np.log2(n_predictions)))
    n_matches = max(max_n_matches//2**(n_matches-1), min_n_matches)
    if n_matches == max_n_matches:
        n_matches += 1 # to solve ties
    return n_matches


def select_indices_for_new_round(matches_results):
    results = matches_results['matches_results']
    previous_rounds = matches_results['rounds']
    if not previous_rounds: # use all the predictions in the initial round
        if len(results) < 2:
            return None
        return np.arange(len(results))

    previous_indices = previous_rounds[-1]
    if len(previous_indices) <= 2: # no more rounds
        return None
    strength = bradley_terry(results)
    n_selected = len(previous_indices) // 2 + len(previous_indices) % 2
    selection = np.argsort(strength)[::-1][:n_selected]
    return selection


def bradley_terry(w, max_iter=1000, tol=1e-3, epsilon=1e-5):
    """
    Estimates the strength of the predictions using the Bradley-Terry model.
    As input receives a wins matrix, with the wins in the rows and the loses in the columns.
    https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """
    p = np.ones(w.shape[0])
    for iteration in range(max_iter):
        p_old = p.copy()
        for i in range(w.shape[0]):
            p[i] = max(np.sum(w[i]*p/(p + p[i] + epsilon))/(np.sum(w[:, i]/(p + p[i] + epsilon)) + epsilon), epsilon)
        normalization_factor = p.prod()**(1/len(p))
        p /= normalization_factor
        if np.linalg.norm(p - p_old) < tol:
            logger.debug(f"Converged after {iteration+1} iterations: {p.round(3)}")
            break
    return p


def update_matches_results(outputs, prompts, matches_results):
    text_outputs = [output.outputs[0].text for output in outputs]
    logger.info(f'Unique text outputs: {np.unique(text_outputs, return_counts=True)}')
    for output, prompt in zip(text_outputs, prompts):
        if output == '1':
            winner = prompt['prediction_indices'][0]
            loser = prompt['prediction_indices'][1]
        elif output == '2':
            winner = prompt['prediction_indices'][1]
            loser = prompt['prediction_indices'][0]
        else:
            logger.warning(f'Invalid output: {output}')
            continue
        matches_results[prompt['task_id']][prompt['sample_idx']]['matches_results'][winner, loser] += 1
    return matches_results


def select_predictions(unique_predictions, matches_results, n):
    selected_predictions = dict()
    for task_id, task_predictions in unique_predictions.items():
        selected_predictions[task_id] = []
        for sample_predictions, sample_matches_results in zip(task_predictions, matches_results[task_id]):
            if sample_matches_results['rounds']:
                strength = bradley_terry(sample_matches_results['matches_results'])
                ranking = np.argsort(strength)[::-1][:n]
                logger.info(f'{task_id}: {sorted(strength.round(2).tolist(), reverse=True)}')
                chosen_predictions = [sample_predictions[idx] for idx in ranking]
            else:
                logger.info(f'{task_id} had just {len(sample_predictions)} predictions')
                chosen_predictions = sample_predictions

            if len(chosen_predictions) < n:
                chosen_predictions += [[]]*(n - len(chosen_predictions))

            selected_predictions[task_id].append(
                {f'attempt_{idx}': prediction for idx, prediction in enumerate(chosen_predictions, 1)})
    return selected_predictions


def create_rich_output(unique_predictions, matches_results):
    rich_output = dict()
    for task_id, task_predictions in unique_predictions.items():
        rich_output[task_id] = []
        for sample_idx, sample_predictions in enumerate(task_predictions):
            rich_output[task_id].append(dict(
                predictions=sample_predictions,
                matches_results=matches_results[task_id][sample_idx]['matches_results'].tolist(),
                rounds=[indices.tolist() if indices is not None else None for indices in matches_results[task_id][sample_idx]['rounds']]))
    return rich_output


if __name__ == '__main__':
    main()
