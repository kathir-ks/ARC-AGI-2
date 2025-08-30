import sys
import os
import json
import shutil
import argparse

from arc24.logging import logging

logger = logging.getLogger(__name__)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    train_conf_path = os.path.join(os.path.dirname(args.checkpoint_path), 'cfg.json')
    with open(train_conf_path, 'r') as f:
        cfg = json.load(f)
    print(cfg)
    output_folder = args.checkpoint_path.replace('arc24/models', 'arc24/evaluations')
    output_filepath = get_output_filepath(
        output_folder, args.predictions_per_task, args.dataset_path, args.temperature)
    if not os.path.exists(output_filepath):
        os.makedirs(output_folder, exist_ok=True)
        model_path = merge_lora_with_model(args.checkpoint_path, cfg['model_path'])
        inference(
            model_path, output_filepath,
            grid_encoder=cfg.get('grid_encoder', 'GridCodeBlockEncoder(MinimalGridEncoder())'),
            predictions_per_task=args.predictions_per_task,
            dataset_path=args.dataset_path,
            prompt_version=_get_prompt_version_from_conf(cfg),
            temperature=args.temperature)
        copy_train_conf(train_conf_path)
    else:
        print('Output file already exists, skipping inference')
    evaluation(output_filepath, args.dataset_path)
    # voting_output_filepath = voting(output_filepath)
    # evaluation(voting_output_filepath, args.dataset_path)
    # v2 voting
    voting_output_filepath = voting(output_filepath.replace('.json', '_task_results.json'))
    evaluation(voting_output_filepath, args.dataset_path)
    print('-'*80)
    print('Done!\n\n\n')


def merge_lora_with_model(lora_path, model_path):
    print('-'*80)
    print(f'Merging lora with model {lora_path}')
    model_path = os.path.join('/home/gbarbadillo/data', os.path.basename(model_path))
    output_path = os.path.join('/home/gbarbadillo/data/temp_model')
    if os.path.exists(output_path):
        os.system(f'rm -r {output_path}')
    ret = os.system(f'python merge_lora.py --base_model_path {model_path} --lora_path {lora_path} --output_path {output_path}')
    if ret != 0:
        raise Exception('Error merging lora with model')
    return output_path


def _get_prompt_version_from_conf(cfg):
    if len(cfg['val_dataset']) == 2:
        prompt_version = cfg['val_dataset'][1]
        logger.info(f'Using prompt version from configuration: {prompt_version}')
    else:
        prompt_version = 'output-from-examples-v0'
        logger.info(f'Using default prompt version: {prompt_version}')
    return prompt_version


def inference(model_path, output_filepath, grid_encoder, predictions_per_task,
              dataset_path, prompt_version, temperature):
    print('-'*80)
    print(f'Inference with model {model_path}')
    cmd = f'python inference.py --model_path {model_path} --output_filepath {output_filepath}'
    cmd += f' --predictions_per_task {predictions_per_task} --grid_encoder "{grid_encoder}"'
    cmd += f' --dataset_path {dataset_path} --prompt_version {prompt_version}'
    cmd += f' --temperature {temperature}'
    # cmd += ' --random_seed 7' # TODO: remove the random seed
    print(cmd)
    ret = os.system(cmd)
    # if ret != 0:
    #     raise Exception('Error running inference')
    if not os.path.exists(output_filepath):
        raise Exception('Output file not found, error running inference')


def get_output_filepath(output_folder, predictions_per_task, dataset_path, temperature):
    if dataset_path.endswith('val_rs7.json'):
        name = 'val-rs7'
    elif dataset_path.endswith('arc-agi_evaluation_challenges.json'):
        name = 'evaluation'
    elif dataset_path.endswith('arc-agi_training_challenges.json'):
        name = 'training'
    elif dataset_path.endswith('smaller_5_tasks.json'):
        name = 'smaller_5_tasks'
    elif dataset_path.endswith('all_test/training.json'):
        name = 'all-test-training'
    elif dataset_path.endswith('all_test/evaluation.json'):
        name = 'all-test-evaluation'
    else:
        raise Exception(f'Unknown dataset path: {dataset_path}')

    if temperature == 0.0:
        output_filepath = os.path.join(output_folder, f'inference_{name}_x{predictions_per_task:03d}.json')
    else:
        output_filepath = os.path.join(output_folder, f'inference_{name}_x{predictions_per_task:03d}_t{temperature:.0e}.json')
    return output_filepath


def copy_train_conf(train_conf_path):
    shutil.copy(train_conf_path, train_conf_path.replace('arc24/models', 'arc24/evaluations'))


def evaluation(filepath, dataset_path):
    print('-'*80)
    print(f'Evaluating {filepath}')
    cmd = f'python evaluation.py {filepath} --dataset_path {dataset_path}'
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception('Error running evaluation')


def voting(filepath):
    print('-'*80)
    print(f'Voting {filepath}')
    output_filepath = filepath.replace('.json', '_voting.json')
    cmd = f'python voting.py --input_filepath={filepath} --output_filepath={output_filepath}'
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception('Error running voting')
    return output_filepath


def parse_args(args):
    epilog = """
Alternative datasets for evaluation:

--dataset_path /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json
    """
    description = "Runs inference and evaluation on a given checkpoint"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('checkpoint_path', help='Path to folder with the checkpoint that we want to evaluate')
    parser.add_argument('--predictions_per_task', type=int, default=64,
                        help="Number of predictions per task, use a multiple of 8")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference and evaluation",
                        default='/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="temperature for sampling, 0.0 for greedy search")
    args = parser.parse_args(args)
    print(args)
    return args


if __name__ == '__main__':
    main()
