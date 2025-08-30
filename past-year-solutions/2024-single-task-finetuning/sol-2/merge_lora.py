import sys
import argparse
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

from arc24.logging import logging
logger = logging.getLogger(__name__)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    merge_lora(args.base_model_path, args.lora_path, args.output_path)


def parse_args(args):
    epilog = """
    """
    description = """
Combines a base model and a lora adapter into a single model.
This is useful for faster inference, and VLLM did not work with lora on Kaggle hardware
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--base_model_path', required=True, help='Path to the folder with the base model')
    parser.add_argument('--lora_path', required=True, help='Path to the folder with the lora adapter')
    parser.add_argument('--output_path', required=True, help='Path to the folder where the merged model will be saved')
    args = parser.parse_args(args)
    print(args)
    return args


def merge_lora(base_model_path, lora_path, output_path):
    if is_lora_path(lora_path):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if 'llama' in base_model_path:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            base_model.resize_token_embeddings(len(tokenizer))


        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()
        logger.info('Saving the merged model to the output path')
        merged_model.save_pretrained(output_path)
        for filepath in glob.glob(os.path.join(lora_path, '*.json')):
            dst = os.path.join(output_path, os.path.basename(filepath))
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)
    else:
        logger.warning('The provided lora_path does not contain a lora adapter model, it is a full model')
        os.makedirs(output_path, exist_ok=True)
        for filepath in glob.glob(os.path.join(lora_path, '*')):
            dst = os.path.join(output_path, os.path.basename(filepath))
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)

    for filepath in glob.glob(os.path.join(base_model_path, '*')):
        dst = os.path.join(output_path, os.path.basename(filepath))
        if not os.path.exists(dst):
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)
    logger.info('Done!')


def is_lora_path(lora_path):
    return os.path.exists(os.path.join(lora_path, 'adapter_model.safetensors'))



if __name__ == '__main__':
    main()
