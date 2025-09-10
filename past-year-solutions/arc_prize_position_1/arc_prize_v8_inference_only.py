# pip install bitsandbytes -U
# pip install unsloth transformers -U

# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

arc_prize_2024_path = kagglehub.competition_download('arc-prize-2024')
dfranzen_unsloth_2024_9_post4_path = kagglehub.dataset_download('dfranzen/unsloth-2024-9-post4')
dfranzen_wb55l_nemomini_fulleval_transformers_default_1_path = kagglehub.model_download('dfranzen/wb55l_nemomini_fulleval/Transformers/default/1')

print('Data source import complete.')

# -*- coding: utf-8 -*-
"""
Full ARC Prize Inference Pipeline in a Single File
"""

import json
import os
import bz2
import pickle
import numpy as np
from tqdm import tqdm
import torch

# It's assumed 'unsloth' and 'transformers' are installed in the environment
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# ==============================================================================
# ## 헬퍼 함수 및 클래스 (Helper Functions & Classes)
# ==============================================================================

# --- arc_loader.py에서 가져옴 ---

def cut_at_token(output, token_id):
    """
    Cuts a token array at the first occurrence of a specific token ID,
    handling both PyTorch tensors and NumPy arrays correctly.
    """
    # First, find the positions of the token
    eos_positions = (output == token_id).nonzero()

    # Case 1: Handle PyTorch Tensors
    if isinstance(output, torch.Tensor):
        if eos_positions.numel() > 0:
            # For tensors, .nonzero() gives a tensor of indices. Get the first one.
            return output[:eos_positions[0].item()]

    # Case 2: Handle NumPy Arrays (or objects that behave like them)
    # np.nonzero() returns a tuple of arrays, so we check the first element.
    elif isinstance(eos_positions, tuple) and len(eos_positions) > 0 and len(eos_positions[0]) > 0:
        return output[:eos_positions[0][0]]

    # If no EOS token is found, or if the type is unexpected, return the original array
    return output

def permute_mod(a, descriptor, invert=False):
    """Permutes the colors in a grid based on a descriptor string."""
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    a = np.asarray(a)
    if invert:
        permutation = np.argsort(permutation)
    return np.asarray(permutation)[a]

class ArcDataset:
    """A simplified version of ArcDataset for loading and managing ARC tasks for inference."""
    def __init__(self, queries, replies={}, keys=None):
        self.queries = queries
        self.replies = replies
        self.keys = sorted(queries.keys()) if keys is None else keys

    @classmethod
    def from_file(cls, file_path):
        """Loads tasks from a JSON file."""
        with open(file_path, 'r') as f:
            queries = json.load(f)
        return cls(queries=queries)

    def get(self, key, formatter):
        """Formats a single task into a text prompt for the model."""
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'], i=len(self.queries[key]['train']))
        # 'last_is_challenge' creates the final prompt ending with the test input
        text = formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, input=train + query, text=text)

    def split_multi_replies(self):
        """Splits tasks with multiple test cases into individual tasks."""
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )
    # NOTE: For a fully functional pipeline, methods like 'mod', 'augment', 'cut_to_len'
    # from your original `arc_loader.py` would be needed here to replicate the full
    # data preparation process.

class ArcFormatter:
    """Formats ARC task data into a string prompt and decodes model output back to a grid."""
    def __init__(self, tokenizer, inp_prefix='I', out_prefix='O', arr_sep='\n', **kwargs):
        self.tokenizer = tokenizer
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.arr_sep = arr_sep
        self.qry_prefix = inp_prefix
        self.rpl_prefix = out_prefix
        self.dec_sep = arr_sep
        self.pretext = kwargs.get('pretext', '')
        self.exa_end = kwargs.get('exa_end', '')
        self.exa_sep = kwargs.get('exa_sep', '')

    def fmt_array(self, array):
        """Formats a 2D numpy array into a string."""
        return self.arr_sep.join("".join(map(str, row)) for row in array)

    def fmt_train(self, train, last_is_challenge=False):
        """Formats the training examples of a task."""
        ex_strings = [
            f"{self.inp_prefix}{self.fmt_array(x['input'])}{self.out_prefix}{self.fmt_array(x['output'])}"
            for x in train
        ]
        if last_is_challenge and train:
            # For the final prompt, use all but the last example for context,
            # and format the last one as the challenge.
            context = (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex_strings[:-1])
            challenge_input = f"{self.inp_prefix}{self.fmt_array(train[-1]['input'])}{self.out_prefix}"
            # Add separator only if there's context
            separator = self.exa_end + self.tokenizer.eos_token + self.exa_sep if context else ""
            return self.pretext + context + separator + challenge_input
        return self.pretext + (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex_strings)

    def fmt_query(self, query, i):
        """Formats the test input part of a prompt."""
        return ''.join(f"{self.qry_prefix}{self.fmt_array(x['input'])}{self.rpl_prefix}" for x in query[:1])

    def de_tokenize(self, tokens, scores=None):
        """Converts model output tokens back to text and calculates the score."""
        tokens_cut = cut_at_token(tokens, self.tokenizer.eos_token_id)
        de_tokenized = self.tokenizer.decode(tokens_cut, skip_special_tokens=True)
        score_val = None
        if scores is not None and len(tokens_cut) > 0:
             # Ensure scores and tokens_cut are tensors for indexing
             tokens_with_eos = torch.tensor(tokens[:len(tokens_cut) + 1], dtype=torch.long)
             log_probs = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1)
             # Slice log_probs to match the length of generated tokens
             log_probs_sliced = log_probs[:len(tokens_with_eos)]
             # Get the log probability of the generated token at each step
             score_val = log_probs_sliced[torch.arange(len(tokens_with_eos)), tokens_with_eos].sum().item()
        return len(tokens), score_val, de_tokenized, scores

    def decode_to_array(self, text, score=None):
        """Parses a string representation back into a numpy grid."""
        try:
            rows = [list(map(int, line)) for line in text.split(self.dec_sep) if line]
            if not rows or len(set(len(r) for r in rows)) != 1:
                return {} # Invalid or jagged grid
            decoded = np.array(rows, dtype=int)
            # Check for valid ARC grid dimensions
            if 0 < decoded.shape[0] <= 30 and 0 < decoded.shape[1] <= 30:
                return {'output': decoded, 'score_val': score if score is not None else 0}
        except (ValueError, TypeError):
            pass
        return {}


# --- selection.py에서 가져됨 ---

def hashable(guess):
    """Converts a numpy array to a hashable tuple so it can be used as a dictionary key."""
    return tuple(map(tuple, guess))

def score_all_probsum(guesses):
    """
    A selection algorithm that ranks guesses by summing the probabilities of identical solutions.
    This is powerful when data augmentation is used, as a consistently generated grid gets a higher score.
    """
    scores = {}
    for guess_dict in guesses.values():
        if 'output' not in guess_dict or 'score_val' not in guess_dict:
            continue
        h = hashable(guess_dict['output'])
        if h not in scores:
            scores[h] = [0, guess_dict['output']] # [total_prob, grid_array]
        # score_val is a log probability, so we use np.exp to convert it back to a probability
        scores[h][0] += np.exp(guess_dict['score_val'])

    # Sort by the summed probability in descending order
    sorted_scores = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    # Return only the ranked list of numpy arrays
    return [output_array for score, output_array in sorted_scores]


# --- model_runner.py에서 가져옴 ---

class Decoder:
    """
    Processes and manages decoded outputs from the ARC model to select the best final answers.
    """
    def __init__(self, formatter, n_guesses=3):
        self.formatter = formatter
        self.n_guesses = n_guesses
        self.decoded_results = {} # Main storage: {base_key: {full_key: guess_dict}}

    def process(self, key, de_tokenized_outputs):
        """Processes a list of de-tokenized outputs for a given task key."""
        base_key = key.split('.')[0]
        if base_key not in self.decoded_results:
            self.decoded_results[base_key] = {}

        for i, de_tokenized in enumerate(de_tokenized_outputs):
            _output_len, score_val, text, _scores_array = de_tokenized
            # The formatter attempts to parse the raw text into one or more grids
            parsed_solutions = self.formatter.decode_to_array(text, score=score_val)

            if not isinstance(parsed_solutions, list):
                parsed_solutions = [parsed_solutions]

            for solution in parsed_solutions:
                if 'output' in solution and isinstance(solution['output'], np.ndarray):
                    full_key = f"{key}.out{i}"
                    self.decoded_results[base_key][full_key] = solution

    def run_selection_algo(self, selection_algorithm):
        """Applies a selection algorithm to the collected results."""
        final_guesses = {}
        for base_key, results in self.decoded_results.items():
            if results:
                final_guesses[base_key] = selection_algorithm(results)
        return final_guesses


def generate_safely(model, tokenizer, prompt_text, generation_params):
    """
    Generates text by first checking if the input fits. If the input is too long,
    it skips generation. Otherwise, it dynamically adjusts max_new_tokens.
    """
    # 1. Get the model's maximum sequence length
    model_max_length = model.config.max_position_embeddings

    # 2. Tokenize the input prompt
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[-1]

    # 3. NEW: Check if the input itself is too long. If so, skip generation.
    if input_length >= model_max_length:
        print(f"Warning: Input length ({input_length}) is >= model's max length ({model_max_length}). Skipping generation.")
        return None

    # 4. Calculate the available space for new tokens
    available_space = model_max_length - input_length

    # Create a copy of the generation parameters to modify
    safe_params = generation_params.copy()
    original_max_new = safe_params.get("max_new_tokens", 1024)

    # 5. Adjust max_new_tokens to fit the remaining space
    if original_max_new > available_space:
        safe_params["max_new_tokens"] = available_space

    # 6. Generate text with the safe parameters
    return model.generate(**inputs, **safe_params)

# ==============================================================================
# ## 🚀 주 추론 파이프라인 (Main Inference Pipeline)
# ==============================================================================

def main():
    # --- 1. 구성 (Configuration) ---
    print("Step 1: Configuring pipeline...")
    # Paths
    # IMPORTANT: Update these paths to match your environment
    # /root/.cache/kagglehub/competitions/arc-prize-2024
    ARC_CHALLENGE_FILE = '/root/.cache/kagglehub/competitions/arc-prize-2024/arc-agi_test_challenges.json'
    FINETUNED_MODEL_PATH = '/kaggle/input/wb55l_nemomini_fulleval/transformers/default/1'
    SUBMISSION_FILE = 'submission.json'

    # Inference Parameters
    MAX_SEQ_LENGTH = 8192
    N_GUESSES = 3 # Number of solutions to generate per task
    INFERENCE_PARAMS = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    # --- 2. 모델 및 토크나이저 로드 (Load Model and Tokenizer) ---
    print("Step 2: Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        # Use local_files_only=True if loading from a local directory
        # to prevent accidental downloads.
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    print(model)
    # --- 3. 데이터셋 및 포맷터 준비 (Prepare Dataset and Formatter) ---
    print("Step 3: Preparing dataset and formatter...")
    formatter = ArcFormatter(tokenizer=tokenizer)
    arc_test_set = ArcDataset.from_file(ARC_CHALLENGE_FILE)
    # This step is crucial: it breaks down tasks with multiple test inputs
    # into individual items that the model can process one by one.
    dataset_for_inference = arc_test_set.split_multi_replies()
    # TODO: Add your data augmentation logic here if needed.
    # e.g., dataset_for_inference = dataset_for_inference.augment(...)

    decoder = Decoder(formatter, n_guesses=N_GUESSES)

    # --- 4. 추론 실행 (Run Inference) ---
    print(f"Step 4: Starting inference on {len(dataset_for_inference.keys)} tasks...")
    Temp_break_no = 0
    with torch.no_grad():
        for key in tqdm(dataset_for_inference.keys, desc="Running Inference"):
            Temp_break_no += 1
            if Temp_break_no > 25:
              break
            prompt_data = dataset_for_inference.get(key, formatter)
            inputs = tokenizer([prompt_data['input']], return_tensors="pt").to(model.device)

            # The INFERENCE_PARAMS dictionary is passed to the helper function
            outputs = generate_safely(
                model,
                tokenizer,
                prompt_text=prompt_data['input'],
                generation_params={
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    **INFERENCE_PARAMS,
                }
            )

            # This check will now catch any prompts that were too long and were skipped.
            if outputs is None:
                continue # Move to the next item in the loop

            # Process and decode each generated sequence
            de_tokenized_outputs = []
            sequences = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
            scores = torch.stack(outputs.scores, dim=1) if hasattr(outputs, 'scores') else None

            for i in range(sequences.shape[0]):
                sequence_tokens = sequences[i].cpu().numpy()
                sequence_scores = scores[i].cpu().numpy() if scores is not None else None
                de_tokenized = formatter.de_tokenize(sequence_tokens, sequence_scores)
                de_tokenized_outputs.append(de_tokenized)

            decoder.process(key, de_tokenized_outputs)

    # --- 5. 최상의 추측 선택 및 제출 파일 생성 (Select Best Guesses & Create Submission) ---
    print("Step 5: Generating submission file...")
    final_results = decoder.run_selection_algo(score_all_probsum)

    submission = {}
    for k in arc_test_set.keys:
        num_test_cases = len(arc_test_set.queries[k]['test'])
        submission[k] = [
            {f"attempt_{i+1}": [[0]] for i in range(N_GUESSES)}
            for _ in range(num_test_cases)
        ]

    for key, guesses in final_results.items():
        base_id, task_idx_str = key.split('_')
        task_idx = int(task_idx_str)

        valid_guesses = [g for g in guesses if isinstance(g, np.ndarray)]
        for i, guess in enumerate(valid_guesses[:N_GUESSES]):
            submission[base_id][task_idx][f"attempt_{i+1}"] = guess.tolist()

    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)

    print(f"✅ Submission file successfully created at: {SUBMISSION_FILE}")

main()