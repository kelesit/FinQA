from functools import partial
from typing import Callable

from transformers import AutoTokenizer

from .hparams import DataArguments

def get_preprocess_func(tokenizer:"AutoTokenizer", data_args:"DataArguments")-> Callable:
    preprocess_func = partial(
                generate_and_tokenize_prompt,
                tokenizer=tokenizer,
                data_args=data_args,
            )
    return preprocess_func

def generate_and_tokenize_prompt(data_point, tokenizer:"AutoTokenizer", data_args:"DataArguments"):
    """This function masks out the labels for the input, so that our loss is computed only on the
    response."""
    if data_point['input']:
        user_prompt = 'Below is an instruction that describes a task, paired with an input that ' \
                      'provides further context. Write a response that appropriately completes ' \
                      'the request.\n\n'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Input:\n{data_point["input"]}\n\n'
        user_prompt += f'### Response:\n'
    else:
        user_prompt = 'Below is an instruction that describes a task. Write a response that ' \
                      'appropriately completes the request.'
        user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
        user_prompt += f'### Response:\n'

    # Count the length of prompt tokens
    len_user_prompt_tokens = len(tokenizer(user_prompt,
                                           truncation=True,
                                           max_length=data_args.cutoff_len + 1,
                                           padding='max_length')['input_ids'])
    len_user_prompt_tokens -= 1  # Minus 1 (one) for eos token

    # Tokenise the input, both prompt and output
    full_tokens = tokenizer(
        user_prompt + data_point['output'],
        truncation=True,
        max_length=data_args.cutoff_len + 1,
        padding='max_length',
    )['input_ids'][:-1]
    return {
        'input_ids': full_tokens,
        'labels': [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        'attention_mask': [1] * (len(full_tokens)),
    }
