import sys
import os
from typing import Literal, Optional, Dict, Any, Tuple

import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser


@dataclass
class FintuningArguments:
    stage: Literal["sft", "dpo", "orpo"] = field(
        default="sft", metadata={"help": "Stage of finetuning"}
    )

    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "Rank of LoRA"}
    )

    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "Alpha of LoRA"}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout of LoRA"}
    )

    lora_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of target modules to apply LoRA.\
            Use commas to separate multiple modules. \
            Use "all" to specify all the linear modules. \
            LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
            BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
            Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
            Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
            InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
            Others choices: the same as LLaMA."""
        }
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
        
        self.lora_target = split_arg(self.lora_target)


@dataclass
class ModelArgumets:
    model_name_or_path: str = field(
        default='meta-llama/Meta-Llama-3-8B',
        metadata={"help": "Model name or path"}
    )

    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token"}
    )

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "Quantization bit"}
    )

    double_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Double quantization"}
    )

    quantization_type: Literal['nf4', 'fp4'] = field(
        default='nf4',
        metadata={"help": "Quantization type"}
    )

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Export directory"}
    )
    def __post_init__(self):
        self.compute_dtype = None


@dataclass
class DataArguments:
    data_dir: str = field(
        default='data',
        metadata={"help": "Data directory"}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name"}
    )

    cutoff_len: int = field(
        default=1024,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )

    block_size: Optional[int] = field(
        default=128,
        metadata={"help": "Block size"}
    )

    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite cache"}
    )

    val_size: float = field(
        default=0.1,
        metadata={"help": "Validation size"}

    )


def _parse_train_args(args: Optional[Dict[str, Any]]=None):
    parser = HfArgumentParser((ModelArgumets, DataArguments, FintuningArguments, TrainingArguments))
    if args is not None:
        return parser.parse_dict(args)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    
    return parser.parse_args_into_dataclasses()



def get_train_args(args: Optional[Dict[str, Any]]=None) -> Tuple[ModelArgumets, DataArguments, FintuningArguments, TrainingArguments]:
    model_args, data_args, finetuning_args, training_args = _parse_train_args(args)

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    return model_args, data_args, finetuning_args, training_args