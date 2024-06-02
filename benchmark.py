import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel

from test_fiqa import test_fiqa

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=args.hf_token,
    )

    if args.quantization_bit is not None:
        if args.quantization_bit == 4:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                device_map='auto',
                token=args.hf_token,
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=args.compute_dtype,
                    bnb_4bit_quant_type=args.quantization_type,
                )
            )

        elif args.quantization_bit == 8:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                device_map='auto',
                token=args.hf_token,
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map='auto',
            token=args.hf_token,
        )

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.eval()

    with torch.no_grad():
        test_fiqa(args, model, tokenizer)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name")
    parser.add_argument("--model", required=True, type=str, help="Base model name or path")
    parser.add_argument("--adapter", required=True, type=str, help="adapter path")
    parser.add_argument("--max_length", default=512, type=int, help="Max length")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--quantization_bit", default=4, type=int, help="Quantization bit")
    parser.add_argument("--hf_token", default=None, type=str, help="HF token")

    args = parser.parse_args()
    print(args.model)
    print(args.adapter)
    print(args.dataset)

    main(args)