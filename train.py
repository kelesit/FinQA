from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

from .hparams import get_train_args
from .tools import get_preprocess_func

def main():

    # parse args
    model_args, data_args, finetuning_args, training_args = get_train_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        token=model_args.hf_token,
    )

    # Load base model
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 4:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                device_map='auto',
                token=model_args.hf_token,
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_use_double_quant=model_args.double_quantization
                )
            )

        elif model_args.quantization_bit == 8:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                device_map='auto',
                token=model_args.hf_token,
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map='auto',
            token=model_args.hf_token,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        bias='none',
        r=finetuning_args.lora_rank,
        target_modules=finetuning_args.lora_target,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    data = load_dataset(data_args.dataset_name)
    data = data.shuffle(seed=finetuning_args.seed)
    preprocess_func = get_preprocess_func(tokenizer, data_args)

    if data_args.val_size > 0:
        train_val = data['train'].train_test_split(test_size=data_args.val_size, seed=finetuning_args.seed)
        train_data = train_val['train'].map(preprocess_func)
        val_data = train_val['test'].map(preprocess_func)
    else:
        train_data = data['train'].map(preprocess_func)
        val_data = None

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), 
    )


    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()

