python benchmark.py \
--dataset fiqa \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--adapter saves/llama3-8b-instruct-wealth-alpaca-lora-4bit \
--max_length 512 \
--batch_size 4 \
--quantization_bit 4 \
--hf_token *** \
