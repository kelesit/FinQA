# 针对金融领域的咨询问答微调模型

## 项目简介

本项目是通过使用金融问答数据对Llama-3-8B-Instruct模型进行4bit量化微调，以提升模型在金融领域的咨询问答功能。

- Llama3 finetune tips
According to [usage tips for Llama3](https://huggingface.co/docs/transformers/main/en/model_doc/llama3#usage-tips) Training the model in float16 is not recommended and is known to produce nan; as such, the model should be trained in bfloat16.

## 项目运行

### Setup

申请Llama-3-8B-Instruct模型的使用权，
将huggingface token填入`loftq_4bit.yaml`与`benchmark.sh`文件中。

### 微调模型

```bash
python train.py loftq_4bit.yaml
```

### 模型评估

```bash
sh benchmark.sh
```
