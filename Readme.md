# Fine-tuning Script with Checkpoint Support

This script allows you to fine-tune a model using a specified dataset and configuration. It supports LoRA-based fine-tuning via a command-line interface.

## Parameters

| Parameter         | Type    | Required | Description                                                                 |
|------------------|---------|----------|-----------------------------------------------------------------------------|
| `--ft_data_dir`  | string  | Yes      | Path to the directory containing the fine-tuning dataset.                   |
| `--model_name`   | string  | Yes      | The base model to be fine-tuned (e.g. `qwen/qwen2.5-0.5B-instruct`).        |
| `--ft_model_name`| string  | Yes      | The name to assign to the resulting fine-tuned model.                       |
| `--num_of_samples`| int    | Yes      | The number of samples from the dataset to use for training.                 |
| `--ft_w_lora`    | flag    | No       | If specified, enables fine-tuning with LoRA.                                |

## Usage

Make sure you have your training script and the `Finetune_w_checkpoint` class available.

### Example 1: Standard Fine-Tuning

```bash
python3 train.py \
  --ft_data_dir data/training/synth_wiki_finance_v2.2 \
  --model_name qwen/qwen2.5-0.5B-instruct \
  --ft_model_name my_finetuned_model \
  --num_of_samples 1000
````

### Example 2: Fine-Tuning with LoRA

```bash
python3 train.py \
  --ft_data_dir data/training/synth_wiki_finance_v2.2 \
  --model_name qwen/qwen2.5-0.5B-instruct \
  --ft_model_name my_finetuned_model_lora \
  --num_of_samples 1000 \
  --ft_w_lora
```

## Notes

* The `--ft_w_lora` flag is optional. If omitted, standard fine-tuning will be used.
* You can adjust `--num_of_samples` based on your dataset size and compute capacity.

