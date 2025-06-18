#!/bin/bash
#SBATCH --account=xov@h100
#SBATCH --job-name synthetic_data_finetuning
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=syed.bukhari@linguacustodia.com
#SBATCH --error=slurm/logs/job-%A_%a_error.out
#SBATCH --output=slurm/logs/job-%A_%a_log.out


# Load necessary modules and activate conda environment
module purge
module load arch/h100
module load python
module load cuda
module load cudnn
module load nccl

conda activate ft-synth

python3 train.py \
  --ft_data_dir data/finetuning_datasets/cite_think_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_lora_cite_think_data-v2.2 \
  --ft_w_lora 

python3 train.py \
  --ft_data_dir data/finetuning_datasets/cite_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_lora_cite_data-v2.2 \
  --ft_w_lora 

python3 train.py \
  --ft_data_dir data/finetuning_datasets/cite_think_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_fft_cite_think_data-v2.2 


python3 train.py \
  --ft_data_dir data/finetuning_datasets/cite_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_fft_cite_data-v2.2 


python3 train.py \
  --ft_data_dir data/finetuning_datasets/think_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_fft_think_data-v2.2


python3 train.py \
  --ft_data_dir data/finetuning_datasets/think_v2.2 \
  --model_name qwen/qwen2.5-7B-instruct \
  --ft_model_name qwen2.5-7B_lora_think_data-v2.2 \
  --ft_w_lora 
