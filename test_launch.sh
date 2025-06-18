#!/bin/bash
#SBATCH --account=xov@a100
#SBATCH --job-name synthetic_data_finetuning
#SBATCH --partition=gpu_p5
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
  --ft_data_dir data/training/synth_wiki_finance_v2.2 \
  --model_name qwen/qwen2.5-0.5B-instruct \
  --ft_model_name test_launch \
  --num_of_samples 1000 \
  --ft_w_lora \
  --add_cite \
  --add_think


