#!/bin/bash
#SBATCH --account=xov@h100
#SBATCH --job-name synthetic_data_finetuning
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
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

export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

accelerate launch --config_file accelerate_config/fsdp2.yaml train.py \
--ft_data_dir data/finetuning_datasets/cite_think_v2.2 \
--model_name Qwen/Qwen2.5-7B-Instruct \
--ft_model_name test_launch
