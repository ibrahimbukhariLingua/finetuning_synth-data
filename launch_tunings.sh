#!/bin/bash
#SBATCH --account=xov@h100
#SBATCH --job-name ft-synth
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=syed.bukhari@linguacustodia.com
#SBATCH --error=slurm/logs/job-%A_%a_error.out
#SBATCH --output=slurm/logs/job-%A_%a_log.out
#SBATCH --array=0-5%6

ALL_DATASETS=(
    "data/finetuning_datasets/cite_think_v2.2"
    "data/finetuning_datasets/cite_v2.2"
    "data/finetuning_datasets/cite_think_v2.2"
    "data/finetuning_datasets/cite_v2.2"
    "data/finetuning_datasets/think_v2.2"
    "data/finetuning_datasets/think_v2.2"
)

ALL_FINETUNE_NAMES=(
    "qwen2.5-7b-peft-cite-think-data-v2.2"
    "qwen2.5-7b-peft-cite-data-v2.2"
    "qwen2.5-7b-fullFT-cite-think-data-v2.2"
    "qwen2.5-7b-fullFT-cite-data-v2.2"
    "qwen2.5-7b-peft-think-data-v2.2"
    "qwen2.5-7b-fullFT-think-data-v2.2"
)

ALL_PEFT_CHECK=(
  "True"
  "True"
  "False"
  "False"
  "True"
  "False"
)

# Load necessary modules and activate conda environment
module purge
module load arch/h100
module load python
module load cuda
module load cudnn
module load nccl

conda activate ft-synth


DATASET=${ALL_DATASETS[$SLURM_ARRAY_TASK_ID]}
FINETUNE=${ALL_FINETUNE_NAMES[$SLURM_ARRAY_TASK_ID]}
PEFTCHECK=${ALL_PEFT_CHECK[$SLURM_ARRAY_TASK_ID]}


export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"


python3 train.py \
  --ft_data_dir ${DATASET} \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --ft_model_name ${FINETUNE} \
  --ft_w_lora ${PEFTCHECK}
