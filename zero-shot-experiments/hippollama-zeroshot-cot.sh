#!/bin/bash
#SBATCH --job-name=ardaboz
#SBATCH --partition=long
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a40
#SBATCH --mem=64G
#SBATCH --time=7-0:0:0
#SBATCH --output=%J-HippoLlama-0shot-cot.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=bozardaanil@gmail.com

SHOT_SIZE=0
python main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=emrecanacikgoz/hippollama,use_accelerate=True" \
    --tasks medmcqa_cot, medqa_usmle_cot, pubmedqa_cot, usmle_step1_cot, usmle_step2_cot, usmle_step3_cot \
    --num_fewshot $SHOT_SIZE \
    --batch_size 1 \
    --device cuda:0 \
    --output_path "/kuacc/users/hpc-aboz/lm-evaluation-harness/logs/output-${SHOT_SIZE}shot.txt" \
    --no_cache \