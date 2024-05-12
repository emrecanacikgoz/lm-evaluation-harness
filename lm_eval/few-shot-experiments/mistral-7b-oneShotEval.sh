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
#SBATCH --output=%J-Mistral-7B-Instruct-v0.2-1shot.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=bozardaanil@gmail.com

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

source activate lm_evaluation_harness
echo 'number of processors:'$(nproc)
nvidia-smi

SHOT_SIZE=1
python main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.2,use_accelerate=True" \
    --tasks arc_challenge_tr \
    --num_fewshot $SHOT_SIZE \
    --batch_size 1 \
    --output_path "/kuacc/users/hpc-aboz/lm-evaluation-harness/logs/output-${SHOT_SIZE}shot.txt" \
    --no_cache \