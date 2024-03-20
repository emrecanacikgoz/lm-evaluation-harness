#!/bin/bash
#SBATCH --job-name=eacikgoz17
#SBATCH --partition=long
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a40
#SBATCH --mem=64G
#SBATCH --time=7-0:0:0
#SBATCH --output=%J-gpt2-xl-0shot.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=eacikgoz17@ku.edu.tr

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

source activate lm_evaluation_harness
echo 'number of processors:'$(nproc)
nvidia-smi

SHOT_SIZE=0
python main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=openai-community/gpt2-xl,use_accelerate=True" \
    --tasks arc_challenge_tr \
    --num_fewshot $SHOT_SIZE \
    --batch_size 1 \
    --output_path "/kuacc/users/eacikgoz17/el-turco/lm-evaluation-harness/logs/open-ai-gpt2-xl/output-${SHOT_SIZE}shot.txt" \
    --no_cache \