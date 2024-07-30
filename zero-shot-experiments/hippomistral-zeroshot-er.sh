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
#SBATCH --output=%J-HippoMistral-0shot-er.log
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

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
module load gcc/11.2.0
module load cuda/11.8.0
module load cudnn/8.2.0/cuda-11.X
echo "======================="

SHOT_SIZE=0
python main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=emrecanacikgoz/hippomistral,use_accelerate=True" \
    --tasks medmcqa_er, medqa_usmle_er, pubmedqa_er, usmle_step1_er, usmle_step2_er, usmle_step3_er \
    --num_fewshot $SHOT_SIZE \
    --batch_size 1 \
    --output_path "/kuacc/users/hpc-aboz/lm-evaluation-harness/logs/output-${SHOT_SIZE}shot.txt" \
    --no_cache \