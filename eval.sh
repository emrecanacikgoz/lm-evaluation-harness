#!/bin/bash
#SBATCH --job-name=deneme-llm
#SBATCH -p palamut-cuda                                            # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A proj12                                                  # Kullanici adi
#SBATCH -o %J-mistral-7b-r8-a16-e3.out           # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1                                               # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                                                       # Gorev kac node'da calisacak?
#SBATCH -n 1                                                       # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16                                         # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=3-0:0:0                                             # Sure siniri koyun.


echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
source activate lm-evaluation-harness
echo 'number of processors:'$(nproc)
nvidia-smi
# 18253, 36507, 54759

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=mistralai/Mistral-7B-v0.1,peft=/truba/home/eacikgoz/llm-deneme/checkpoints/mistral-7b-r8-a16-lr0.0001-bs16-mbs2-e3/checkpoint-54759 \
    --tasks medmcqa,pubmedqa,medqa_usmle,usmle_step1,usmle_step2,usmle_step3 \
    --device cuda \
    --max_batch_size 1