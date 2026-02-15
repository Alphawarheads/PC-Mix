#!/bin/bash
#SBATCH -J bam_eval
#SBATCH -p l20-gpu
#SBATCH -w dkucc-core-gpu-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/bam_eval_%j.out
#SBATCH -e logs/bam_eval_%j.err

source /dkucc/home/zz324/miniconda3/etc/profile.d/conda.sh
conda activate bam2
cd /dkucc/home/zz324/partComSpoof/BAM-master
mkdir -p logs

echo "Running on host: $(hostname)"
nvidia-smi || echo "nvidia-smi not found"
free -h

# 限制线程，避免 MKL/OMP 抢爆
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python train_multihead.py \
  --train_root ./fake_partialspoof_data/train/wav \
  --dev_root   ./fake_partialspoof_data/dev/wav \
  --eval_root  ./fake_partialspoof_data/eval/wav \
  --ref_speech_root ./fake_partialspoof_data/ref_speech \
  --ref_env_root    ./fake_partialspoof_data/ref_env \
  --label_root ./fake_partialspoof_data/labels \
  --resolution 0.16 --samplerate 16000 \
  --label_maxlength 25 --pad_mode label \
  --batch_size 1 --num_workers 16 \
  --joint_start_epoch 2\
  --max_epochs 5
# python train_multihead.py \
#   --train_root ./fake_partialspoof_data/train/wav \
#   --dev_root   ./fake_partialspoof_data/dev/wav \
#   --eval_root  ./fake_partialspoof_data/eval/wav \
#   --ref_speech_root ./fake_partialspoof_data/ref_speech \
#   --ref_env_root    ./fake_partialspoof_data/ref_env \
#   --label_root ./fake_partialspoof_data/labels \
#   --resolution 0.16 --samplerate 16000 \
#   --label_maxlength 25 --pad_mode label \
#   --batch_size 1 --num_workers 32 \
#   --test_only \
#   --checkpoint /dkucc/home/zz324/partComSpoof/BAM-master/exp/bam_multihead/train/lightning_logs/version_7/checkpoints/2-3.09853.ckpt