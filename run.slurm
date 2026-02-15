#!/bin/bash
#SBATCH -J bam_eval            # 作业名
#SBATCH -p common-gpu          # GPU 分区（DKU 文档里叫 common-gpu）
#SBATCH --gres=gpu:1           # 申请 1 块 GPU
#SBATCH --cpus-per-task=4      # 给一点 CPU
#SBATCH --mem=16G              # 内存
#SBATCH -t 02:00:00            # 运行时间上限 2 小时
#SBATCH -o logs/bam_eval_%j.out  # 标准输出
#SBATCH -e logs/bam_eval_%j.err  # 标准错误

# ---- 准备环境 ----
# 确保有 anaconda 的初始化脚本路径（按你自己的安装路径改）
source /dkucc/home/zz324/miniconda3/etc/profile.d/conda.sh

# 激活你的环境
conda activate bam2

# 进入项目目录
cd /dkucc/home/zz324/partComSpoof/BAM-master

# 确保日志目录存在
mkdir -p logs

# （可选）显示一下节点和 GPU 信息
echo "Running on host: $(hostname)"
nvidia-smi || echo "nvidia-smi not found"

# ---- 正式运行测试 ----
# 在 Slurm 中你只申请了 1 块 GPU，这块卡在作业里就是可见的 0
# 如果你坚持模仿原来那条命令，可以写成 CUDA_VISIBLE_DEVICES=0
# python train.py \
#   --test_only \
#   --checkpoint ./models/model.ckpt \
#   --eval_root /dkucc/home/zz324/partComSpoof/BAM-master/data/raw/eval
# python train.py --train_root ./data/raw/train --dev_root ./data/raw/dev


python generateDummy.py --out_root ./fake_partialspoof_data --resolution 0.16 --T 20