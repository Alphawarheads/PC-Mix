#!/bin/bash
#SBATCH --job-name=MultinodeBam
#SBATCH --nodes=4
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=32
#SBATCH --partition=kshdnormal
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/dkucc/home/zz324/partComSpoof/BAM-master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
set -eo pipefail
mkdir -p logs

# module purge
 
conda activate bam2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
# 如果 IB 不稳请先打开：
export NCCL_IB_DISABLE=1



# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip   
NODE_RANK=$SLURM_NODEID
echo NODE_RANK: $NODE_RANK
srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  --rdzv_id=$RANDOM \
  multinode_train --train_root ./data/raw/train --dev_root ./data/raw/dev \
  --num_workers 32
  

