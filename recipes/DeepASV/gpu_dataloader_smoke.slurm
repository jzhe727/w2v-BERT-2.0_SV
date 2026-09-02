#!/bin/bash
#SBATCH --job-name=vox2-gpu-loader-smoke
#SBATCH --output=/home/john.zheng1/voicegeneration/vox2_gpu_loader_smoke_%j.out
#SBATCH --error=/home/john.zheng1/voicegeneration/vox2_gpu_loader_smoke_%j.err
#SBATCH --chdir=/home/john.zheng1/voicegeneration/w2v-BERT-2.0_SV/recipes/DeepASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-v100,gpu-l40

set -euo pipefail

source ~/software/miniconda3/bin/activate
conda activate w2vbert-sv-tar

SCRATCH_JOB_ID=$(cat /home/john.zheng1/voicegeneration/voxceleb2_scratch_id.txt)
INDEX_PATH="/scratch/${SCRATCH_JOB_ID}/voxceleb2-dev-wds/voxceleb2.index.sqlite3"

test -f "${INDEX_PATH}"
nvidia-smi

# Same launcher settings as the published training command (2 ranks, 16 workers each).
OMP_NUM_THREADS="16" torchrun --nnodes 1 --nproc_per_node=2 --master_port=12885 \
    tests/gpu_dataloader_smoke.py \
    --tar-index "${INDEX_PATH}" \
    --batch-size 64 \
    --num-workers 16 \
    --epochs 2 \
    --max-iters 50 \
    --result-json /home/john.zheng1/voicegeneration/vox2_gpu_loader_smoke_${SLURM_JOB_ID}.json

echo "Finished GPU dataloader smoke test on $(date)"
