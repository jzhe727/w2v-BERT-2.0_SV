#!/bin/bash
#SBATCH --job-name=w2vbert-sv
#SBATCH --output=/home/john.zheng1/voicegeneration/w2vbert_sv_train/_%j.out
#SBATCH --error=/home/john.zheng1/voicegeneration/w2vbert_sv_train/_%j.err
#SBATCH --chdir=/home/john.zheng1/voicegeneration/w2v-BERT-2.0_SV/recipes/DeepASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40
# Note: s2/s3 full finetuning peaks ~49GB (measured ~42GB activations on L40S in s1
# + ~6.8GB grads/AdamW state for the 580M-param backbone) — L40S (48GB) may OOM there.
#SBATCH --exclude=egh2,mgh5

set -euo pipefail

source ~/software/miniconda3/bin/activate
conda activate w2vbert-sv-tar

STAGE=${STAGE:-s1}
case "${STAGE}" in
    s1)
        if [[ -n "${PRETRAIN:-}" ]]; then
            test -f "${PRETRAIN}"
            PRETRAIN_ARGS=(--pretrain "${PRETRAIN}")
        else
            PRETRAIN_ARGS=()
        fi
        ;;
    s2)
        : "${PRETRAIN:?Set PRETRAIN to the preceding stage checkpoint}"
        test -f "${PRETRAIN}"
        MERGED_PRETRAIN=${MERGED_PRETRAIN:-"${PRETRAIN%.pth}_lora_merged.pth"}
        if [[ ! -f "${MERGED_PRETRAIN}" ]]; then
            python utils/lora_merge.py \
                --checkpoint "${PRETRAIN}" \
                --output "${MERGED_PRETRAIN}"
        fi
        PRETRAIN_ARGS=(--pretrain "${MERGED_PRETRAIN}")
        ;;
    s3)
        : "${PRETRAIN:?Set PRETRAIN to the Stage 2 checkpoint}"
        test -f "${PRETRAIN}"
        PRETRAIN_ARGS=(--pretrain "${PRETRAIN}")
        ;;
    *)
        echo "STAGE must be s1, s2, or s3" >&2
        exit 2
        ;;
esac

CONFIG="conf/w2v-bert/${STAGE}.yaml"
test -f /scratch/48054831/voxceleb2-dev-wds/voxceleb2.index.sqlite3
test -f /scratch/48054834/voxceleb1/protocols/vox1-o/wav.scp
test -f /scratch/48054834/voxceleb1/protocols/vox1-o/trials
test -f /home/john.zheng1/voicegeneration/models/facebook/w2v-bert-2.0/model.safetensors
test -f /home/john.zheng1/voicegeneration/musan.tar
test -d /home/john.zheng1/voicegeneration/RIRS_NOISES

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    test -f "${RESUME}"
    RESUME_ARGS=(--resume "${RESUME}")
fi

nvidia-smi
MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
OMP_NUM_THREADS=1 torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --master_port "${MASTER_PORT}" \
    train.py \
    --tag "${STAGE}_2xgpu_" \
    --is_distributed true \
    --yaml "${CONFIG}" \
    "${PRETRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}"