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
#SBATCH --partition=gpu-a100,gpu-h100
# Note: s2/s3 full finetuning peaks ~49GB (measured ~42GB activations on L40S in s1
# + ~6.8GB grads/AdamW state for the 580M-param backbone) — L40S (48GB) may OOM there.
#SBATCH --exclude=egh2,mgh5

# NOTE: intentionally no `set -euo pipefail`. That combo has killed jobs on
# harmless nonzero exits (conda activation, env quirks). Critical preconditions
# below are guarded explicitly with `|| die`, and torchrun's exit code
# propagates as the job's exit status on its own.

die() { echo "FATAL [$(hostname) $(date +%H:%M:%S)]: $*" >&2; exit 1; }

source ~/software/miniconda3/bin/activate || die "conda base activation failed"
conda activate w2vbert-sv-tar || die "conda env w2vbert-sv-tar activation failed"
echo "[startup] node=$(hostname) stage=${STAGE:-s1} job=${SLURM_JOB_ID} $(date)"

STAGE=${STAGE:-s1}
# A resumed job (RESUME=<.../ckpt_000N.pth>) restores everything from that
# checkpoint, so PRETRAIN (and the s2 LoRA merge) is skipped.
if [[ -n "${RESUME:-}" ]]; then
    PRETRAIN_ARGS=()
else
case "${STAGE}" in
    s1)
        if [[ -n "${PRETRAIN:-}" ]]; then
            test -f "${PRETRAIN}" || die "PRETRAIN checkpoint not found: ${PRETRAIN}"
            PRETRAIN_ARGS=(--pretrain "${PRETRAIN}")
        else
            PRETRAIN_ARGS=()
        fi
        ;;
    s2)
        [[ -n "${PRETRAIN:-}" ]] || die "Set PRETRAIN to the preceding stage checkpoint"
        test -f "${PRETRAIN}" || die "PRETRAIN checkpoint not found: ${PRETRAIN}"
        MERGED_PRETRAIN=${MERGED_PRETRAIN:-"${PRETRAIN%.pth}_lora_merged.pth"}
        if [[ ! -f "${MERGED_PRETRAIN}" ]]; then
            echo "[merge] building ${MERGED_PRETRAIN}"
            python utils/lora_merge.py \
                --checkpoint "${PRETRAIN}" \
                --output "${MERGED_PRETRAIN}" || die "LoRA merge failed"
        fi
        PRETRAIN_ARGS=(--pretrain "${MERGED_PRETRAIN}")
        ;;
    s3)
        [[ -n "${PRETRAIN:-}" ]] || die "Set PRETRAIN to the Stage 2 checkpoint"
        test -f "${PRETRAIN}" || die "PRETRAIN checkpoint not found: ${PRETRAIN}"
        PRETRAIN_ARGS=(--pretrain "${PRETRAIN}")
        ;;
    *)
        echo "STAGE must be s1, s2, or s3" >&2
        exit 2
        ;;
esac
fi

CONFIG="conf/w2v-bert/${STAGE}.yaml"
test -f "${CONFIG}" || die "config not found: ${CONFIG}"
echo "[startup] config=${CONFIG} pretrain=${PRETRAIN:-none} resume=${RESUME:-none}"
test -f /scratch/49572342/voxceleb2-dev-wds/voxceleb2.index.sqlite3 \
    || die "missing vox2 index /scratch/49572342/voxceleb2-dev-wds/voxceleb2.index.sqlite3 — scratch staging may have been purged"
test -f /scratch/49572343/voxceleb1/protocols/vox1-o/wav.scp \
    || die "missing vox1 wav.scp — scratch staging may have been purged"
test -f /scratch/49572343/voxceleb1/protocols/vox1-o/trials \
    || die "missing vox1 trials — scratch staging may have been purged"
test -f /home/john.zheng1/voicegeneration/models/facebook/w2v-bert-2.0/model.safetensors \
    || die "missing w2v-bert model.safetensors"
test -f /home/john.zheng1/voicegeneration/musan.tar || die "missing musan.tar"
test -d /home/john.zheng1/voicegeneration/RIRS_NOISES || die "missing RIRS_NOISES"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    test -f "${RESUME}" || die "RESUME checkpoint not found: ${RESUME}"
    RESUME_ARGS=(--resume "${RESUME}")
fi

nvidia-smi || echo "[warn] nvidia-smi returned nonzero (continuing)"
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