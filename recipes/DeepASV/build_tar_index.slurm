#!/bin/bash
#SBATCH --job-name=vox2-tar-index
#SBATCH --output=/home/john.zheng1/voicegeneration/vox2_tar_index_%j.out
#SBATCH --error=/home/john.zheng1/voicegeneration/vox2_tar_index_%j.err
#SBATCH --chdir=/home/john.zheng1/voicegeneration/w2v-BERT-2.0_SV/recipes/DeepASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu2022,cpu2023,cpu2025

set -euo pipefail

source ~/software/miniconda3/bin/activate
conda activate w2vbert-sv-tar

SCRATCH_JOB_ID=$(cat /home/john.zheng1/voicegeneration/voxceleb2_scratch_id.txt)
DATASET_DIR="/scratch/${SCRATCH_JOB_ID}/voxceleb2-dev-wds"
INDEX_PATH="${DATASET_DIR}/voxceleb2.index.sqlite3"

test -f "${DATASET_DIR}/DOWNLOAD_COMPLETE.json"

if [ -f "${INDEX_PATH}" ]; then
    echo "Index already exists at ${INDEX_PATH}"
    exit 0
fi

echo "Building tar index on $(date)"
python local/tar_index.py \
    --shard-glob "${DATASET_DIR}/voxceleb2-dev-*.tar" \
    --output "${INDEX_PATH}"

echo "Finished tar index on $(date)"
ls -lh "${INDEX_PATH}"
