#!/bin/bash
#SBATCH --job-name=s2-probe
#SBATCH --output=/home/john.zheng1/voicegeneration/w2vbert_sv_train/probe_%j.out
#SBATCH --error=/home/john.zheng1/voicegeneration/w2vbert_sv_train/probe_%j.err
#SBATCH --chdir=/home/john.zheng1/voicegeneration/w2v-BERT-2.0_SV/recipes/DeepASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-h100

set -x
echo "=== step 0: shell alive on $(hostname) $(date) ==="
source ~/software/miniconda3/bin/activate
echo "=== step 1: conda base sourced, rc=$? ==="
conda activate w2vbert-sv-tar
echo "=== step 2: env activated, rc=$? which python=$(which python) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=== step 3: torch import rc=$? ==="
PRETRAIN=/home/john.zheng1/voicegeneration/w2vbert_sv_results/checkpoints/s1_2xgpu_260916070351/ckpt_0015.pth
test -f "${PRETRAIN}"
echo "=== step 4: pretrain file test rc=$? ==="
python -c "
import sys; sys.path.insert(0, '.')
from utils import lora_merge
print('lora_merge import OK')
"
echo "=== step 5: lora_merge import rc=$? ==="
nvidia-smi -L
echo "=== step 6: done ==="
