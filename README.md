# [Enhancing Speaker Verification with W2V-BERT 2.0 and Knowledge Distillation-Guided Structured Pruning](https://arxiv.org/abs/2510.04213)

![Diagram](assets/framework.png)

### Preparation Stage

Download the W2V-BERT 2.0 pre-trained weights from Hugging Face and place them in the designated directory:

```
URL: https://huggingface.co/facebook/w2v-bert-2.0/blob/main/model.safetensors
Destination folder: deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/
```

Environment Setup

```
conda create -y -n asv python=3.9

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip uninstall -y transformers
pip install --no-deps -e deeplab/pretrained/audio2vector/module/transformers

conda install -c conda-forge sox
```

The local setup uses the `w2vbert-sv-tar` environment and stores the pretrained
model outside Git at
`/home/john.zheng1/voicegeneration/models/facebook/w2v-bert-2.0`. The local
W2V-BERT training YAML files are wired to that directory.

### Train Stage

#### Indexed tar training data

The training loader can read uncompressed WebDataset tar shards without extracting
their members. Build the SQLite index once from `recipes/DeepASV`:

```bash
python local/tar_index.py \
  --shard-glob '/scratch/46889734/voxceleb2-dev-wds/voxceleb2-dev-*.tar' \
  --output /scratch/46889734/voxceleb2-dev-wds/voxceleb2.index.sqlite3
```

Set `train_tar_index` in the selected training YAML to the generated index path.
`tar_max_open_shards` controls the per-worker file-descriptor cache. When
`train_tar_index` is null or omitted, the original `train_data` loader is used.
The indexed loader retains exact `RandomSampler`/`DistributedSampler` shuffling,
speed-perturbed labels, cropping, and augmentation. It supports epoch-boundary
stage transitions through model checkpoints, but full optimizer/scheduler resume
and mid-epoch dataloader state are not implemented.

PyAV decodes M4A members directly from bytes. SoX remains required for speed
perturbation:

```bash
conda install -c conda-forge sox
```

MUSAN augmentation can also run from one packed, uncompressed tar. Decompress
only the gzip layer, then set `musan_path` to the resulting `.tar`:

```bash
gzip -dk /path/to/musan.tar.gz
```

The first dataset initialization builds an atomic `musan.tar.sqlite3` sidecar.
Workers then read WAV members by indexed offsets without extracting them. Music
selection continues to use MUSAN's `ANNOTATIONS` files and excludes tracks
marked as vocal. Compressed `.tar.gz` input is rejected because it cannot
support direct positional reads.

#### VoxCeleb1 validation

Extract the VoxCeleb1 development and test archives under one `wav/` directory,
then prepare each supplied protocol from `recipes/DeepASV`:

```bash
python local/prepare_voxceleb1.py \
  --wav-root /scratch/47731887/voxceleb1/wav \
  --trial-source /home/john.zheng1/voicegeneration/veri_test2.txt \
  --output-dir /scratch/47731887/voxceleb1/protocols/vox1-o
```

Use `list_test_all2.txt` and `list_test_hard2.txt` similarly for Vox1-E and
Vox1-H. Training validation uses Vox1-O; the other manifests are retained for
final evaluation.

Run the CPU loader tests from `recipes/DeepASV`:

```bash
PYTHONPATH=../.. python -m unittest discover -s tests -v
```

#### Local two-GPU A100/H100 training

The local configs retain the original per-GPU microbatches: 64 for Stages 1-2
and 32 for Stage 3. Two GPUs give one quarter of the original eight-GPU global
batch, so all learning-rate bounds are scaled by `sqrt(1/4) = 1/2`. Gradient
accumulation is disabled, and checkpoints are written only at epoch boundaries.

Submit Stage 1:

```bash
sbatch train_2xgpu.slurm
```

Submit Stage 2 with the final Stage 1 epoch checkpoint. The job merges LoRA
weights automatically before full fine-tuning:

```bash
sbatch --export=ALL,STAGE=s2,PRETRAIN=/path/to/stage1/ckpt_0015.pth \
  train_2xgpu.slurm
```

Submit Stage 3 with the final Stage 2 checkpoint:

```bash
sbatch --export=ALL,STAGE=s3,PRETRAIN=/path/to/stage2/ckpt_0019.pth \
  train_2xgpu.slurm
```

![Diagram](assets/table1.png)

### Prune Stage

**Stage1: knowledge distillation guided structured pruning**

```
OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train_prune_s1.py \
--tag prune_ \
--is_distributed true \
--yaml conf/prune/dis_prune_s1.yaml
```

**Stage2: further distillation**

```
cd utils
python3 apply_prune_s1.py

OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train_prune_s2.py \
--tag prune_ \
--is_distributed true \
--yaml conf/prune/dis_prune_s2.yaml \
--pretrain /path/prune_stage1/prune_update.pth
```

**Stage2: further fine-tuning**

```
cd utils
python3 apply_prune_s2.py

OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train.py \
--tag prune_ft_ \
--is_distributed true \
--yaml conf/prune/s1.yaml \
--pretrain /path/prune_stage2/prune_dis.pth

OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train.py \
--tag prune_ft_ \
--is_distributed true \
--yaml conf/prune/s2.yaml \
--pretrain /path/prune_ft_stage1/best_ckpt.pth

OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train.py \
--tag prune_ft_ \
--is_distributed true \
--yaml conf/prune/s3.yaml \
--pretrain /path/prune_ft_stage2/best_ckpt.pth

```

![Diagram](assets/prune_new.png)

### Test stage

```
cd utils
python3 get_embd_w2v.py
```



### Model download

#### **Training sets: VoxCeleb2 & VoxBlink2**

**Model: LoRA_Adapter_MFA** 

**Params: 580+6.2M**

**The training YAML configuration**: [config](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/config/v1)

| Vox1-O (EER) | Vox1-E (EER) | Vox1-H (EER) | LMFT | Download Link                                                |
| ------------ | ------------ | ------------ | ---- | ------------------------------------------------------------ |
| 0.23%        | 0.38%        | 0.81%        | ×    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_base_0.23.pth) |
| 0.14%        | 0.31%        | 0.73%        | √    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth) |

#### **Training sets: VoxCeleb2**

**Model: Adapter_MFA （LoRA is not used in Stage 1）** 

**Params: 580+6.2M**

|            | Vox1-O (EER) | Vox1-E (EER) | Vox1-H (EER) | LMFT | Download Link                                                |
| ---------- | ------------ | ------------ | ------------ | ---- | ------------------------------------------------------------ |
| **Stage1** | 0.43%        | 0.65%        | 1.26%        | ×    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Adapter_MFA_voxceleb2/s1) |
| **Stage2** | 0.28%        | 0.50%        | 1.04%        | ×    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Adapter_MFA_voxceleb2/s2) |
| **Satge3** | 0.18%        | 0.37%        | 0.81%        | √    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Adapter_MFA_voxceleb2/s3) |

**Model: LoRA_Adapter_MFA** 

**Params: 580+6.2M**

|            | Vox1-O (EER) | Vox1-E (EER) | Vox1-H (EER) | LMFT | Download Link                                                |
| ---------- | ------------ | ------------ | ------------ | ---- | ------------------------------------------------------------ |
| **Stage1** | 0.31%        | 0.55%        | 1.17%        | ×    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Lora_Adapter_MFA_voxceleb2/s1) |
| **Stage2** | 0.30%        | 0.53%        | 1.15%        | ×    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Lora_Adapter_MFA_voxceleb2/s2) |
| **Stage3** | 0.23%        | 0.46%        | 1.03%        | √    | [Link](https://huggingface.co/zl389/w2v-bert-2.0_SV/tree/main/Lora_Adapter_MFA_voxceleb2/s3) |

## Citations

```
@article{li2025enhancing,
  title={Enhancing Speaker Verification with w2v-BERT 2.0 and Knowledge Distillation guided Structured Pruning},
  author={Li, Ze and Cheng, Ming and Li, Ming},
  journal={arXiv preprint arXiv:2510.04213},
  year={2025}
}
```

