# Copilot instructions — w2v-BERT-2.0 speaker verification

## Environment

- All Python runs in conda env `w2vbert-sv-tar`:
  `source ~/software/miniconda3/bin/activate && conda activate w2vbert-sv-tar`
- The login box is CPU-only. Never run training or GPU code here; GPU work goes through
  slurm (`sbatch recipes/DeepASV/train_2xgpu.slurm`, `STAGE=s1|s2|s3`, 2×GPU, torchrun).
  Slurm scratch is `/scratch/{job_id}` and persists only ~5 days after the job ends.
- pip/conda installs can be slow (solver); run them serially, never concurrently.

## Config files

- `recipes/DeepASV/conf/w2v-bert/{s1,s2,s3}.yaml` use hyperpyyaml custom tags
  (`!apply:`, `!new:`, `!ref`, `!name`). Plain `yaml.safe_load` fails on them.
- Do NOT fully load these configs for validation — it instantiates the ~600M-param
  w2v-bert-2.0 model. Use a tag-tolerant loader, or assert on raw text/keys.

## Checkpointing

- All checkpoint logic lives in `deeplab/core/trainer.py` (shared Trainer).
- Epoch checkpoints contain: `modules`, `epoch_idx`, `optimizer`, `scheduler`,
  `scheduler_lmft`, `amp_scaler`, `rng`.
- `--resume <ckpt>` restores full training state (all ranks); `--pretrain <ckpt>`
  loads weights only. Rank 0 saves; restore is not rank-gated (intentional).
- This repo runs PyTorch 2.6: `torch.load` defaults to `weights_only=True`, so
  checkpoints must contain only tensors/Python primitives — no pickled numpy arrays
  (RNG state is stored as tensors + ints for this reason).

## wandb

- Enabled per-config via a `wandb_cfgs` block in the YAML (`watch`, `log`, `log_freq`).
  System metrics need `nvidia-ml-py` installed; `WANDB_API_KEY` is exported from
  `~/.bashrc`. `running_loss` logging is currently coupled to `watch: true`.

## Data paths (fixed, asserted by the slurm script)

- VoxCeleb2 webdataset index: `/scratch/48054831/voxceleb2-dev-wds/voxceleb2.index.sqlite3`
- VoxCeleb1 trials: `/scratch/48054834/voxceleb1/protocols/vox1-o/`
- Model: `~/voicegeneration/models/facebook/w2v-bert-2.0/model.safetensors`
- Noise: `~/voicegeneration/musan.tar`, `~/voicegeneration/RIRS_NOISES/`
- Slurm logs: `~/voicegeneration/w2vbert_sv_train/`

## Validation practice

- CPU box: validate with targeted round-trip tests (e.g. construct Trainer via
  `Trainer.__new__` to bypass I/O), plus `python -m py_compile`. CUDA branches stay
  guarded and are smoke-tested on the cluster, not locally.
