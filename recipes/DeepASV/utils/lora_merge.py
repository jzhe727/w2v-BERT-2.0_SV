import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch

from deeplab.pretrained.audio2vector.api import create_lora_config
from local.spk_model import Audio2Vec_based_Adapter


def merge_lora(checkpoint_path, output_path, model_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    model_path = str(Path(model_path).resolve())

    peft_config = create_lora_config(
        model_type="w2v-bert",
        r=64,
        lora_alpha=128,
        target_modules=["linear_q", "linear_v"],
        lora_dropout=0.0,
        bias="none",
    )
    model = Audio2Vec_based_Adapter(
        model_name=model_path,
        frozen_encoder=True,
        n_mfa_layers=-1,
        pooling_layer="ASP",
        peft_config=peft_config,
        encoder_config="config_prune_tea.json",
        embd_dim=256,
        adapter_dim=128,
        dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["modules"]["spk_model"], strict=True)
    model.front.encoder = model.front.encoder.merge_and_unload()
    checkpoint["modules"]["spk_model"] = model.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Merge Stage 1 LoRA weights for full fine-tuning")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-path",
        default="/home/john.zheng1/voicegeneration/models/facebook/w2v-bert-2.0",
    )
    args = parser.parse_args()
    merge_lora(args.checkpoint, args.output, args.model_path)
    print(f"Merged LoRA checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()