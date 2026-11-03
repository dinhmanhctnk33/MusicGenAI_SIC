"""Cấu hình tập trung cho toàn bộ pipeline MusicGenAI."""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def recommended_encoder_device() -> str:
    """Chọn CUDA cho encoder khi VRAM >= 8 GB; cho phép override bằng env."""
    override = os.environ.get("MUSICGEN_ENCODER_DEVICE")
    if override:
        if override not in {"cpu", "cuda"}:
            raise ValueError("MUSICGEN_ENCODER_DEVICE phải là 'cpu' hoặc 'cuda'.")
        if override == "cuda" and not torch.cuda.is_available():
            raise ValueError("MUSICGEN_ENCODER_DEVICE=cuda nhưng CUDA không khả dụng.")
        return override
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        if total_vram >= 8 * 1024**3:
            return "cuda"
    return "cpu"


DATA_CONFIG = {
    "dataset_dir": str(DATASET_DIR),
    "subset": "small",
    "sample_rate": 32_000,
    "audio_duration": 5.0,
    "text_max_length": 64,
    "t5_model_name": "t5-base",
    "encodec_model_name": "facebook/encodec_32khz",
    "min_audio_rms": 1e-4,
    "max_prompt_field_length": 200,
    "cache_version": "v1",
    "use_encoded_cache": True,
    "precompute_cache": True,
    "encoder_device": recommended_encoder_device(),
}

MODEL_CONFIG = {
    "codebook_size": 2_048,
    "num_codebooks": 4,
    "text_dim": 768,
    "d_model": 256,
    "num_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1_024,
    "dropout": 0.1,
    "max_seq_len": 512,
}

TRAIN_CONFIG = {
    "batch_size": 1,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "gradient_accumulation_steps": 8,
    "gradient_clip_norm": 1.0,
    "validation_ratio": 0.05,
    "test_ratio": 0.05,
    "early_stopping_patience": 4,
    "cfg_drop_prob": 0.1,
    "random_seed": 42,
    "log_every_batches": 10,
}

GENERATION_CONFIG = {
    "duration_sec": 5.0,
    "temperature": 1.0,
    "top_k": 250,
    "frames_per_second": 50,
    "output_path": "generated_sample.mp3",
    "prompt": (
        "A joyful upbeat Rock track with electric guitar and energetic drums."
    ),
}

ARCHITECTURE_KEYS = tuple(MODEL_CONFIG.keys())


def build_run_config() -> dict:
    """Trả về config phẳng, serializable để lưu cùng checkpoint."""
    return {**DATA_CONFIG, **MODEL_CONFIG, **TRAIN_CONFIG}
