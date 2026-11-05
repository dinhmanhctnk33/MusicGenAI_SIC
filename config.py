
"""Cấu hình tập trung cho toàn bộ pipeline MusicGenAI."""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CODEC_CHECKPOINT_DIR = CHECKPOINT_DIR / "neural_codec_v1"


def recommended_encoder_device() -> str:
    """Chọn CUDA cho GPU hạng 4 GB trở lên; encoder chạy FP16."""
    override = os.environ.get("MUSICGEN_ENCODER_DEVICE")
    if override:
        if override not in {"cpu", "cuda"}:
            raise ValueError("MUSICGEN_ENCODER_DEVICE phải là 'cpu' hoặc 'cuda'.")
        if override == "cuda" and not torch.cuda.is_available():
            raise ValueError("MUSICGEN_ENCODER_DEVICE=cuda nhưng CUDA không khả dụng.")
        return override
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        # Driver thường báo thấp hơn dung lượng danh nghĩa một chút.
        if total_vram >= int(3.5 * 1024**3):
            return "cuda"
    return "cpu"


DATA_CONFIG = {
    "dataset_dir": str(DATASET_DIR),
    "subset": "small",
    "sample_rate": 32_000,
    "audio_duration": 5.0,
    "segments_per_track": 6,
    "cache_batch_size": 8,
    # Sau khi precompute cache, cac worker chi doc tensor nho tu dia.
    # Gioi han 4 de tranh ton RAM/IO qua muc tren Windows va o OneDrive.
    "num_workers": min(4, max(0, (os.cpu_count() or 2) - 1)),
    "prefetch_factor": 2,
    "minimum_audio_duration": 1.0,
    "audio_token_rate": 50,
    "text_max_length": 64,
    "text_vocab_size": 259,
    "min_audio_rms": 1e-4,
    "max_prompt_field_length": 200,
    "cache_version": "v5_neural_codec_bilingual",
    "run_name": "neural_codec_bilingual_v5",
    "use_encoded_cache": True,
    "precompute_cache": True,
    "encoder_device": recommended_encoder_device(),
}

MODEL_CONFIG = {
    "codebook_size": 1_024,
    # 8 x 1024 code o 50 Hz = 4 kbps; du bang thong cho nhac hon codec 4 codebook.
    "num_codebooks": 8,
    "text_dim": 384,
    "text_num_layers": 4,
    "text_nhead": 8,
    "text_dim_feedforward": 1_024,
    "d_model": 384,
    "num_layers": 6,
    "nhead": 8,
    "dim_feedforward": 1_536,
    "dropout": 0.1,
    "max_seq_len": 512,
}

CODEC_CONFIG = {
    "latent_dim": 128,
    "num_codebooks": MODEL_CONFIG["num_codebooks"],
    "codebook_size": MODEL_CONFIG["codebook_size"],
    "base_channels": 32,
}

CODEC_TRAIN_CONFIG = {
    "audio_duration": 2.0,
    "batch_size": 2,
    "epochs": 50,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "commitment_weight": 1.0,
    "stft_weight": 1.0,
    "waveform_weight": 1.0,
    "validation_ratio": 0.02,
    "early_stopping_patience": 7,
}

TRAIN_CONFIG = {
    "batch_size": 1,
    "epochs": 30,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-5,
    "weight_decay": 1e-2,
    "gradient_accumulation_steps": 16,
    "gradient_clip_norm": 1.0,
    "validation_ratio": 0.05,
    "test_ratio": 0.05,
    "early_stopping_patience": 5,
    "cfg_drop_prob": 0.1,
    "random_seed": 42,
    "log_every_batches": 50,
}

GENERATION_CONFIG = {
    "duration_sec": 5.0,
    "temperature": 0.8,
    "top_k": 50,
    "guidance_scale": 2.5,
    "frames_per_second": 50,
    "output_path": "generated_sample.wav",
    "prompt": (
        "A joyful upbeat Rock track with electric guitar and energetic drums."
    ),
}

ARCHITECTURE_KEYS = tuple(MODEL_CONFIG.keys())


def validate_config() -> None:
    """Kiểm tra các ràng buộc chéo trước khi tạo cache/train."""
    split_ratio = TRAIN_CONFIG["validation_ratio"] + TRAIN_CONFIG["test_ratio"]
    if not 0 < split_ratio < 1:
        raise ValueError("Tổng validation_ratio + test_ratio phải nằm trong (0, 1).")
    if TRAIN_CONFIG["batch_size"] < 1:
        raise ValueError("batch_size phải >= 1.")
    if TRAIN_CONFIG["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps phải >= 1.")
    if DATA_CONFIG["segments_per_track"] < 1:
        raise ValueError("segments_per_track phải >= 1.")
    if not 0 <= TRAIN_CONFIG["cfg_drop_prob"] < 1:
        raise ValueError("cfg_drop_prob phải nằm trong [0, 1).")
    if TRAIN_CONFIG["min_learning_rate"] > TRAIN_CONFIG["learning_rate"]:
        raise ValueError("min_learning_rate không được lớn hơn learning_rate.")
    if MODEL_CONFIG["d_model"] % MODEL_CONFIG["nhead"] != 0:
        raise ValueError("d_model phải chia hết cho nhead.")
    if MODEL_CONFIG["codebook_size"] > 32_768:
        raise ValueError("codebook_size vượt miền lưu cache int16.")
    if GENERATION_CONFIG["guidance_scale"] < 1.0:
        raise ValueError("guidance_scale phải >= 1.0.")

    audio_steps = round(
        DATA_CONFIG["audio_duration"] * DATA_CONFIG["audio_token_rate"]
    )
    delayed_steps = audio_steps + MODEL_CONFIG["num_codebooks"] - 1
    if delayed_steps > MODEL_CONFIG["max_seq_len"]:
        raise ValueError(
            f"Delay pattern cần {delayed_steps} bước, "
            f"vượt max_seq_len={MODEL_CONFIG['max_seq_len']}."
        )
    generation_steps = round(
        GENERATION_CONFIG["duration_sec"]
        * GENERATION_CONFIG["frames_per_second"]
    )
    max_generation_steps = (
        MODEL_CONFIG["max_seq_len"] - MODEL_CONFIG["num_codebooks"] + 1
    )
    if generation_steps > max_generation_steps:
        raise ValueError(
            f"Generation cần {generation_steps} bước, "
            f"tối đa là {max_generation_steps}."
        )


def build_run_config() -> dict:
    """Trả về config phẳng, serializable để lưu cùng checkpoint."""
    validate_config()
    return {**DATA_CONFIG, **MODEL_CONFIG, **TRAIN_CONFIG}


validate_config()
