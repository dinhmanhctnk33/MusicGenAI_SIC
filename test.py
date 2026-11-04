import argparse
import os
import shutil
import subprocess
import tempfile

import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoProcessor, EncodecModel, T5EncoderModel, T5Tokenizer

from config import (
    CHECKPOINT_DIR,
    DATA_CONFIG,
    GENERATION_CONFIG,
    MODEL_CONFIG,
    build_run_config,
)
from model import MusicGenTransformer


class MusicGenTester:
    """Generate audio from a text prompt using a trained checkpoint."""

    def __init__(
        self,
        checkpoint_path,
        t5_model_name=DATA_CONFIG["t5_model_name"],
        encodec_model_name=DATA_CONFIG["encodec_model_name"],
        device=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        print(f"-> Device: {self.device}")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {os.path.abspath(checkpoint_path)}"
            )
        print(f"-> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")

        saved_config = checkpoint.get("config", {})
        model_config = {
            key: saved_config.get(key, default)
            for key, default in MODEL_CONFIG.items()
        }
        if not saved_config:
            print("-> Warning: checkpoint has no config; using training defaults.")

        print("-> Loading T5 tokenizer and encoder...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name).to(self.device).eval()

        print("-> Loading EnCodec processor and decoder...")
        self.encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
        self.encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(self.device).eval()

        print(f"-> Model config: {model_config}")
        self.model = MusicGenTransformer(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_seq_len = model_config["max_seq_len"]
        print("-> Checkpoint loaded successfully.")

    @torch.inference_mode()
    def generate_music(
        self,
        prompt,
        output_path=GENERATION_CONFIG["output_path"],
        duration_sec=GENERATION_CONFIG["duration_sec"],
        temperature=GENERATION_CONFIG["temperature"],
        top_k=GENERATION_CONFIG["top_k"],
        guidance_scale=GENERATION_CONFIG["guidance_scale"],
    ):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        if duration_sec <= 0:
            raise ValueError("duration_sec must be greater than zero.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative (0 disables filtering).")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be at least 1.0.")

        frames_per_second = GENERATION_CONFIG["frames_per_second"]
        max_steps = round(duration_sec * frames_per_second)
        max_allowed_steps = self.max_seq_len - self.model.num_codebooks + 1
        if max_steps > max_allowed_steps:
            max_duration = max_allowed_steps / frames_per_second
            raise ValueError(
                f"Maximum duration is about {max_duration:.2f}s "
                f"for max_seq_len={self.max_seq_len}."
            )

        print(f"\n[1/3] Processing prompt: {prompt!r}")
        text_inputs = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=DATA_CONFIG["text_max_length"],
            truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_embeddings = self.t5_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        print(f"[2/3] Generating {max_steps} token steps ({duration_sec:.2f}s)...")
        generated_tokens = self.model.generate(
            text_embeddings=text_embeddings,
            text_attention_mask=attention_mask,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=guidance_scale,
        )

        print("[3/3] Decoding tokens to waveform...")
        # EnCodec expects [num_frames, batch, num_codebooks, frame_length].
        audio_codes = generated_tokens.unsqueeze(0)
        waveform = self.encodec_model.decode(audio_codes, [None]).audio_values
        waveform = waveform[0].detach().float().cpu().transpose(0, 1).numpy()

        # EnCodec có thể trả về tín hiệu khá nhỏ. Khử DC, nâng RMS lên
        # khoảng -18 dBFS và giới hạn peak ở -1 dBFS để nghe rõ mà không vỡ tiếng.
        waveform = waveform - waveform.mean(axis=0, keepdims=True)
        rms = float((waveform ** 2).mean() ** 0.5)
        peak = float(abs(waveform).max())
        if rms > 1e-7 and peak > 1e-7:
            target_rms = 10 ** (-18 / 20)
            peak_limit = 10 ** (-1 / 20)
            gain = min(target_rms / rms, peak_limit / peak, 20.0)
            waveform = waveform * gain
            print(
                f"-> Loudness normalized: {20 * torch.log10(torch.tensor(gain)).item():+.1f} dB"
            )
        else:
            print("-> Warning: decoded waveform is nearly silent; normalization skipped.")

        output_path = os.path.abspath(output_path)
        extension = os.path.splitext(output_path)[1].lower()
        if extension not in {".wav", ".mp3"}:
            raise ValueError("output_path must have a .wav or .mp3 extension.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if extension == ".wav":
            sf.write(
                output_path,
                waveform,
                DATA_CONFIG["sample_rate"],
                subtype="PCM_16",
            )
            print(f"-> Audio saved to: {output_path}")
            return output_path

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg is required to create MP3 files. "
                "Use an output path ending in .wav, or install FFmpeg."
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            sf.write(
                temp_path,
                waveform,
                DATA_CONFIG["sample_rate"],
                subtype="PCM_16",
            )
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    temp_path,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    output_path,
                ],
                check=True,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"-> Audio saved to: {output_path}")
        return output_path


def evaluate_test_set(checkpoint_path, device=None):
    """Evaluate a saved checkpoint on the deterministic held-out test split."""
    from train import _create_data_loaders, validate

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.abspath(checkpoint_path)}"
        )

    print(f"-> Device: {device}")
    print(f"-> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_config = checkpoint.get("config", {})
    config = build_run_config()
    config.update(saved_config)
    model_config = {
        key: config.get(key, default) for key, default in MODEL_CONFIG.items()
    }

    print("-> Recreating the original deterministic dataset split...")
    _, _, test_loader = _create_data_loaders(config)
    print(f"-> Test samples: {len(test_loader.dataset)}")

    model = MusicGenTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    print(f"-> Evaluating checkpoint from epoch {checkpoint.get('epoch', '?')}...")
    test_loss, codebook_losses = validate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(
        "Test loss/codebook: "
        + ", ".join(
            f"CB{k}={value:.4f}"
            for k, value in enumerate(codebook_losses.tolist())
        )
    )
    return test_loss, codebook_losses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music from a MusicGen checkpoint."
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "generate"],
        default="evaluate",
        help="Evaluate the held-out test set (default) or generate audio.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            CHECKPOINT_DIR / DATA_CONFIG["run_name"] / "best_musicgen_model.pt"
        ),
        help="Checkpoint .pt path (mounted Google Drive paths are supported).",
    )
    parser.add_argument(
        "--prompt",
        default=GENERATION_CONFIG["prompt"],
    )
    parser.add_argument("--output", default=GENERATION_CONFIG["output_path"])
    parser.add_argument(
        "--duration", type=float, default=GENERATION_CONFIG["duration_sec"]
    )
    parser.add_argument(
        "--temperature", type=float, default=GENERATION_CONFIG["temperature"]
    )
    parser.add_argument("--top-k", type=int, default=GENERATION_CONFIG["top_k"])
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=GENERATION_CONFIG["guidance_scale"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def run_test():
    args = parse_args()
    if args.mode == "evaluate":
        evaluate_test_set(args.checkpoint, device=args.device)
        return
    tester = MusicGenTester(args.checkpoint, device=args.device)
    tester.generate_music(
        prompt=args.prompt,
        output_path=args.output,
        duration_sec=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    run_test()
